from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neafe_v2.evaluation import concatenate_feature_sets, fit_and_evaluate, load_station_data
from neafe_v2.gwbp import GWBPConfig, GWBPSelector
from neafe_v2.pafa import PAFAConfig, PAFAFeatureGenerator


DEVELOPMENT_STATIONS = ["solar1", "solar2", "wind1", "wind2"]
BUDGETS = [60, 120, 240]
FEATURE_CAPS = [5, 10, 20]
ADAPTIVE_STRENGTHS = [0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only NE-AFE sensitivity analysis.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "results" / "neafe_v2_sensitivity",
    )
    parser.add_argument("--stations", nargs="+", default=DEVELOPMENT_STATIONS)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-iterations", type=int, default=80)
    return parser.parse_args()


def pafa_config(budget: int, allocation: str, strength: float, seed: int) -> PAFAConfig:
    return PAFAConfig(
        mode="physics",
        allocation=allocation,
        budget=budget,
        windows=(4, 8, 16),
        pilot_per_group=8,
        min_group_budget=4,
        adaptive_strength=strength,
        soft_constraints=False,
        random_state=seed,
    )


def gwbp_config(max_features: int, n_splits: int, seed: int) -> GWBPConfig:
    return GWBPConfig(
        n_splits=n_splits,
        calibration_fraction=0.08,
        validation_fraction=0.08,
        gap=16,
        max_features=max_features,
        score_mode="mean",
        correction_shrinkage=0.5,
        redundancy_gamma=0.0,
        min_positive_fraction=0.5,
        min_score=0.0,
        base_max_iter=60,
        base_max_leaf_nodes=15,
        random_state=seed,
    )


def _write_incremental(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _is_done(
    results: pd.DataFrame,
    station: str,
    experiment: str,
    allocation: str,
    budget: int,
    max_features: int,
    strength: float,
) -> bool:
    if results.empty:
        return False
    return (
        (results["station"] == station)
        & (results["experiment"] == experiment)
        & (results["allocation"] == allocation)
        & (results["budget"] == budget)
        & (results["max_features"] == max_features)
        & np.isclose(results["adaptive_strength"], strength)
    ).any()


def run(args: argparse.Namespace) -> pd.DataFrame:
    path = args.output_dir / "sensitivity_results.csv"
    results = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for station in args.stations:
        data = load_station_data(args.data_root, station)
        configurations = []
        for allocation, strength in (("uniform", 0.0), ("adaptive", 1.0)):
            for budget in BUDGETS:
                configurations.append(("budget_feature_grid", allocation, budget, strength, FEATURE_CAPS))
        for strength in ADAPTIVE_STRENGTHS:
            configurations.append(("adaptive_strength", "adaptive", 120, strength, [10]))

        generated = {}
        for experiment, allocation, budget, strength, feature_caps in configurations:
            pending = [
                cap
                for cap in feature_caps
                if not _is_done(
                    results, station, experiment, allocation, budget, cap, strength
                )
            ]
            if not pending:
                continue
            cache_key = (allocation, budget, strength)
            if cache_key not in generated:
                generation_start = time.perf_counter()
                generator = PAFAFeatureGenerator(
                    pafa_config(budget, allocation, strength, args.seed)
                )
                generator.fit(data.train_x, data.train_y)
                train_candidates = generator.transform(data.train_x)
                validation_candidates = generator.transform(
                    data.validation_x, history=data.train_x
                )
                generated[cache_key] = (
                    train_candidates,
                    validation_candidates,
                    time.perf_counter() - generation_start,
                    generator.diagnostics_["group_allocation"],
                )
            (
                train_candidates,
                validation_candidates,
                generation_seconds,
                group_allocation,
            ) = generated[cache_key]
            for max_features in pending:
                start = time.perf_counter()
                selector = GWBPSelector(
                    gwbp_config(max_features, args.n_splits, args.seed)
                )
                selector.fit(data.train_x, train_candidates, data.train_y)
                selected_train = selector.transform(train_candidates)
                selected_validation = selector.transform(validation_candidates)
                metrics, _ = fit_and_evaluate(
                    concatenate_feature_sets(data.train_x, selected_train),
                    data.train_y,
                    concatenate_feature_sets(data.validation_x, selected_validation),
                    data.validation_y,
                    random_state=args.seed,
                    max_iter=args.model_iterations,
                )
                row = {
                    "station": station,
                    "experiment": experiment,
                    "allocation": allocation,
                    "budget": budget,
                    "max_features": max_features,
                    "adaptive_strength": strength,
                    "selected_count": len(selector.selected_features_),
                    "generation_seconds": generation_seconds,
                    "selection_evaluation_seconds": time.perf_counter() - start,
                    "runtime_seconds": generation_seconds + time.perf_counter() - start,
                    "group_allocation": str(group_allocation),
                    **metrics,
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                _write_incremental(results, path)
                print(
                    f"[{station}] {experiment} {allocation} B={budget} M={max_features} "
                    f"a={strength:.2f}: nRMSE={metrics['nrmse']:.5f}",
                    flush=True,
                )
    return results


def summarize(results: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = results[results["experiment"] == "budget_feature_grid"]
    grid_summary = (
        grid.groupby(["allocation", "budget", "max_features"], as_index=False)
        .agg(
            mean_nrmse=("nrmse", "mean"),
            worst_nrmse=("nrmse", "max"),
            mean_selected=("selected_count", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["allocation", "mean_nrmse"])
    )
    strength = results[results["experiment"] == "adaptive_strength"]
    strength_summary = (
        strength.groupby("adaptive_strength", as_index=False)
        .agg(
            mean_nrmse=("nrmse", "mean"),
            worst_nrmse=("nrmse", "max"),
            mean_selected=("selected_count", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("adaptive_strength")
    )
    grid_summary.to_csv(output_dir / "budget_feature_summary.csv", index=False)
    strength_summary.to_csv(output_dir / "adaptive_strength_summary.csv", index=False)
    return grid_summary, strength_summary


def make_plots(
    grid_summary: pd.DataFrame,
    strength_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for axis, allocation in zip(axes, ["uniform", "adaptive"]):
        part = grid_summary[grid_summary["allocation"] == allocation]
        matrix = part.pivot(index="max_features", columns="budget", values="mean_nrmse")
        matrix = matrix.reindex(index=FEATURE_CAPS, columns=BUDGETS)
        image = axis.imshow(matrix.to_numpy(), cmap="YlGnBu_r", aspect="auto")
        axis.set_xticks(range(len(BUDGETS)), BUDGETS)
        axis.set_yticks(range(len(FEATURE_CAPS)), FEATURE_CAPS)
        axis.set_xlabel("Candidate budget B")
        axis.set_title(allocation.capitalize())
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                axis.text(column, row, f"{value:.4f}", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Feature cap M")
    figure.colorbar(image, ax=axes, label="Mean validation nRMSE", shrink=0.82)
    figure.subplots_adjust(left=0.09, right=0.90, bottom=0.16, top=0.88, wspace=0.18)
    figure.savefig(output_dir / "budget_feature_sensitivity.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.8, 3.6))
    axis.plot(
        strength_summary["adaptive_strength"],
        strength_summary["mean_nrmse"],
        marker="o",
        color="#167A72",
        label="Mean",
    )
    axis.plot(
        strength_summary["adaptive_strength"],
        strength_summary["worst_nrmse"],
        marker="s",
        color="#C2543D",
        label="Worst station",
    )
    axis.set_xlabel("Adaptive allocation strength")
    axis.set_ylabel("Validation nRMSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "adaptive_strength_sensitivity.png", dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run(args)
    grid_summary, strength_summary = summarize(results, args.output_dir)
    make_plots(grid_summary, strength_summary, args.output_dir)
    print("\nBest B/M configurations:\n", grid_summary.head(8).to_string(index=False))
    print("\nAdaptive strength summary:\n", strength_summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
