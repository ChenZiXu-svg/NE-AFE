from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch as _torch  # Load before pandas on Windows; see neafe_v2.models.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neafe_v2.evaluation import concatenate_feature_sets, load_station_data
from neafe_v2.gwbp import GWBPConfig, GWBPSelector
from neafe_v2.models import MODEL_NAMES, ModelConfig, fit_and_evaluate_model
from neafe_v2.pafa import PAFAConfig, PAFAFeatureGenerator


ALL_STATIONS = [*(f"solar{i}" for i in range(1, 9)), *(f"wind{i}" for i in range(1, 7))]
DEVELOPMENT_STATIONS = {"solar1", "solar2", "wind1", "wind2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leakage-safe 14-station NE-AFE evaluation.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "results" / "neafe_v2_full",
    )
    parser.add_argument("--stations", nargs="+", default=ALL_STATIONS)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2003])
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def pafa_config(args: argparse.Namespace) -> PAFAConfig:
    return PAFAConfig(
        mode="physics",
        allocation="uniform",
        budget=args.budget,
        windows=(4, 8, 16),
        pilot_per_group=8,
        min_group_budget=4,
        adaptive_strength=0.0,
        soft_constraints=False,
        random_state=42,
    )


def gwbp_config(args: argparse.Namespace) -> GWBPConfig:
    return GWBPConfig(
        n_splits=args.n_splits,
        calibration_fraction=0.08,
        validation_fraction=0.08,
        gap=16,
        max_features=args.max_features,
        score_mode="mean",
        correction_shrinkage=0.5,
        redundancy_gamma=0.0,
        min_positive_fraction=0.5,
        min_score=0.0,
        base_max_iter=60,
        base_max_leaf_nodes=15,
        random_state=42,
    )


def model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        sequence_length=args.sequence_length,
        max_train_samples=args.max_train_samples,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        num_threads=args.threads,
    )


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_signature(args: argparse.Namespace) -> dict:
    return {"pafa": asdict(pafa_config(args)), "gwbp": asdict(gwbp_config(args))}


def prepare_station_features(
    station: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    cache_dir = args.output_dir / "feature_cache" / station
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature_path = cache_dir / "feature_config.json"
    development_path = cache_dir / "development_selected.feather"
    test_path = cache_dir / "test_selected.feather"
    details_path = cache_dir / "selection_details.json"
    signature = _feature_signature(args)
    data = load_station_data(args.data_root, station)
    development_x = pd.concat([data.train_x, data.validation_x], ignore_index=True)
    development_y = pd.concat([data.train_y, data.validation_y], ignore_index=True)

    cache_valid = False
    if signature_path.exists() and development_path.exists() and test_path.exists() and details_path.exists():
        cache_valid = json.loads(signature_path.read_text(encoding="utf-8")) == signature
    if cache_valid and not args.force_features:
        selected_development = pd.read_feather(development_path)
        selected_test = pd.read_feather(test_path)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        return (
            development_x,
            development_y,
            data.test_x,
            data.test_y,
            selected_development,
            selected_test,
            details,
        )

    start = time.perf_counter()
    generator = PAFAFeatureGenerator(pafa_config(args))
    generator.fit(development_x, development_y)
    development_candidates = generator.transform(development_x)
    test_candidates = generator.transform(data.test_x, history=development_x)
    selector = GWBPSelector(gwbp_config(args))
    selector.fit(development_x, development_candidates, development_y)
    selected_development = selector.transform(development_candidates).reset_index(drop=True)
    selected_test = selector.transform(test_candidates).reset_index(drop=True)
    details = {
        "station": station,
        "selected_features": selector.selected_features_,
        "selection_history": selector.history_.to_dict(orient="records"),
        "fold_losses": selector.fold_losses_.to_dict(orient="records"),
        "pafa_diagnostics": generator.diagnostics_,
        "gwbp_diagnostics": selector.diagnostics_,
        "runtime_seconds": time.perf_counter() - start,
    }
    selected_development.to_feather(development_path)
    selected_test.to_feather(test_path)
    signature_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    details_path.write_text(
        json.dumps(json_ready(details), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        development_x,
        development_y,
        data.test_x,
        data.test_y,
        selected_development,
        selected_test,
        details,
    )


def _seeds_for_model(model: str, seeds: List[int]) -> List[int]:
    return [seeds[0]] if model == "Ridge" else seeds


def _write_incremental(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_models(args: argparse.Namespace) -> pd.DataFrame:
    results_path = args.output_dir / "model_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame()
    config = model_config(args)
    for station in args.stations:
        station_start = time.perf_counter()
        (
            development_x,
            development_y,
            test_x,
            test_y,
            selected_development,
            selected_test,
            details,
        ) = prepare_station_features(station, args)
        feature_sets = {
            "Raw": (development_x, test_x),
            "NE-AFE": (
                concatenate_feature_sets(development_x, selected_development),
                concatenate_feature_sets(test_x, selected_test),
            ),
        }
        for model in args.models:
            for seed in _seeds_for_model(model, args.seeds):
                for feature_set, (train_x, evaluation_x) in feature_sets.items():
                    if not results.empty:
                        done = (
                            (results["station"] == station)
                            & (results["model"] == model)
                            & (results["feature_set"] == feature_set)
                            & (results["seed"] == seed)
                        ).any()
                        if done:
                            continue
                    metrics, _ = fit_and_evaluate_model(
                        model,
                        train_x,
                        development_y,
                        evaluation_x,
                        test_y,
                        seed,
                        config=config,
                    )
                    row = {
                        "station": station,
                        "station_type": "solar" if station.startswith("solar") else "wind",
                        "protocol_group": (
                            "development" if station in DEVELOPMENT_STATIONS else "extension"
                        ),
                        "feature_set": feature_set,
                        "selected_count": len(details["selected_features"]),
                        **metrics,
                    }
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    _write_incremental(results, results_path)
                    print(
                        f"[{station}] {model} seed={seed} {feature_set}: "
                        f"nRMSE={float(metrics['nrmse']):.5f} "
                        f"({float(metrics['runtime_seconds']):.1f}s)",
                        flush=True,
                    )
        print(
            f"[{station}] complete in {time.perf_counter() - station_start:.1f}s; "
            f"selected={len(details['selected_features'])}",
            flush=True,
        )
    return results


def summarize(results: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    station_summary = (
        results.groupby(
            ["station", "station_type", "protocol_group", "model", "feature_set"],
            as_index=False,
        )
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    model_summary = (
        station_summary.groupby(["model", "feature_set"], as_index=False)
        .agg(
            mean_nrmse=("nrmse_mean", "mean"),
            std_nrmse=("nrmse_mean", "std"),
            worst_nrmse=("nrmse_mean", "max"),
            mean_runtime_seconds=("runtime_mean", "mean"),
        )
        .sort_values(["model", "feature_set"])
    )
    paired = station_summary.pivot_table(
        index=["station", "station_type", "protocol_group", "model"],
        columns="feature_set",
        values="nrmse_mean",
    ).reset_index()
    paired["improvement_percent"] = 100.0 * (paired["Raw"] - paired["NE-AFE"]) / paired["Raw"]
    paired_summary = (
        paired.groupby("model", as_index=False)
        .agg(
            mean_improvement_percent=("improvement_percent", "mean"),
            median_improvement_percent=("improvement_percent", "median"),
            improved_stations=("improvement_percent", lambda values: int((values > 0).sum())),
            stations=("station", "nunique"),
        )
        .sort_values("mean_improvement_percent", ascending=False)
    )
    station_summary.to_csv(output_dir / "station_model_summary.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_improvements.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_improvement_summary.csv", index=False)
    return {
        "station": station_summary,
        "model": model_summary,
        "paired": paired,
        "paired_summary": paired_summary,
    }


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def make_plots(summaries: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    _paper_style()
    model_summary = summaries["model"]
    pivot = model_summary.pivot(index="model", columns="feature_set", values="mean_nrmse")
    pivot = pivot.reindex([model for model in MODEL_NAMES if model in pivot.index])
    axis = pivot[["Raw", "NE-AFE"]].plot(
        kind="bar", figsize=(7.2, 3.8), color=["#707070", "#167A72"], width=0.72
    )
    axis.set_ylabel("Mean nRMSE")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.figure.tight_layout()
    axis.figure.savefig(output_dir / "raw_vs_neafe_by_model.png", dpi=300)
    plt.close(axis.figure)

    runtime = model_summary.copy()
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    colors = {"Raw": "#707070", "NE-AFE": "#C2543D"}
    markers = {"Raw": "o", "NE-AFE": "s"}
    for feature_set, part in runtime.groupby("feature_set"):
        axis.scatter(
            part["mean_runtime_seconds"],
            part["mean_nrmse"],
            c=colors[feature_set],
            marker=markers[feature_set],
            s=48,
            label=feature_set,
        )
        for row in part.itertuples():
            axis.annotate(row.model, (row.mean_runtime_seconds, row.mean_nrmse), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Mean model runtime (s)")
    axis.set_ylabel("Mean nRMSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "runtime_accuracy_pareto.png", dpi=300)
    plt.close(figure)


def feature_distribution(stations: Iterable[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    for station in stations:
        path = output_dir / "feature_cache" / station / "selection_details.json"
        if not path.exists():
            continue
        details = json.loads(path.read_text(encoding="utf-8"))
        for feature in details["selected_features"]:
            parts = feature.split("::", 2)
            group = parts[1] if len(parts) > 1 else "unknown"
            operator = parts[2].split("(", 1)[0] if len(parts) > 2 else "unknown"
            rows.append({"station": station, "group": group, "operator": operator, "feature": feature})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "selected_feature_distribution.csv", index=False)
    if frame.empty:
        return frame
    counts = frame.groupby("operator").size().sort_values()
    figure, axis = plt.subplots(figsize=(7.2, max(3.6, 0.3 * len(counts))))
    counts.plot(kind="barh", ax=axis, color="#3B6C9E")
    axis.set_xlabel("Selections across stations")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "selected_operator_distribution.png", dpi=300)
    plt.close(figure)
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "method": "Physics-hard uniform PAFA + GWBP Mean",
            "pafa": asdict(pafa_config(args)),
            "gwbp": asdict(gwbp_config(args)),
            "models": asdict(model_config(args)),
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = run_models(args)
    summaries = summarize(results, args.output_dir)
    make_plots(summaries, args.output_dir)
    feature_distribution(args.stations, args.output_dir)
    print("\nPaired improvement summary:\n", summaries["paired_summary"].to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
