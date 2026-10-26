from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neafe_v2.evaluation import (
    StationData,
    concatenate_feature_sets,
    fit_and_evaluate,
    load_station_data,
)
from neafe_v2.gwbp import GWBPConfig, GWBPSelector
from neafe_v2.pafa import PAFAConfig, PAFAFeatureGenerator


SCORERS = ("mean", "mean_variance", "quantile", "mean_variance_redundancy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-station NE-AFE v2 MVP.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "results" / "neafe_v2_mvp")
    parser.add_argument("--stations", nargs="+", default=["solar1", "solar2", "wind1", "wind2"])
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--model-iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def pafa_config(
    args: argparse.Namespace,
    mode: str,
    allocation: str,
    soft_constraints: bool = True,
) -> PAFAConfig:
    budget = min(args.budget, 80) if args.quick else args.budget
    return PAFAConfig(
        mode=mode,
        allocation=allocation,
        budget=budget,
        windows=(4, 8, 16),
        pilot_per_group=4 if args.quick else 8,
        min_group_budget=2 if args.quick else 4,
        temperature=1.0,
        soft_constraints=soft_constraints,
        random_state=args.seed,
    )


def gwbp_config(
    args: argparse.Namespace,
    scorer: str,
    force_redundancy: bool = False,
) -> GWBPConfig:
    mode = "mean_variance" if scorer == "mean_variance_redundancy" else scorer
    redundancy = 0.02 if scorer == "mean_variance_redundancy" or force_redundancy else 0.0
    return GWBPConfig(
        n_splits=args.n_splits,
        calibration_fraction=0.08,
        validation_fraction=0.08,
        gap=16,
        max_features=min(args.max_features, 8) if args.quick else args.max_features,
        score_mode=mode,
        stability_lambda=0.5,
        quantile=0.25,
        correction_shrinkage=0.5,
        redundancy_gamma=redundancy,
        min_positive_fraction=0.5,
        min_score=0.0,
        base_max_iter=30 if args.quick else 60,
        base_max_leaf_nodes=15,
        random_state=args.seed,
    )


def causal_candidate_sets(
    generator: PAFAFeatureGenerator,
    train_x: pd.DataFrame,
    evaluation_x: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_candidates = generator.transform(train_x)
    evaluation_candidates = generator.transform(evaluation_x, history=train_x)
    return train_candidates, evaluation_candidates


def evaluate_selected_features(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    evaluation_x: pd.DataFrame,
    evaluation_y: pd.Series,
    train_candidates: pd.DataFrame,
    evaluation_candidates: pd.DataFrame,
    selector: GWBPSelector,
    args: argparse.Namespace,
) -> Dict[str, float]:
    selected_train = selector.transform(train_candidates)
    selected_evaluation = selector.transform(evaluation_candidates)
    metrics, _ = fit_and_evaluate(
        concatenate_feature_sets(train_x, selected_train),
        train_y,
        concatenate_feature_sets(evaluation_x, selected_evaluation),
        evaluation_y,
        random_state=args.seed,
        max_iter=40 if args.quick else args.model_iterations,
    )
    metrics["selected_count"] = len(selector.selected_features_)
    return metrics


def validation_scorer_experiment(
    station_data: Dict[str, StationData],
    args: argparse.Namespace,
    selected_features: Dict[str, dict],
    diagnostics: Dict[str, dict],
) -> pd.DataFrame:
    rows: List[dict] = []
    for station, data in station_data.items():
        start = time.perf_counter()
        generator = PAFAFeatureGenerator(
            pafa_config(args, "physics", "adaptive", soft_constraints=True)
        )
        generator.fit(data.train_x, data.train_y)
        train_candidates, validation_candidates = causal_candidate_sets(
            generator, data.train_x, data.validation_x
        )
        diagnostics[f"validation/{station}/pafa"] = generator.diagnostics_

        for scorer in SCORERS:
            selector = GWBPSelector(gwbp_config(args, scorer))
            selector.fit(data.train_x, train_candidates, data.train_y)
            metrics = evaluate_selected_features(
                data.train_x,
                data.train_y,
                data.validation_x,
                data.validation_y,
                train_candidates,
                validation_candidates,
                selector,
                args,
            )
            rows.append(
                {
                    "phase": "validation",
                    "station": station,
                    "method": scorer,
                    **metrics,
                }
            )
            selected_features[f"validation/{station}/{scorer}"] = {
                "features": selector.selected_features_,
                "history": selector.history_.to_dict(orient="records"),
            }
        print(f"[validation] {station} completed in {time.perf_counter() - start:.1f}s", flush=True)
    return pd.DataFrame(rows)


def choose_scorer(validation_results: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    summary = (
        validation_results.groupby("method", as_index=False)
        .agg(
            mean_nrmse=("nrmse", "mean"),
            worst_nrmse=("nrmse", "max"),
            mean_selected=("selected_count", "mean"),
        )
        .sort_values(["mean_nrmse", "worst_nrmse", "mean_selected"])
    )
    return str(summary.iloc[0]["method"]), summary


def fit_pafa_and_candidates(
    development_x: pd.DataFrame,
    development_y: pd.Series,
    test_x: pd.DataFrame,
    config: PAFAConfig,
) -> Tuple[PAFAFeatureGenerator, pd.DataFrame, pd.DataFrame]:
    generator = PAFAFeatureGenerator(config)
    generator.fit(development_x, development_y)
    development_candidates, test_candidates = causal_candidate_sets(
        generator, development_x, test_x
    )
    return generator, development_candidates, test_candidates


def final_ablation_experiment(
    station_data: Dict[str, StationData],
    chosen_scorer: str,
    args: argparse.Namespace,
    selected_features: Dict[str, dict],
    diagnostics: Dict[str, dict],
) -> pd.DataFrame:
    rows: List[dict] = []
    for station, data in station_data.items():
        start = time.perf_counter()
        development_x = pd.concat([data.train_x, data.validation_x], ignore_index=True)
        development_y = pd.concat([data.train_y, data.validation_y], ignore_index=True)

        raw_metrics, _ = fit_and_evaluate(
            development_x,
            development_y,
            data.test_x,
            data.test_y,
            random_state=args.seed,
            max_iter=40 if args.quick else args.model_iterations,
        )
        rows.append(
            {
                "phase": "test",
                "station": station,
                "method": "raw",
                **raw_metrics,
                "selected_count": 0,
                "runtime_seconds": 0.0,
            }
        )

        generators = {}
        for label, mode, allocation, soft_constraints in (
            ("generic_uniform", "generic", "uniform", True),
            ("physics_hard_uniform", "physics", "uniform", False),
            ("physics_soft_uniform", "physics", "uniform", True),
            ("physics_soft_adaptive", "physics", "adaptive", True),
        ):
            generation_start = time.perf_counter()
            generator, development_candidates, test_candidates = fit_pafa_and_candidates(
                development_x,
                development_y,
                data.test_x,
                pafa_config(
                    args,
                    mode,
                    allocation,
                    soft_constraints=soft_constraints,
                ),
            )
            generators[label] = (
                generator,
                development_candidates,
                test_candidates,
                time.perf_counter() - generation_start,
            )
            diagnostics[f"test/{station}/{label}/pafa"] = generator.diagnostics_

        configurations = [
            ("generic_uniform+gwbp", "generic_uniform", chosen_scorer, False),
            ("physics_hard_uniform+gwbp", "physics_hard_uniform", chosen_scorer, False),
            ("physics_soft_uniform+gwbp", "physics_soft_uniform", chosen_scorer, False),
            ("physics_soft_adaptive+mean", "physics_soft_adaptive", "mean", False),
            ("physics_soft_adaptive+gwbp", "physics_soft_adaptive", chosen_scorer, False),
            (
                "physics_soft_adaptive+gwbp+redundancy",
                "physics_soft_adaptive",
                chosen_scorer,
                True,
            ),
        ]
        for method, generator_key, scorer, force_redundancy in configurations:
            generator, development_candidates, test_candidates, generation_time = generators[generator_key]
            selection_start = time.perf_counter()
            selector = GWBPSelector(
                gwbp_config(args, scorer, force_redundancy=force_redundancy)
            )
            selector.fit(development_x, development_candidates, development_y)
            metrics = evaluate_selected_features(
                development_x,
                development_y,
                data.test_x,
                data.test_y,
                development_candidates,
                test_candidates,
                selector,
                args,
            )
            rows.append(
                {
                    "phase": "test",
                    "station": station,
                    "method": method,
                    **metrics,
                    "runtime_seconds": generation_time + time.perf_counter() - selection_start,
                }
            )
            selected_features[f"test/{station}/{method}"] = {
                "features": selector.selected_features_,
                "history": selector.history_.to_dict(orient="records"),
            }
        print(f"[test] {station} completed in {time.perf_counter() - start:.1f}s", flush=True)
    return pd.DataFrame(rows)


def save_bar_plot(frame: pd.DataFrame, output: Path, title: str) -> None:
    summary = frame.groupby("method")["nrmse"].mean().sort_values(ascending=False)
    height = max(4.2, 0.58 * len(summary) + 1.4)
    figure, axis = plt.subplots(figsize=(10, height))
    summary.plot(kind="barh", ax=axis, color="#2f6f8f")
    axis.set_xlabel("Mean nRMSE")
    axis.set_ylabel("")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_data = {
        station: load_station_data(args.data_root, station) for station in args.stations
    }
    selected_features: Dict[str, dict] = {}
    diagnostics: Dict[str, dict] = {}

    validation = validation_scorer_experiment(
        station_data, args, selected_features, diagnostics
    )
    chosen_scorer, validation_summary = choose_scorer(validation)
    print(f"Selected GWBP scorer: {chosen_scorer}", flush=True)

    test = final_ablation_experiment(
        station_data, chosen_scorer, args, selected_features, diagnostics
    )
    test_summary = (
        test.groupby("method", as_index=False)
        .agg(
            mean_nrmse=("nrmse", "mean"),
            worst_nrmse=("nrmse", "max"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_selected=("selected_count", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("mean_nrmse")
    )

    validation.to_csv(args.output_dir / "validation_scorer_results.csv", index=False)
    validation_summary.to_csv(args.output_dir / "validation_scorer_summary.csv", index=False)
    test.to_csv(args.output_dir / "test_ablation_results.csv", index=False)
    test_summary.to_csv(args.output_dir / "test_ablation_summary.csv", index=False)
    (args.output_dir / "selected_features.json").write_text(
        json.dumps(json_ready(selected_features), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(json_ready(diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_config = {
        "arguments": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "chosen_scorer": chosen_scorer,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_bar_plot(
        validation,
        args.output_dir / "validation_scorer_comparison.png",
        "GWBP scorer selection on chronological validation sets",
    )
    save_bar_plot(
        test,
        args.output_dir / "test_ablation_comparison.png",
        "NE-AFE v2 module ablation on sealed test sets",
    )
    print("\nValidation scorer summary:\n", validation_summary.to_string(index=False))
    print("\nTest ablation summary:\n", test_summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
