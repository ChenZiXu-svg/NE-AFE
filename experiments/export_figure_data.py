from __future__ import annotations

import argparse
import ctypes
import json
import re
import threading
import time
import sys
if sys.platform == "win32":
    from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

from neafe_v2.evaluation import (
    concatenate_feature_sets,
    load_station_data,
    make_forecaster,
    regression_metrics,
)
from neafe_v2.gwbp import GWBPConfig, GWBPSelector
from neafe_v2.pafa import PAFAConfig, PAFAFeatureGenerator


ALL_STATIONS = [*(f"solar{i}" for i in range(1, 9)), *(f"wind{i}" for i in range(1, 7))]
DEVELOPMENT_STATIONS = ["solar1", "solar2", "wind1", "wind2"]


def parse_args() -> argparse.Namespace:
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Export plotting data from NE-AFE experiments.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--full-results", type=Path, default=root / "results" / "neafe_v2_full_final"
    )
    parser.add_argument(
        "--mvp-results", type=Path, default=root / "results" / "neafe_v2_mvp_final_strict"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "figure_data"
    )
    parser.add_argument("--skip-runtime-refresh", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    return parser.parse_args()


def display_plant(station: str) -> str:
    match = re.fullmatch(r"(solar|wind)(\d+)", station.lower())
    if not match:
        return station
    return f"{match.group(1).capitalize()} {match.group(2)}"


def energy_type(station: str) -> str:
    return "solar" if station.lower().startswith("solar") else "wind"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def forecast_results(args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_csv(args.full_results / "model_results.csv")
    ranges = {}
    for station in ALL_STATIONS:
        data = load_station_data(args.data_root, station)
        ranges[station] = float(data.test_y.max() - data.test_y.min())
    output = pd.DataFrame(
        {
            "plant": source["station"].map(display_plant),
            "energy_type": source["station"].map(energy_type),
            "model": source["model"],
            "method": source["feature_set"],
            "seed": source["seed"].astype(int),
            "nrmse": source["nrmse"],
            "nmae": [
                mae / ranges[station] if ranges[station] > 0 else np.nan
                for mae, station in zip(source["mae"], source["station"])
            ],
            "mae": source["mae"],
            "r2": source["r2"],
            "split": "test",
        }
    )
    return output


def pafa_config(strategy: str) -> PAFAConfig:
    configurations = {
        "generic_uniform": ("generic", "uniform", True),
        "physics_hard_uniform": ("physics", "uniform", False),
        "physics_soft_uniform": ("physics", "uniform", True),
        "physics_soft_adaptive": ("physics", "adaptive", True),
    }
    mode, allocation, soft_constraints = configurations[strategy]
    return PAFAConfig(
        mode=mode,
        allocation=allocation,
        budget=120,
        windows=(4, 8, 16),
        pilot_per_group=8,
        min_group_budget=4,
        temperature=1.0,
        adaptive_strength=1.0,
        soft_constraints=soft_constraints,
        random_state=42,
    )


def final_gwbp_config(
    scoring: str = "mean",
    sequential: bool = True,
    redundancy: bool = False,
    max_features: int = 12,
) -> GWBPConfig:
    return GWBPConfig(
        n_splits=4,
        calibration_fraction=0.08,
        validation_fraction=0.08,
        gap=16,
        max_features=max_features,
        score_mode=scoring,
        stability_lambda=0.5,
        quantile=0.25,
        correction_shrinkage=0.5,
        redundancy_gamma=0.02 if redundancy else 0.0,
        min_positive_fraction=0.5,
        min_score=0.0,
        base_max_iter=60,
        base_max_leaf_nodes=15,
        random_state=42,
        sequential_updates=sequential,
    )


def pafa_ablation(args: argparse.Namespace) -> pd.DataFrame:
    official = pd.read_csv(args.mvp_results / "test_ablation_results.csv")
    method_map = {
        "raw": "raw",
        "generic_uniform+gwbp": "generic_uniform",
        "physics_hard_uniform+gwbp": "physics_hard_uniform",
        "physics_soft_uniform+gwbp": "physics_soft_uniform",
        "physics_soft_adaptive+gwbp": "physics_soft_adaptive",
    }
    official = official[official["method"].isin(method_map)].copy()
    timings = {}
    candidate_counts = {}
    for station in DEVELOPMENT_STATIONS:
        data = load_station_data(args.data_root, station)
        development_x = pd.concat([data.train_x, data.validation_x], ignore_index=True)
        development_y = pd.concat([data.train_y, data.validation_y], ignore_index=True)
        for strategy in method_map.values():
            if strategy == "raw":
                timings[(station, strategy)] = 0.0
                candidate_counts[(station, strategy)] = 0
                continue
            start = time.perf_counter()
            generator = PAFAFeatureGenerator(pafa_config(strategy))
            generator.fit(development_x, development_y)
            generator.transform(development_x)
            generator.transform(data.test_x, history=development_x)
            timings[(station, strategy)] = time.perf_counter() - start
            candidate_counts[(station, strategy)] = int(
                generator.diagnostics_["retained_count"]
            )
    rows = []
    for row in official.itertuples(index=False):
        strategy = method_map[row.method]
        rows.append(
            {
                "plant": display_plant(row.station),
                "generation_strategy": strategy,
                "scoring": "none" if strategy == "raw" else "mean",
                "nrmse": row.nrmse,
                "candidate_count": candidate_counts[(row.station, strategy)],
                "selected_count": int(row.selected_count),
                "generation_seconds": timings[(row.station, strategy)],
                "rmse": row.rmse,
                "mae": row.mae,
                "r2": row.r2,
                "total_runtime_seconds": row.runtime_seconds,
                "split": "test",
            }
        )
    return pd.DataFrame(rows)


def gwbp_ablation_and_candidates(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = [
        ("mean", True, False),
        ("mean", False, False),
        ("mean", True, True),
        ("mean_variance", True, False),
        ("quantile", True, False),
    ]
    rows = []
    candidate_rows = []
    for station in DEVELOPMENT_STATIONS:
        data = load_station_data(args.data_root, station)
        generator = PAFAFeatureGenerator(pafa_config("physics_hard_uniform"))
        generator.fit(data.train_x, data.train_y)
        train_candidates = generator.transform(data.train_x)
        validation_candidates = generator.transform(
            data.validation_x, history=data.train_x
        )
        for scoring, sequential, redundancy in variants:
            start = time.perf_counter()
            selector = GWBPSelector(
                final_gwbp_config(scoring, sequential, redundancy)
            )
            selector.fit(data.train_x, train_candidates, data.train_y)
            selection_seconds = time.perf_counter() - start
            selected_train = selector.transform(train_candidates)
            selected_validation = selector.transform(validation_candidates)
            model = make_forecaster(random_state=42, max_iter=80)
            model.fit(
                concatenate_feature_sets(data.train_x, selected_train), data.train_y
            )
            prediction = model.predict(
                concatenate_feature_sets(data.validation_x, selected_validation)
            )
            metrics = regression_metrics(data.validation_y, prediction)
            rows.append(
                {
                    "plant": display_plant(station),
                    "scoring": scoring,
                    "sequential": sequential,
                    "redundancy": redundancy,
                    "nrmse": metrics["nrmse"],
                    "selected_count": len(selector.selected_features_),
                    "candidate_count": train_candidates.shape[1],
                    "selection_seconds": selection_seconds,
                    "stop_reason": selector.stop_reason_,
                    "split": "validation",
                }
            )
            if scoring == "mean" and sequential and not redundancy:
                trace = selector.candidate_history_.copy()
                trace.insert(0, "plant", display_plant(station))
                trace.insert(1, "scoring", scoring)
                trace.insert(2, "sequential", sequential)
                trace.insert(3, "redundancy_enabled", redundancy)
                candidate_rows.extend(trace.to_dict(orient="records"))
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def parse_feature(feature: str) -> tuple[str, str, str, str]:
    match = re.fullmatch(r"PAFA::([^:]+)::([^()]+)\((.*)\)", feature)
    if not match:
        return feature, "unknown", "unknown", ""
    group, operator, arguments = match.groups()
    parts = [part.strip() for part in arguments.split(",")]
    variables = [part for part in parts if not part.startswith("w=")]
    expression = f"{operator}({arguments})"
    return expression, group, operator, ";".join(variables)


def selected_features(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for station in ALL_STATIONS:
        path = args.full_results / "feature_cache" / station / "selection_details.json"
        details = json.loads(path.read_text(encoding="utf-8"))
        for history in details["selection_history"]:
            feature = history["feature"]
            expression, group, operator, sources = parse_feature(feature)
            rows.append(
                {
                    "plant": display_plant(station),
                    "selection_step": int(history["iteration"]),
                    "feature_name": feature,
                    "expression": expression,
                    "operator_group": group,
                    "operator": operator,
                    "source_variables": sources,
                    "conditional_gain": history["mean_gain"],
                    "gain_std": history["gain_std"],
                    "positive_fraction": history["positive_fraction"],
                }
            )
    return pd.DataFrame(rows)


def selection_trace(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for station in ALL_STATIONS:
        path = args.full_results / "feature_cache" / station / "selection_details.json"
        details = json.loads(path.read_text(encoding="utf-8"))
        history = details["selection_history"]
        selected_count = len(history)
        terminal_reason = (
            "max_features_reached" if selected_count >= 12 else "no_positive_gain"
        )
        for fold_info in details["fold_losses"]:
            fold = int(fold_info["fold"])
            final_loss = float(fold_info["validation_rmse"]) ** 2
            reconstructed = []
            loss_after = final_loss
            for item in reversed(history):
                gain = float(item[f"fold_{fold}_gain"])
                denominator = 1.0 - gain
                loss_before = loss_after / denominator if abs(denominator) > 1e-15 else np.nan
                reconstructed.append((item, gain, loss_before, loss_after))
                loss_after = loss_before
            for item, gain, loss_before, loss_after in reversed(reconstructed):
                rows.append(
                    {
                        "plant": display_plant(station),
                        "fold": fold,
                        "step": int(item["iteration"]),
                        "selected_feature": item["feature"],
                        "validation_gain": gain,
                        "current_loss": loss_before,
                        "loss_after": loss_after,
                        "stop_reason": "",
                    }
                )
            rows.append(
                {
                    "plant": display_plant(station),
                    "fold": fold,
                    "step": selected_count + 1,
                    "selected_feature": "",
                    "validation_gain": np.nan,
                    "current_loss": final_loss,
                    "loss_after": final_loss,
                    "stop_reason": terminal_reason,
                }
            )
    return pd.DataFrame(rows)


if sys.platform == "win32":
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]


    def _windows_memory_mb() -> float:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if not success:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / (1024.0**2)


def current_memory_mb() -> float:
    """Current resident memory, with native Windows and optional Unix support."""
    if sys.platform == "win32":
        return _windows_memory_mb()
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("Runtime-memory export on Unix requires pip install psutil") from error
    return psutil.Process().memory_info().rss / (1024.0**2)


class MemoryMonitor:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = current_memory_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, current_memory_mb())

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.peak = max(self.peak, current_memory_mb())
        self._stop.set()
        self._thread.join()


def runtime_results(args: argparse.Namespace) -> pd.DataFrame:
    existing = args.output_dir / "05_runtime_results.csv"
    if args.skip_runtime_refresh and existing.exists():
        return pd.read_csv(existing)
    rows = []
    configurations = []
    for allocation, strength in (("uniform", 0.0), ("adaptive", 1.0)):
        for budget in (60, 120, 240):
            configurations.append(
                (f"physics_hard_{allocation}", allocation, budget, strength, (5, 10, 20))
            )
    for strength in (0.25, 0.5, 0.75, 1.0):
        configurations.append(
            (f"physics_hard_adaptive_a{strength:.2f}", "adaptive", 120, strength, (10,))
        )

    for station in DEVELOPMENT_STATIONS:
        data = load_station_data(args.data_root, station)
        generated = {}
        for method, allocation, budget, strength, feature_caps in configurations:
            key = (allocation, budget, strength)
            if key not in generated:
                config = PAFAConfig(
                    mode="physics",
                    allocation=allocation,
                    budget=budget,
                    windows=(4, 8, 16),
                    pilot_per_group=8,
                    min_group_budget=4,
                    adaptive_strength=strength,
                    soft_constraints=False,
                    random_state=42,
                )
                with MemoryMonitor() as memory:
                    start = time.perf_counter()
                    generator = PAFAFeatureGenerator(config)
                    generator.fit(data.train_x, data.train_y)
                    train_candidates = generator.transform(data.train_x)
                    validation_candidates = generator.transform(
                        data.validation_x, history=data.train_x
                    )
                    generation_seconds = time.perf_counter() - start
                generated[key] = (
                    train_candidates,
                    validation_candidates,
                    generation_seconds,
                    memory.peak,
                )
            (
                train_candidates,
                validation_candidates,
                generation_seconds,
                generation_peak,
            ) = generated[key]
            for max_features in feature_caps:
                with MemoryMonitor() as memory:
                    selector = GWBPSelector(
                        final_gwbp_config(max_features=max_features)
                    )
                    start = time.perf_counter()
                    selector.fit(data.train_x, train_candidates, data.train_y)
                    selection_seconds = time.perf_counter() - start
                    selected_train = selector.transform(train_candidates)
                    selected_validation = selector.transform(validation_candidates)
                    train_x = concatenate_feature_sets(data.train_x, selected_train)
                    validation_x = concatenate_feature_sets(
                        data.validation_x, selected_validation
                    )
                    model = make_forecaster(random_state=42, max_iter=80)
                    start = time.perf_counter()
                    model.fit(train_x, data.train_y)
                    training_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    prediction = model.predict(validation_x)
                    metrics = regression_metrics(data.validation_y, prediction)
                    evaluation_seconds = time.perf_counter() - start
                rows.append(
                    {
                        "plant": display_plant(station),
                        "method": method,
                        "B": budget,
                        "M": max_features,
                        "adaptive_strength": strength,
                        "nrmse": metrics["nrmse"],
                        "generation_seconds": generation_seconds,
                        "evaluation_seconds": evaluation_seconds,
                        "selection_seconds": selection_seconds,
                        "training_seconds": training_seconds,
                        "peak_memory_mb": max(generation_peak, memory.peak),
                        "candidate_count": train_candidates.shape[1],
                        "selected_count": len(selector.selected_features_),
                        "split": "validation",
                    }
                )
                print(
                    f"[runtime] {station} {method} B={budget} M={max_features}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime_only:
        frame = runtime_results(args)
        atomic_csv(frame, args.output_dir / "05_runtime_results.csv")
        print(f"05_runtime_results.csv: {len(frame)} rows", flush=True)
        return
    frames = {
        "01_forecast_results.csv": forecast_results(args),
        "02_pafa_ablation.csv": pafa_ablation(args),
    }
    gwbp, candidates = gwbp_ablation_and_candidates(args)
    frames["03_gwbp_ablation.csv"] = gwbp
    frames["04_selected_features.csv"] = selected_features(args)
    frames["05_runtime_results.csv"] = runtime_results(args)
    frames["06_selection_trace.csv"] = selection_trace(args)
    frames["07_candidate_scores.csv"] = candidates
    for filename, frame in frames.items():
        atomic_csv(frame, args.output_dir / filename)
        print(f"{filename}: {len(frame)} rows", flush=True)
    manifest = {
        filename: {
            "rows": len(frame),
            "columns": list(frame.columns),
        }
        for filename, frame in frames.items()
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
