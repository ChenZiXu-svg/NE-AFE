from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metadata import VariableMetadata, infer_metadata


@dataclass(frozen=True)
class CandidateSpec:
    group: str
    operator: str
    inputs: Tuple[str, ...]
    window: int = 0
    priority: float = 0.0

    @property
    def name(self) -> str:
        args = ",".join(self.inputs)
        suffix = f",w={self.window}" if self.window else ""
        return f"PAFA::{self.group}::{self.operator}({args}{suffix})"


@dataclass
class PAFAConfig:
    mode: str = "physics"
    allocation: str = "adaptive"
    budget: int = 160
    windows: Tuple[int, ...] = (4, 8, 16)
    pilot_per_group: int = 8
    min_group_budget: int = 4
    temperature: float = 1.0
    adaptive_strength: float = 1.0
    max_group_fraction: float = 0.45
    ridge: float = 1e-6
    max_nan_fraction: float = 0.25
    min_std: float = 1e-10
    soft_constraints: bool = True
    random_state: int = 42

    def validate(self) -> None:
        if self.mode not in {"generic", "physics"}:
            raise ValueError("mode must be 'generic' or 'physics'.")
        if self.allocation not in {"uniform", "adaptive"}:
            raise ValueError("allocation must be 'uniform' or 'adaptive'.")
        if self.budget < 1:
            raise ValueError("budget must be positive.")
        if not 0 < self.max_group_fraction <= 1:
            raise ValueError("max_group_fraction must be in (0, 1].")
        if not 0 <= self.adaptive_strength <= 1:
            raise ValueError("adaptive_strength must be in [0, 1].")
        if any(window < 2 for window in self.windows):
            raise ValueError("All temporal windows must be at least two.")


class PAFAFeatureGenerator:
    """Physics-aware, budget-constrained and leakage-safe feature generator."""

    def __init__(self, config: Optional[PAFAConfig] = None):
        self.config = config or PAFAConfig()
        self.config.validate()

    def fit(self, data: pd.DataFrame, label: pd.Series) -> "PAFAFeatureGenerator":
        data = self._validate_data(data)
        label = pd.Series(label, index=data.index, dtype=float)
        self.input_columns_ = list(data.columns)
        self.metadata_ = infer_metadata(data)

        specs = self._build_candidate_specs()
        specs = list({spec.name: spec for spec in specs}.values())
        grouped = self._group_specs(specs)
        pilot_gains, pilot_costs = self._pilot_group_evaluation(data, label, grouped)
        allocation = self._allocate_budget(grouped, pilot_gains, pilot_costs)
        selected = self._select_specs(grouped, allocation, pilot_gains)

        train_features = self._transform_specs(data, selected)
        self.selected_specs_, train_features, filtering = self._filter_train_features(
            selected, train_features
        )
        self.feature_names_ = [spec.name for spec in self.selected_specs_]
        self.diagnostics_ = {
            "mode": self.config.mode,
            "allocation": self.config.allocation,
            "candidate_count_before_budget": len(specs),
            "allocated_count": len(selected),
            "retained_count": len(self.selected_specs_),
            "group_allocation": allocation,
            "pilot_gain": pilot_gains,
            "pilot_cost_seconds": pilot_costs,
            "filtering": filtering,
        }
        return self

    def fit_transform(self, data: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
        self.fit(data, label)
        return self.transform(data)

    def transform(self, data: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self._check_is_fitted()
        data = self._validate_data(data)
        missing = set(self.input_columns_) - set(data.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")

        if history is None or len(history) == 0:
            combined = data[self.input_columns_].reset_index(drop=True)
        else:
            history = self._validate_data(history)
            combined = pd.concat(
                [history[self.input_columns_], data[self.input_columns_]],
                axis=0,
                ignore_index=True,
            )
        transformed = self._transform_specs(combined, self.selected_specs_)
        transformed = transformed.iloc[-len(data):].copy()
        transformed.index = data.index
        return transformed

    def get_metadata_frame(self) -> pd.DataFrame:
        self._check_is_fitted()
        return pd.DataFrame([item.to_dict() for item in self.metadata_.values()])

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("PAFA expects a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input data cannot be empty.")
        converted = data.copy()
        for column in converted.columns:
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
        return converted

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "selected_specs_"):
            raise RuntimeError("PAFAFeatureGenerator must be fitted before transform().")

    def _build_candidate_specs(self) -> List[CandidateSpec]:
        if self.config.mode == "generic":
            return self._generic_specs()
        physics = self._physics_specs()
        if self.config.soft_constraints:
            physics.extend(
                CandidateSpec(
                    group=spec.group,
                    operator=spec.operator,
                    inputs=spec.inputs,
                    window=spec.window,
                    priority=spec.priority - 2.0,
                )
                for spec in self._generic_specs()
            )
        return physics

    def _generic_specs(self) -> List[CandidateSpec]:
        specs: List[CandidateSpec] = []
        columns = self.input_columns_
        for column in columns:
            for operator in ("square", "cube", "log1p_abs"):
                specs.append(CandidateSpec("nonlinear", operator, (column,), priority=1.0))
            for window in self.config.windows:
                for operator in ("lag", "delta", "rolling_mean", "rolling_std"):
                    specs.append(
                        CandidateSpec("temporal", operator, (column,), window=window, priority=1.0)
                    )
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                for operator in ("add", "subtract", "multiply", "safe_divide"):
                    specs.append(
                        CandidateSpec("interaction", operator, (left, right), priority=0.5)
                    )
                specs.append(
                    CandidateSpec("interaction", "safe_divide", (right, left), priority=0.5)
                )
        return specs

    def _physics_specs(self) -> List[CandidateSpec]:
        specs: List[CandidateSpec] = []
        metadata = self.metadata_
        for column, meta in metadata.items():
            if meta.periodic:
                specs.extend(
                    [
                        CandidateSpec("physical", "direction_sin", (column,), priority=5.0),
                        CandidateSpec("physical", "direction_cos", (column,), priority=5.0),
                    ]
                )
                for window in self.config.windows:
                    specs.extend(
                        [
                            CandidateSpec(
                                "temporal", "circular_mean_sin", (column,), window, priority=4.0
                            ),
                            CandidateSpec(
                                "temporal", "circular_mean_cos", (column,), window, priority=4.0
                            ),
                        ]
                    )
                continue

            if meta.group == "wind_speed":
                specs.extend(
                    [
                        CandidateSpec("physical", "square", (column,), priority=4.0),
                        CandidateSpec("physical", "cube", (column,), priority=5.0),
                    ]
                )
            elif meta.group == "irradiance":
                specs.extend(
                    [
                        CandidateSpec("physical", "square", (column,), priority=3.0),
                        CandidateSpec("nonlinear", "log1p_nonnegative", (column,), priority=2.0),
                    ]
                )
            elif meta.group in {"temperature", "humidity", "pressure"}:
                specs.append(CandidateSpec("nonlinear", "square", (column,), priority=2.0))

            for window in self.config.windows:
                for operator in ("lag", "delta", "rolling_mean", "rolling_std"):
                    priority = 4.0 if meta.group in {"wind_speed", "irradiance", "historical_power"} else 2.5
                    specs.append(
                        CandidateSpec("temporal", operator, (column,), window, priority=priority)
                    )

        columns = list(metadata)
        for left_index, left in enumerate(columns):
            left_meta = metadata[left]
            for right in columns[left_index + 1:]:
                right_meta = metadata[right]
                if (
                    left_meta.unit == right_meta.unit
                    and left_meta.unit != "unknown"
                    and not left_meta.periodic
                    and not right_meta.periodic
                ):
                    specs.extend(
                        [
                            CandidateSpec("interaction", "add", (left, right), priority=3.0),
                            CandidateSpec("interaction", "subtract", (left, right), priority=3.0),
                            CandidateSpec("interaction", "subtract", (right, left), priority=3.0),
                        ]
                    )
                groups = {left_meta.group, right_meta.group}
                if groups == {"irradiance", "temperature"}:
                    specs.append(
                        CandidateSpec("physical", "multiply", (left, right), priority=5.0)
                    )
                if left_meta.group == right_meta.group == "wind_speed":
                    specs.extend(
                        [
                            CandidateSpec("physical", "subtract", (left, right), priority=4.0),
                            CandidateSpec("physical", "subtract", (right, left), priority=4.0),
                            CandidateSpec("physical", "safe_divide", (left, right), priority=4.0),
                            CandidateSpec("physical", "safe_divide", (right, left), priority=4.0),
                        ]
                    )
                if left_meta.group == right_meta.group == "wind_direction":
                    specs.append(
                        CandidateSpec("physical", "circular_difference", (left, right), priority=5.0)
                    )
        return specs

    @staticmethod
    def _group_specs(specs: Iterable[CandidateSpec]) -> Dict[str, List[CandidateSpec]]:
        grouped: Dict[str, List[CandidateSpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.group, []).append(spec)
        for group in grouped:
            grouped[group].sort(key=lambda item: (-item.priority, item.name))
        return grouped

    def _pilot_group_evaluation(
        self,
        data: pd.DataFrame,
        label: pd.Series,
        grouped: Dict[str, List[CandidateSpec]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        split = max(64, int(len(data) * 0.7))
        if split >= len(data) - 32:
            split = len(data) - 32
        y_train = label.iloc[:split].to_numpy(dtype=float)
        y_valid = label.iloc[split:].to_numpy(dtype=float)
        baseline = float(np.mean((y_valid - np.mean(y_train)) ** 2)) + 1e-12
        gains: Dict[str, float] = {}
        costs: Dict[str, float] = {}

        for group, specs in grouped.items():
            pilot_size = min(self.config.pilot_per_group, len(specs))
            pilot_positions = np.linspace(0, len(specs) - 1, pilot_size, dtype=int)
            pilot = [specs[position] for position in dict.fromkeys(pilot_positions)]
            start = time.perf_counter()
            features = self._transform_specs(data, pilot)
            elapsed = time.perf_counter() - start
            candidate_gains = []
            for column in features.columns:
                x_train = features[column].iloc[:split].to_numpy(dtype=float)
                x_valid = features[column].iloc[split:].to_numpy(dtype=float)
                gain = self._univariate_gain(x_train, y_train, x_valid, y_valid, baseline)
                if np.isfinite(gain):
                    candidate_gains.append(max(0.0, gain))
            gains[group] = float(np.mean(candidate_gains)) if candidate_gains else 0.0
            costs[group] = elapsed / max(1, len(pilot))
        return gains, costs

    def _univariate_gain(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        baseline: float,
    ) -> float:
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        x_valid = np.nan_to_num(x_valid, nan=0.0, posinf=0.0, neginf=0.0)
        x_mean = float(x_train.mean())
        y_mean = float(y_train.mean())
        centered_x = x_train - x_mean
        denominator = float(centered_x @ centered_x + self.config.ridge)
        slope = float(centered_x @ (y_train - y_mean) / denominator)
        intercept = y_mean - slope * x_mean
        prediction = intercept + slope * x_valid
        mse = float(np.mean((y_valid - prediction) ** 2))
        return (baseline - mse) / baseline

    def _allocate_budget(
        self,
        grouped: Dict[str, List[CandidateSpec]],
        gains: Dict[str, float],
        costs: Dict[str, float],
    ) -> Dict[str, int]:
        groups = sorted(grouped)
        total_available = sum(len(grouped[group]) for group in groups)
        budget = min(self.config.budget, total_available)
        allocation = {group: min(self.config.min_group_budget, len(grouped[group])) for group in groups}
        remaining = budget - sum(allocation.values())
        if remaining <= 0:
            return self._trim_allocation(allocation, budget)

        if self.config.allocation == "uniform":
            weights = np.ones(len(groups), dtype=float)
        else:
            cost_values = np.array([costs.get(group, 0.0) for group in groups], dtype=float)
            positive_costs = cost_values[cost_values > 0]
            reference_cost = float(np.median(positive_costs)) if len(positive_costs) else 1.0
            normalized_cost = cost_values / max(reference_cost, 1e-12)
            utilities = np.array([gains.get(group, 0.0) for group in groups], dtype=float)
            utilities = utilities / (1.0 + normalized_cost)
            if np.allclose(utilities, utilities[0]):
                weights = np.ones(len(groups), dtype=float)
            else:
                standardized = (utilities - utilities.mean()) / (utilities.std() + 1e-12)
                weights = np.exp(standardized / max(self.config.temperature, 1e-6))
        weights /= weights.sum()
        uniform_weights = np.full(len(groups), 1.0 / len(groups))
        weights = (
            self.config.adaptive_strength * weights
            + (1.0 - self.config.adaptive_strength) * uniform_weights
        )

        maximum_per_group = {
            group: min(
                len(grouped[group]),
                max(allocation[group], int(math.ceil(budget * self.config.max_group_fraction))),
            )
            for group in groups
        }
        raw = weights * remaining
        for group, amount in zip(groups, np.floor(raw).astype(int)):
            capacity = maximum_per_group[group] - allocation[group]
            allocation[group] += min(capacity, int(amount))
        remaining = budget - sum(allocation.values())
        order = np.argsort(-(raw - np.floor(raw)))
        while remaining > 0:
            changed = False
            for index in order:
                group = groups[int(index)]
                if allocation[group] < maximum_per_group[group]:
                    allocation[group] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                for group in groups:
                    if allocation[group] < len(grouped[group]):
                        allocation[group] += 1
                        remaining -= 1
                        changed = True
                        if remaining == 0:
                            break
            if not changed:
                break
        return allocation

    @staticmethod
    def _trim_allocation(allocation: Dict[str, int], budget: int) -> Dict[str, int]:
        trimmed = {group: 0 for group in allocation}
        for group in sorted(allocation):
            take = min(allocation[group], budget - sum(trimmed.values()))
            trimmed[group] = max(0, take)
            if sum(trimmed.values()) == budget:
                break
        return trimmed

    @staticmethod
    def _select_specs(
        grouped: Dict[str, List[CandidateSpec]],
        allocation: Dict[str, int],
        gains: Dict[str, float],
    ) -> List[CandidateSpec]:
        selected: List[CandidateSpec] = []
        for group in sorted(grouped, key=lambda item: (-gains.get(item, 0.0), item)):
            selected.extend(
                PAFAFeatureGenerator._diversified_take(
                    grouped[group], allocation.get(group, 0)
                )
            )
        return selected

    @staticmethod
    def _diversified_take(specs: Sequence[CandidateSpec], count: int) -> List[CandidateSpec]:
        """Avoid lexical column bias while retaining physics-priority ordering."""
        remaining = list(specs)
        chosen: List[CandidateSpec] = []
        input_counts: Dict[str, int] = {}
        operator_counts: Dict[str, int] = {}
        while remaining and len(chosen) < count:
            best_index = 0
            best_key = None
            for index, spec in enumerate(remaining):
                input_penalty = sum(input_counts.get(name, 0) for name in spec.inputs)
                operator_penalty = operator_counts.get(spec.operator, 0)
                diversity_score = spec.priority - 0.30 * input_penalty - 0.15 * operator_penalty
                key = (diversity_score, spec.priority, -len(spec.inputs), spec.name)
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            selected = remaining.pop(best_index)
            chosen.append(selected)
            for name in selected.inputs:
                input_counts[name] = input_counts.get(name, 0) + 1
            operator_counts[selected.operator] = operator_counts.get(selected.operator, 0) + 1
        return chosen

    def _filter_train_features(
        self,
        specs: Sequence[CandidateSpec],
        features: pd.DataFrame,
    ) -> Tuple[List[CandidateSpec], pd.DataFrame, dict]:
        kept_specs: List[CandidateSpec] = []
        kept_columns: List[str] = []
        signatures = set()
        invalid = constant = duplicate = 0
        stride = max(1, len(features) // 2048)

        for spec in specs:
            values = features[spec.name].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.mean() < 1.0 - self.config.max_nan_fraction:
                invalid += 1
                continue
            finite_values = values[finite]
            if not len(finite_values) or float(np.std(finite_values)) <= self.config.min_std:
                constant += 1
                continue
            sampled = np.nan_to_num(values[::stride], nan=9.87654321e37, posinf=8.7654321e37, neginf=-8.7654321e37)
            signature = hashlib.sha1(np.round(sampled, 10).tobytes()).hexdigest()
            if signature in signatures:
                duplicate += 1
                continue
            signatures.add(signature)
            kept_specs.append(spec)
            kept_columns.append(spec.name)

        return (
            kept_specs,
            features[kept_columns].copy(),
            {"invalid": invalid, "constant": constant, "duplicate": duplicate},
        )

    def _transform_specs(
        self,
        data: pd.DataFrame,
        specs: Sequence[CandidateSpec],
    ) -> pd.DataFrame:
        output = {}
        for spec in specs:
            output[spec.name] = self._apply_spec(data, spec)
        return pd.DataFrame(output, index=data.index)

    @staticmethod
    def _apply_spec(data: pd.DataFrame, spec: CandidateSpec) -> pd.Series:
        left = data[spec.inputs[0]].astype(float)
        operator = spec.operator
        if operator == "square":
            return left.pow(2)
        if operator == "cube":
            return left.pow(3)
        if operator == "log1p_abs":
            return np.log1p(left.abs())
        if operator == "log1p_nonnegative":
            return np.log1p(left.clip(lower=0))
        if operator == "direction_sin":
            return np.sin(np.deg2rad(left))
        if operator == "direction_cos":
            return np.cos(np.deg2rad(left))
        if operator == "lag":
            return left.shift(spec.window)
        if operator == "delta":
            return left - left.shift(spec.window)
        if operator == "rolling_mean":
            return left.rolling(spec.window, min_periods=1).mean()
        if operator == "rolling_std":
            return left.rolling(spec.window, min_periods=2).std()
        if operator == "circular_mean_sin":
            return np.sin(np.deg2rad(left)).rolling(spec.window, min_periods=1).mean()
        if operator == "circular_mean_cos":
            return np.cos(np.deg2rad(left)).rolling(spec.window, min_periods=1).mean()

        right = data[spec.inputs[1]].astype(float)
        if operator == "add":
            return left + right
        if operator == "subtract":
            return left - right
        if operator == "multiply":
            return left * right
        if operator == "safe_divide":
            denominator = right.where(right.abs() > 1e-12)
            return left / denominator
        if operator == "circular_difference":
            return ((left - right + 180.0) % 360.0) - 180.0
        raise ValueError(f"Unsupported PAFA operator: {operator}")
