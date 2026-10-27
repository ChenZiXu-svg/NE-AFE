from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .splits import ChronologicalFold, make_walk_forward_folds


@dataclass
class GWBPConfig:
    n_splits: int = 4
    calibration_fraction: float = 0.1
    validation_fraction: float = 0.1
    gap: int = 0
    max_features: int = 20
    score_mode: str = "mean_variance"
    stability_lambda: float = 0.5
    quantile: float = 0.25
    correction_shrinkage: float = 0.5
    correction_ridge: float = 1e-4
    redundancy_gamma: float = 0.0
    min_positive_fraction: float = 0.5
    min_score: float = 0.0
    base_max_iter: int = 80
    base_max_leaf_nodes: int = 15
    random_state: int = 42
    correlation_sample_size: int = 5000
    sequential_updates: bool = True

    def validate(self) -> None:
        if self.score_mode not in {"mean", "mean_variance", "quantile"}:
            raise ValueError("Unsupported GWBP score_mode.")
        if not 0 < self.quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5].")
        if not 0 < self.correction_shrinkage <= 1:
            raise ValueError("correction_shrinkage must be in (0, 1].")
        if self.max_features < 1:
            raise ValueError("max_features must be positive.")
        if not 0 <= self.min_positive_fraction <= 1:
            raise ValueError("min_positive_fraction must be in [0, 1].")


@dataclass
class _FoldState:
    fold: ChronologicalFold
    candidate_calibration: np.ndarray
    candidate_validation: np.ndarray
    y_calibration: np.ndarray
    y_validation: np.ndarray
    prediction_calibration: np.ndarray
    prediction_validation: np.ndarray


class GWBPSelector:
    """Generalization-aware sequential walk-forward boosting selector."""

    def __init__(self, config: Optional[GWBPConfig] = None):
        self.config = config or GWBPConfig()
        self.config.validate()

    def fit(
        self,
        base_features: pd.DataFrame,
        candidate_features: pd.DataFrame,
        label: pd.Series,
    ) -> "GWBPSelector":
        base_features, candidate_features, label = self._validate_inputs(
            base_features, candidate_features, label
        )
        self.base_columns_ = list(base_features.columns)
        self.candidate_columns_ = list(candidate_features.columns)
        candidate_matrix = self._sanitize_candidates(candidate_features)
        y = label.to_numpy(dtype=float)
        folds = make_walk_forward_folds(
            len(label),
            n_splits=self.config.n_splits,
            calibration_fraction=self.config.calibration_fraction,
            validation_fraction=self.config.validation_fraction,
            gap=self.config.gap,
        )
        states = self._prepare_fold_states(base_features, candidate_matrix, y, folds)
        correlation = self._candidate_correlation(candidate_matrix)

        selected: List[int] = []
        history = []
        candidate_history = []
        stop_reason = "max_features_reached"
        for iteration in range(self.config.max_features):
            gains, slopes, intercepts = self._candidate_gains(states)
            robust, means, standard_deviations, quantiles, positive_fraction = self._robust_score(gains)
            redundancy = np.zeros(len(self.candidate_columns_), dtype=float)
            if selected and self.config.redundancy_gamma > 0:
                redundancy = np.max(np.abs(correlation[:, selected]), axis=1)
            final_score = robust - self.config.redundancy_gamma * redundancy
            if selected:
                final_score[selected] = -np.inf
            final_score[positive_fraction < self.config.min_positive_fraction] = -np.inf

            best = int(np.argmax(final_score))
            best_score = float(final_score[best])
            accepted = np.isfinite(best_score) and best_score > self.config.min_score
            candidate_history.extend(
                {
                    "iteration": iteration + 1,
                    "feature": self.candidate_columns_[candidate_index],
                    "final_score": float(final_score[candidate_index]),
                    "robust_score": float(robust[candidate_index]),
                    "mean_gain": float(means[candidate_index]),
                    "gain_std": float(standard_deviations[candidate_index]),
                    "quantile_gain": float(quantiles[candidate_index]),
                    "positive_fraction": float(positive_fraction[candidate_index]),
                    "redundancy": float(redundancy[candidate_index]),
                    "selected": bool(accepted and candidate_index == best),
                }
                for candidate_index in range(len(self.candidate_columns_))
            )
            if not np.isfinite(best_score) or best_score <= self.config.min_score:
                stop_reason = "no_positive_gain"
                break

            selected.append(best)
            history.append(
                {
                    "iteration": iteration + 1,
                    "feature": self.candidate_columns_[best],
                    "score": best_score,
                    "mean_gain": float(means[best]),
                    "gain_std": float(standard_deviations[best]),
                    "quantile_gain": float(quantiles[best]),
                    "positive_fraction": float(positive_fraction[best]),
                    "redundancy": float(redundancy[best]),
                    **{
                        f"fold_{fold_index + 1}_gain": float(gains[fold_index, best])
                        for fold_index in range(gains.shape[0])
                    },
                }
            )
            if self.config.sequential_updates:
                self._update_states(states, best, slopes, intercepts)

        self.selected_indices_ = selected
        self.selected_features_ = [self.candidate_columns_[index] for index in selected]
        self.history_ = pd.DataFrame(history)
        self.candidate_history_ = pd.DataFrame(candidate_history)
        self.fold_losses_ = self._fold_loss_frame(states)
        self.stop_reason_ = stop_reason
        self.diagnostics_ = {
            "candidate_count": len(self.candidate_columns_),
            "selected_count": len(self.selected_features_),
            "score_mode": self.config.score_mode,
            "redundancy_gamma": self.config.redundancy_gamma,
            "n_splits": len(folds),
            "sequential_updates": self.config.sequential_updates,
            "stop_reason": stop_reason,
        }
        return self

    def transform(self, candidate_features: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        missing = set(self.selected_features_) - set(candidate_features.columns)
        if missing:
            raise ValueError(f"Missing selected candidate columns: {sorted(missing)}")
        return candidate_features[self.selected_features_].copy()

    def fit_transform(
        self,
        base_features: pd.DataFrame,
        candidate_features: pd.DataFrame,
        label: pd.Series,
    ) -> pd.DataFrame:
        self.fit(base_features, candidate_features, label)
        return self.transform(candidate_features)

    def _validate_inputs(
        self,
        base_features: pd.DataFrame,
        candidate_features: pd.DataFrame,
        label: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        if not isinstance(base_features, pd.DataFrame) or not isinstance(candidate_features, pd.DataFrame):
            raise TypeError("GWBP expects pandas DataFrames for both feature sets.")
        if candidate_features.shape[1] == 0:
            raise ValueError("GWBP requires at least one candidate feature.")
        if len(base_features) != len(candidate_features) or len(base_features) != len(label):
            raise ValueError("Base features, candidate features and label must have equal length.")
        base = base_features.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        candidates = candidate_features.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
        y = pd.Series(np.asarray(label, dtype=float)).reset_index(drop=True)
        return base, candidates, y

    @staticmethod
    def _sanitize_candidates(candidate_features: pd.DataFrame) -> np.ndarray:
        matrix = candidate_features.to_numpy(dtype=float)
        matrix[~np.isfinite(matrix)] = np.nan
        medians = np.nanmedian(matrix, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        missing_rows, missing_columns = np.where(~np.isfinite(matrix))
        matrix[missing_rows, missing_columns] = medians[missing_columns]
        return matrix.astype(np.float64, copy=False)

    def _base_estimator(self) -> Pipeline:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.config.base_max_iter,
            max_leaf_nodes=self.config.base_max_leaf_nodes,
            learning_rate=0.08,
            l2_regularization=1e-4,
            random_state=self.config.random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    def _prepare_fold_states(
        self,
        base_features: pd.DataFrame,
        candidate_matrix: np.ndarray,
        label: np.ndarray,
        folds: Sequence[ChronologicalFold],
    ) -> List[_FoldState]:
        states: List[_FoldState] = []
        for fold in folds:
            estimator = self._base_estimator()
            estimator.fit(base_features.iloc[fold.base], label[fold.base])
            prediction_calibration = estimator.predict(base_features.iloc[fold.calibration])
            prediction_validation = estimator.predict(base_features.iloc[fold.validation])
            states.append(
                _FoldState(
                    fold=fold,
                    candidate_calibration=candidate_matrix[fold.calibration],
                    candidate_validation=candidate_matrix[fold.validation],
                    y_calibration=label[fold.calibration],
                    y_validation=label[fold.validation],
                    prediction_calibration=np.asarray(prediction_calibration, dtype=float),
                    prediction_validation=np.asarray(prediction_validation, dtype=float),
                )
            )
        return states

    def _candidate_gains(
        self,
        states: Sequence[_FoldState],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_folds = len(states)
        n_candidates = len(self.candidate_columns_)
        gains = np.empty((n_folds, n_candidates), dtype=float)
        slopes = np.empty_like(gains)
        intercepts = np.empty_like(gains)

        for fold_index, state in enumerate(states):
            x_calibration = state.candidate_calibration
            residual = state.y_calibration - state.prediction_calibration
            x_mean = x_calibration.mean(axis=0)
            residual_mean = float(residual.mean())
            centered = x_calibration - x_mean
            denominator = np.einsum("ij,ij->j", centered, centered) + self.config.correction_ridge
            fold_slopes = centered.T @ (residual - residual_mean) / denominator
            fold_intercepts = residual_mean - fold_slopes * x_mean
            correction = (
                state.candidate_validation * fold_slopes[None, :]
                + fold_intercepts[None, :]
            )
            candidate_prediction = (
                state.prediction_validation[:, None]
                + self.config.correction_shrinkage * correction
            )
            current_loss = float(
                np.mean((state.y_validation - state.prediction_validation) ** 2)
            )
            candidate_loss = np.mean(
                (state.y_validation[:, None] - candidate_prediction) ** 2,
                axis=0,
            )
            gains[fold_index] = (current_loss - candidate_loss) / (current_loss + 1e-12)
            slopes[fold_index] = fold_slopes
            intercepts[fold_index] = fold_intercepts
        return gains, slopes, intercepts

    def _robust_score(
        self,
        gains: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = gains.mean(axis=0)
        standard_deviations = gains.std(axis=0)
        quantiles = np.quantile(gains, self.config.quantile, axis=0)
        positive_fraction = (gains > 0).mean(axis=0)
        if self.config.score_mode == "mean":
            robust = means
        elif self.config.score_mode == "mean_variance":
            robust = means - self.config.stability_lambda * standard_deviations
        else:
            robust = quantiles
        return robust, means, standard_deviations, quantiles, positive_fraction

    def _update_states(
        self,
        states: Sequence[_FoldState],
        selected: int,
        slopes: np.ndarray,
        intercepts: np.ndarray,
    ) -> None:
        for fold_index, state in enumerate(states):
            slope = slopes[fold_index, selected]
            intercept = intercepts[fold_index, selected]
            state.prediction_calibration += self.config.correction_shrinkage * (
                state.candidate_calibration[:, selected] * slope + intercept
            )
            state.prediction_validation += self.config.correction_shrinkage * (
                state.candidate_validation[:, selected] * slope + intercept
            )

    def _candidate_correlation(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) > self.config.correlation_sample_size:
            positions = np.linspace(
                0,
                len(matrix) - 1,
                self.config.correlation_sample_size,
                dtype=int,
            )
            matrix = matrix[positions]
        standard_deviation = matrix.std(axis=0)
        standard_deviation[standard_deviation < 1e-12] = 1.0
        standardized = (matrix - matrix.mean(axis=0)) / standard_deviation
        correlation = standardized.T @ standardized / max(1, len(standardized) - 1)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        return correlation

    @staticmethod
    def _fold_loss_frame(states: Sequence[_FoldState]) -> pd.DataFrame:
        rows = []
        for index, state in enumerate(states, start=1):
            rows.append(
                {
                    "fold": index,
                    "calibration_rmse": float(
                        np.sqrt(np.mean((state.y_calibration - state.prediction_calibration) ** 2))
                    ),
                    "validation_rmse": float(
                        np.sqrt(np.mean((state.y_validation - state.prediction_validation) ** 2))
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("GWBPSelector must be fitted before transform().")
