from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class StationData:
    name: str
    train_x: pd.DataFrame
    train_y: pd.Series
    validation_x: pd.DataFrame
    validation_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def load_station_data(root: Path, station: str) -> StationData:
    def read_x(split: str) -> pd.DataFrame:
        return pd.read_feather(root / f"{split}_raw_{station}.feather").reset_index(drop=True)

    def read_y(split: str) -> pd.Series:
        frame = pd.read_feather(root / f"{split}_y_{station}.feather")
        return frame.iloc[:, 0].astype(float).reset_index(drop=True)

    return StationData(
        name=station,
        train_x=read_x("train"),
        train_y=read_y("train"),
        validation_x=read_x("valid"),
        validation_y=read_y("valid"),
        test_x=read_x("test"),
        test_y=read_y("test"),
    )


def concatenate_feature_sets(
    raw: pd.DataFrame,
    candidates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    if candidates is None or candidates.shape[1] == 0:
        return raw
    candidates = candidates.reset_index(drop=True)
    return pd.concat([raw, candidates], axis=1)


def make_forecaster(random_state: int = 42, max_iter: int = 100) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.08,
                    max_iter=max_iter,
                    max_leaf_nodes=31,
                    l2_regularization=1e-4,
                    random_state=random_state,
                ),
            ),
        ]
    )


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = truth - prediction
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.max(truth) - np.min(truth))
    nrmse = rmse / denominator if denominator > 0 else np.nan
    total = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error ** 2)) / total if total > 0 else np.nan
    return {"rmse": rmse, "nrmse": nrmse, "mae": mae, "r2": r2}


def fit_and_evaluate(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    evaluation_x: pd.DataFrame,
    evaluation_y: pd.Series,
    random_state: int = 42,
    max_iter: int = 100,
) -> Tuple[Dict[str, float], np.ndarray]:
    model = make_forecaster(random_state=random_state, max_iter=max_iter)
    model.fit(train_x, train_y)
    prediction = model.predict(evaluation_x)
    return regression_metrics(evaluation_y, prediction), prediction
