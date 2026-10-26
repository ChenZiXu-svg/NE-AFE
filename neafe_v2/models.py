from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from .evaluation import regression_metrics


MODEL_NAMES = ("Ridge", "XGBoost", "MLP", "LSTM", "Transformer")


@dataclass
class ModelConfig:
    validation_fraction: float = 0.15
    sequence_length: int = 12
    max_train_samples: int = 12000
    batch_size: int = 256
    max_epochs: int = 16
    patience: int = 3
    learning_rate: float = 1e-3
    hidden_size: int = 32
    num_threads: int = 4
    ridge_alpha: float = 1.0
    xgb_estimators: int = 300
    xgb_patience: int = 20
    clip_quantile: float = 0.001

    def validate(self) -> None:
        if not 0.05 <= self.validation_fraction <= 0.4:
            raise ValueError("validation_fraction must be in [0.05, 0.4].")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two.")
        if self.max_train_samples < 256:
            raise ValueError("max_train_samples must be at least 256.")
        if not 0 <= self.clip_quantile < 0.1:
            raise ValueError("clip_quantile must be in [0, 0.1).")


def _set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, num_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _chronological_split(n_samples: int, fraction: float, minimum: int = 256) -> int:
    split = int(n_samples * (1.0 - fraction))
    return min(n_samples - minimum, max(minimum, split))


def _fit_preprocessor(
    train_x: pd.DataFrame,
    evaluation_x: pd.DataFrame,
    fit_end: int,
    clip_quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_values = imputer.fit_transform(train_x.iloc[:fit_end])
    lower, upper = _clip_bounds(fit_values, clip_quantile)
    fit_values = np.clip(fit_values, lower, upper)
    scaler.fit(fit_values)
    train_values = np.clip(imputer.transform(train_x), lower, upper)
    evaluation_values = np.clip(imputer.transform(evaluation_x), lower, upper)
    train_values = scaler.transform(train_values).astype(np.float32)
    evaluation_values = scaler.transform(evaluation_values).astype(np.float32)
    return train_values, evaluation_values


def _clip_bounds(values: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    if quantile <= 0:
        return np.full(values.shape[1], -np.inf), np.full(values.shape[1], np.inf)
    lower = np.quantile(values, quantile, axis=0)
    upper = np.quantile(values, 1.0 - quantile, axis=0)
    return lower, upper


def _fit_tree_preprocessor(
    train_x: pd.DataFrame,
    evaluation_x: pd.DataFrame,
    fit_end: int,
    clip_quantile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    fit_values = imputer.fit_transform(train_x.iloc[:fit_end])
    lower, upper = _clip_bounds(fit_values, clip_quantile)
    fit_values = np.clip(fit_values, lower, upper)
    validation_values = np.clip(imputer.transform(train_x.iloc[fit_end:]), lower, upper)
    evaluation_values = np.clip(imputer.transform(evaluation_x), lower, upper)
    return fit_values, validation_values, evaluation_values


def _sample_positions(size: int, cap: int) -> np.ndarray:
    if size <= cap:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, cap, dtype=int))


def _tabular_tensors(
    values: np.ndarray,
    targets: np.ndarray,
    split: int,
    cap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = _sample_positions(split, cap)
    return (
        torch.from_numpy(values[positions]),
        torch.from_numpy(targets[positions, None].astype(np.float32)),
        torch.from_numpy(values[split:]),
        torch.from_numpy(targets[split:, None].astype(np.float32)),
    )


def _sequence_arrays(
    values: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    start_target: int,
    end_target: int,
    cap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    first = max(sequence_length - 1, start_target)
    target_positions = np.arange(first, end_target)
    if cap is not None:
        target_positions = target_positions[_sample_positions(len(target_positions), cap)]
    sequences = np.stack(
        [values[position - sequence_length + 1 : position + 1] for position in target_positions]
    )
    return sequences.astype(np.float32), targets[target_positions, None].astype(np.float32)


def _evaluation_sequences(
    train_values: np.ndarray,
    evaluation_values: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    history = train_values[-(sequence_length - 1) :]
    combined = np.concatenate([history, evaluation_values], axis=0)
    sequences = [combined[index : index + sequence_length] for index in range(len(evaluation_values))]
    return np.stack(sequences).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.recurrent = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.recurrent(values)
        return self.output(encoded[:, -1])


class _Transformer(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, sequence_length: int):
        super().__init__()
        n_heads = 4 if hidden_size % 4 == 0 else 2
        self.input_projection = nn.Linear(n_features, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(values) + self.position[:, : values.shape[1]])
        return self.output(encoded[:, -1])


def _fit_torch_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    config: ModelConfig,
    seed: int,
) -> Tuple[nn.Module, int, float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    best_loss = math.inf
    best_state = None
    stale_epochs = 0
    completed_epochs = 0
    for epoch in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).item())
        completed_epochs = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, completed_epochs, best_loss


def _batched_prediction(model: nn.Module, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size])
            predictions.append(model(batch).numpy().reshape(-1))
    return np.concatenate(predictions)


def fit_predict_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    evaluation_x: pd.DataFrame,
    seed: int,
    config: ModelConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    config = config or ModelConfig()
    config.validate()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    _set_seed(seed, config.num_threads)
    start = time.perf_counter()
    targets = np.asarray(train_y, dtype=np.float32)
    split = _chronological_split(len(train_x), config.validation_fraction)

    if model_name == "Ridge":
        train_values, evaluation_values = _fit_preprocessor(
            train_x,
            evaluation_x,
            fit_end=len(train_x),
            clip_quantile=config.clip_quantile,
        )
        model = Ridge(alpha=config.ridge_alpha)
        model.fit(train_values, targets)
        prediction = model.predict(evaluation_values)
        details = {"epochs": 0, "validation_loss": np.nan}
    elif model_name == "XGBoost":
        fit_values, validation_values, evaluation_values = _fit_tree_preprocessor(
            train_x, evaluation_x, split, config.clip_quantile
        )
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.xgb_estimators,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=config.num_threads,
            random_state=seed,
            early_stopping_rounds=config.xgb_patience,
        )
        model.fit(
            fit_values,
            targets[:split],
            eval_set=[(validation_values, targets[split:])],
            verbose=False,
        )
        prediction = model.predict(evaluation_values)
        details = {
            "epochs": int(getattr(model, "best_iteration", config.xgb_estimators - 1)) + 1,
            "validation_loss": float(getattr(model, "best_score", np.nan)),
        }
    else:
        train_values, evaluation_values = _fit_preprocessor(
            train_x,
            evaluation_x,
            fit_end=split,
            clip_quantile=config.clip_quantile,
        )
        target_mean = float(np.mean(targets[:split]))
        target_scale = float(np.std(targets[:split]))
        if target_scale < 1e-8:
            target_scale = 1.0
        scaled_targets = ((targets - target_mean) / target_scale).astype(np.float32)
        if model_name == "MLP":
            train_tensor, target_tensor, validation_tensor, validation_target = _tabular_tensors(
                train_values, scaled_targets, split, config.max_train_samples
            )
            model = _MLP(train_values.shape[1], config.hidden_size)
            evaluation_input = evaluation_values
        else:
            train_sequences, train_targets = _sequence_arrays(
                train_values,
                scaled_targets,
                config.sequence_length,
                0,
                split,
                cap=config.max_train_samples,
            )
            validation_sequences, validation_targets = _sequence_arrays(
                train_values,
                scaled_targets,
                config.sequence_length,
                split,
                len(train_values),
            )
            train_tensor = torch.from_numpy(train_sequences)
            target_tensor = torch.from_numpy(train_targets)
            validation_tensor = torch.from_numpy(validation_sequences)
            validation_target = torch.from_numpy(validation_targets)
            evaluation_input = _evaluation_sequences(
                train_values, evaluation_values, config.sequence_length
            )
            if model_name == "LSTM":
                model = _LSTM(train_values.shape[1], config.hidden_size)
            else:
                model = _Transformer(
                    train_values.shape[1], config.hidden_size, config.sequence_length
                )
        model, epochs, validation_loss = _fit_torch_model(
            model,
            train_tensor,
            target_tensor,
            validation_tensor,
            validation_target,
            config,
            seed,
        )
        prediction = (
            _batched_prediction(model, evaluation_input, config.batch_size * 2) * target_scale
            + target_mean
        )
        details = {"epochs": epochs, "validation_loss": validation_loss}

    details.update(
        {
            "model": model_name,
            "seed": seed,
            "fit_samples": split if model_name != "Ridge" else len(train_x),
            "runtime_seconds": time.perf_counter() - start,
        }
    )
    return np.asarray(prediction, dtype=float), details


def fit_and_evaluate_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    evaluation_x: pd.DataFrame,
    evaluation_y: pd.Series,
    seed: int,
    config: ModelConfig | None = None,
) -> Tuple[Dict[str, float | int | str], np.ndarray]:
    prediction, details = fit_predict_model(
        model_name, train_x, train_y, evaluation_x, seed, config=config
    )
    metrics: Dict[str, float | int | str] = regression_metrics(evaluation_y, prediction)
    metrics.update(details)
    return metrics, prediction


def model_config_dict(config: ModelConfig) -> dict:
    return asdict(config)
