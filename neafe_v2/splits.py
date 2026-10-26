from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ChronologicalFold:
    base: np.ndarray
    calibration: np.ndarray
    validation: np.ndarray

    def validate(self) -> None:
        if not (len(self.base) and len(self.calibration) and len(self.validation)):
            raise ValueError("Every chronological fold must contain all three segments.")
        if not (self.base[-1] < self.calibration[0] <= self.calibration[-1] < self.validation[0]):
            raise ValueError("Fold segments must be strictly time ordered.")


def chronological_three_way_split(
    n_samples: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError("At least three observations are required.")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test segment is required.")

    train_end = int(n_samples * train_fraction)
    validation_end = int(n_samples * (train_fraction + validation_fraction))
    index = np.arange(n_samples)
    return index[:train_end], index[train_end:validation_end], index[validation_end:]


def make_walk_forward_folds(
    n_samples: int,
    n_splits: int = 4,
    calibration_fraction: float = 0.1,
    validation_fraction: float = 0.1,
    gap: int = 0,
    min_base_size: int = 256,
) -> List[ChronologicalFold]:
    if n_splits < 2:
        raise ValueError("GWBP requires at least two chronological folds.")
    if gap < 0:
        raise ValueError("gap cannot be negative.")

    calibration_size = max(64, int(n_samples * calibration_fraction))
    validation_size = max(64, int(n_samples * validation_fraction))
    required = calibration_size + n_splits * validation_size + 2 * gap + min_base_size
    if required > n_samples:
        scale = (n_samples - min_base_size - 2 * gap) / (n_splits + 1)
        calibration_size = validation_size = int(scale)
    if calibration_size < 32 or validation_size < 32:
        raise ValueError("Not enough observations for the requested walk-forward folds.")

    first_validation_start = n_samples - n_splits * validation_size
    folds: List[ChronologicalFold] = []
    for split_number in range(n_splits):
        validation_start = first_validation_start + split_number * validation_size
        calibration_end = validation_start - gap
        calibration_start = calibration_end - calibration_size
        base_end = calibration_start - gap
        if base_end < min_base_size:
            raise ValueError("The first base segment is too small.")
        fold = ChronologicalFold(
            base=np.arange(0, base_end),
            calibration=np.arange(calibration_start, calibration_end),
            validation=np.arange(validation_start, validation_start + validation_size),
        )
        fold.validate()
        folds.append(fold)
    return folds
