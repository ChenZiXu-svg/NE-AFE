from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariableMetadata:
    name: str
    variable_type: str
    unit: str
    domain: str
    availability: str
    group: str
    nonnegative: bool
    periodic: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _normalized_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def infer_variable_metadata(name: str, values: pd.Series) -> VariableMetadata:
    key = _normalized_name(name)
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed_min = finite.min(skipna=True)
    nonnegative = bool(pd.isna(observed_min) or observed_min >= 0)

    if key.startswith("wd") or "winddirection" in key or "direction" in key:
        return VariableMetadata(
            name=name,
            variable_type="circular",
            unit="degree",
            domain="[0, 360)",
            availability="forecast",
            group="wind_direction",
            nonnegative=True,
            periodic=True,
        )
    if key.startswith("ws") or "windspeed" in key:
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="m/s",
            domain="[0, +inf)",
            availability="forecast",
            group="wind_speed",
            nonnegative=True,
        )
    if "irradiance" in key or "radiation" in key:
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="W/m2",
            domain="[0, +inf)",
            availability="forecast",
            group="irradiance",
            nonnegative=True,
        )
    if "temperature" in key or key.startswith("temp"):
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="degC",
            domain="real",
            availability="forecast",
            group="temperature",
            nonnegative=False,
        )
    if "humidity" in key:
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="percent",
            domain="[0, 100]",
            availability="forecast",
            group="humidity",
            nonnegative=True,
        )
    if "atmosphere" in key or "pressure" in key:
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="pressure",
            domain="[0, +inf)",
            availability="forecast",
            group="pressure",
            nonnegative=True,
        )
    if key.startswith("lag") or "powerlag" in key:
        return VariableMetadata(
            name=name,
            variable_type="continuous",
            unit="MW",
            domain="[0, capacity]",
            availability="historical",
            group="historical_power",
            nonnegative=True,
        )
    return VariableMetadata(
        name=name,
        variable_type="continuous",
        unit="unknown",
        domain="[0, +inf)" if nonnegative else "real",
        availability="forecast",
        group="other",
        nonnegative=nonnegative,
    )


def infer_metadata(data: pd.DataFrame) -> Dict[str, VariableMetadata]:
    return {column: infer_variable_metadata(column, data[column]) for column in data.columns}


def metadata_frame(metadata: Iterable[VariableMetadata]) -> pd.DataFrame:
    return pd.DataFrame([item.to_dict() for item in metadata])
