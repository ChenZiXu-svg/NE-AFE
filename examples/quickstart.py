"""Demonstrate the released API using a reproducible synthetic time series."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neafe_v2 import PAFAConfig, PAFAFeatureGenerator, GWBPConfig, GWBPSelector
from neafe_v2.evaluation import concatenate_feature_sets, regression_metrics


def synthetic_series(n: int = 1800, seed: int = 42):
    rng = np.random.default_rng(seed)
    time = np.arange(n)
    speed = np.clip(6 + 2 * np.sin(time / 37) + rng.normal(0, 1.2, n), 0, 14)
    direction = (180 + 60 * np.sin(time / 83) + rng.normal(0, 12, n)) % 360
    temperature = 18 + 6 * np.sin(time / 96) + rng.normal(0, 1, n)
    power = np.maximum(0, 0.035 * speed**3 + rng.normal(0, 1.5, n))
    x = pd.DataFrame({"WS10": speed, "WD10": direction,
                      "Temperature": temperature, "Lag1": np.r_[0, power[:-1]]})
    return x, pd.Series(power, name="Power")


def main():
    x, y = synthetic_series()
    boundary = int(len(x) * 0.8)
    development_x, test_x = x.iloc[:boundary], x.iloc[boundary:]
    development_y, test_y = y.iloc[:boundary], y.iloc[boundary:]
    generator = PAFAFeatureGenerator(PAFAConfig(
        mode="physics", allocation="uniform", budget=120, windows=(4, 8, 16),
        soft_constraints=False, adaptive_strength=0.0,
    ))
    candidates = generator.fit_transform(development_x, development_y)
    selector = GWBPSelector(GWBPConfig(
        n_splits=4, calibration_fraction=0.08, validation_fraction=0.08,
        gap=16, max_features=12, score_mode="mean", base_max_iter=60,
    )).fit(development_x, candidates, development_y)
    selected = selector.transform(candidates)
    selected_test = selector.transform(generator.transform(test_x, history=development_x))
    output = {"data": "synthetic demonstration", "candidates": len(candidates.columns),
              "selected_features": selector.selected_features_, "metrics": {}}
    for name, dx, tx in [
        ("Raw", development_x, test_x),
        ("NE-AFE", concatenate_feature_sets(development_x, selected),
         concatenate_feature_sets(test_x, selected_test)),
    ]:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        model.fit(dx, development_y)
        output["metrics"][name] = regression_metrics(test_y, model.predict(tx))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
