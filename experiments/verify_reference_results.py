"""Check reported aggregation against the recorded, sample-free result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]/"reference_results"
    frame = pd.read_csv(root/"neafe_v2_full_final/model_results.csv")
    assert frame.station.nunique() == 14
    assert frame.model.nunique() == 5
    averages = frame.groupby(["station", "model", "feature_set"]).nrmse.mean()
    paired = averages.unstack("feature_set")
    gains = ((paired["Raw"] - paired["NE-AFE"])/paired["Raw"]*100).groupby("model").mean()
    expected = {"Ridge": 13.46, "XGBoost": 12.39, "MLP": 13.11}
    for model, value in expected.items():
        assert np.isclose(gains[model], value, atol=0.005), (model, gains[model])
    selected = pd.read_csv(root/"figure_data/04_selected_features.csv")
    assert len(selected) == 148
    assert selected.groupby("operator_group").size().to_dict() == {
        "temporal": 112, "physical": 19, "interaction": 10, "nonlinear": 7,
    }
    print(json.dumps({"stations": 14, "models": 5, "result_rows": len(frame),
                      "mean_station_relative_gain_percent": gains.to_dict(),
                      "selected_features": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
