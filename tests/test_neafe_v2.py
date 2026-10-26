import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neafe_v2 import (  # noqa: E402
    GWBPConfig,
    GWBPSelector,
    PAFAConfig,
    PAFAFeatureGenerator,
    make_walk_forward_folds,
)


class NEAFEV2Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 1000
        self.features = pd.DataFrame(
            {
                "WS10": rng.uniform(0, 12, n),
                "WD10": rng.uniform(0, 360, n),
                "Temperature": rng.normal(20, 5, n),
                "Lag1": rng.uniform(0, 50, n),
            }
        )
        self.label = 0.4 * self.features["WS10"] ** 3 + 0.2 * self.features["Lag1"]
        self.label += rng.normal(0, 5, n)

    def test_walk_forward_segments_are_ordered(self):
        folds = make_walk_forward_folds(1000, n_splits=3, gap=4)
        for fold in folds:
            self.assertLess(fold.base[-1], fold.calibration[0])
            self.assertLess(fold.calibration[-1], fold.validation[0])

    def test_future_rows_do_not_change_past_temporal_features(self):
        generator = PAFAFeatureGenerator(
            PAFAConfig(budget=30, windows=(4, 8), pilot_per_group=2)
        ).fit(self.features.iloc[:700], self.label.iloc[:700])
        evaluation = self.features.iloc[700:].copy()
        original = generator.transform(evaluation, history=self.features.iloc[:700])
        changed = evaluation.copy()
        changed.iloc[-1] = changed.iloc[-1] * 100
        transformed = generator.transform(changed, history=self.features.iloc[:700])
        pd.testing.assert_frame_equal(original.iloc[:-1], transformed.iloc[:-1])

    def test_gwbp_selects_predictive_candidates(self):
        generator = PAFAFeatureGenerator(
            PAFAConfig(budget=40, windows=(4, 8), pilot_per_group=3)
        ).fit(self.features, self.label)
        candidates = generator.transform(self.features)
        selector = GWBPSelector(
            GWBPConfig(n_splits=3, max_features=5, base_max_iter=20)
        ).fit(self.features, candidates, self.label)
        self.assertGreater(len(selector.selected_features_), 0)
        self.assertTrue(any("WS10" in name for name in selector.selected_features_))


if __name__ == "__main__":
    unittest.main()
