"""NE-AFE on a synthetic wind-power time series."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from neafe import PAFAConfig, PAFAFeatureGenerator, GWBPConfig, GWBPSelector

# Synthetic inputs and a strictly historical power lag.
rng = np.random.default_rng(42)
t = np.arange(1800)
speed = np.clip(6 + 2 * np.sin(t / 37) + rng.normal(0, 1.2, len(t)), 0, 14)
power = np.maximum(0, 0.035 * speed**3 + rng.normal(0, 1.5, len(t)))
x = pd.DataFrame({"WS10": speed, "WD10": (180 + 60 * np.sin(t / 83)) % 360,
                  "Temperature": 18 + 6 * np.sin(t / 96),
                  "Lag1": np.r_[0, power[:-1]]})
y = pd.Series(power)
train_x, test_x = x.iloc[:1440], x.iloc[1440:]
train_y, test_y = y.iloc[:1440], y.iloc[1440:]

# Construct candidates on development data and select features chronologically.
generator = PAFAFeatureGenerator(PAFAConfig(
    mode="physics", allocation="uniform", budget=120, windows=(4, 8, 16),
    soft_constraints=False, adaptive_strength=0.0,
))
candidates = generator.fit_transform(train_x, train_y)
selector = GWBPSelector(GWBPConfig(
    n_splits=4, calibration_fraction=0.08, validation_fraction=0.08,
    gap=16, max_features=12, score_mode="mean", base_max_iter=60,
))
selected_train = selector.fit_transform(train_x, candidates, train_y)
selected_test = selector.transform(generator.transform(test_x, history=train_x))

# Train a forecasting model on the augmented inputs.
augmented_train = pd.concat([train_x, selected_train], axis=1)
augmented_test = pd.concat([test_x, selected_test], axis=1)
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
model.fit(augmented_train, train_y)
prediction = model.predict(augmented_test)
nrmse = np.sqrt(np.mean((test_y.to_numpy() - prediction)**2)) / np.ptp(test_y)
print("Selected features:", selector.selected_features_)
print(f"Test nRMSE: {nrmse:.4f}")
