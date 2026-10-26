# Station input data

Original measurements, weather forecasts, spreadsheets and Feather files are not distributed with this repository. Obtain the station data from their original provider under the applicable terms, or supply your own data using this schema. A stable provider download URL and a redistribution license have not been verified in this release.

For station `wind1`, place these files under your chosen `--data-root`:

```text
train_raw_wind1.feather   train_y_wind1.feather
valid_raw_wind1.feather   valid_y_wind1.feather
test_raw_wind1.feather    test_y_wind1.feather
```

Use the same naming pattern for `wind1`–`wind6` and `solar1`–`solar8`.

## Schema

Each `*_raw_*` file is a pandas-compatible numeric DataFrame containing only predictor inputs. Each `*_y_*` file contains one numeric target column; the loader reads its first column. Rows must align within each pair. Column names, order and units must match across splits. Provide at least one predictor, finite target values and non-empty segments. Duplicate column names are unsupported.

The main protocol originally splits observations chronologically in a 60:20:20 ratio. The full runner concatenates train and validation into development data and retains test data for final evaluation. It consumes already prepared files and does not reconstruct the original spreadsheet preprocessing, missing-row removal or forecast horizon.

The series have a nominal 15-minute resolution. All supplied temporal windows and gaps count **observations**, not elapsed clock time. Preserve chronology after any preprocessing and account for gaps in timestamps when interpreting windows.

## Variable metadata

The current implementation infers metadata from column names in `neafe_v2/metadata.py`:

| Example name | Interpretation | Expected unit |
|---|---|---|
| `WS10`, `wind_speed` | Wind speed | m/s |
| `WD10`, `wind_direction` | Circular wind direction | degrees |
| `Irradiance`, `Radiation` | Solar irradiance | W/m² |
| `Temperature`, `Temp` | Temperature | °C |
| `Humidity` | Relative humidity | percent |
| `Lag1`, `power_lag1` | Available historical power | MW |

Unknown names receive generic metadata. These heuristics do not convert units or independently establish input availability. Verify the mapping for your own columns, and adapt `metadata.py` if your schema differs. The main operator rules apply to recognized physical quantities; they do not establish a full physical simulator.

## Forecast-time availability

Each input row must contain only information available at its forecast issue time. Construct target lags before splitting, using strictly earlier target observations. Include current-row meteorological values only when they are available forecasts or already observed values for the stated task.

`generator.transform(test_x, history=development_x)` computes temporal features from the available covariate stream and preceding history. For rolling-origin evaluation, earlier test-period observations can enter later forecasts only when they would have become available operationally. The repository does not simulate delayed observations or multi-step recursive target-lag updates automatically.

## Synthetic files for an execution check

```bash
python -m examples.make_demo_data --output-dir data/demo
python -m experiments.run_full --data-root data/demo --output-dir results/demo --stations wind1 --models Ridge XGBoost --seeds 42
```

The generated files are synthetic and are ignored by Git.
