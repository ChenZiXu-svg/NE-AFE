# NE-AFE

**Domain-guided automated feature engineering with time-ordered incremental selection for variable renewable energy forecasting.**

NE-AFE constructs explicit features for wind and photovoltaic power forecasting in two stages:

1. **Domain-guided feature expansion (PAFA).** Variable semantics, known units and circularity guide operator eligibility. Group budgets, diversity control and numerical screening bound the candidate pool.
2. **Sequential temporal feature selection (GWBP).** Ordered base-fitting, residual-calibration and validation segments evaluate incremental residual-correction gains. Accepted candidates update residual states before the next selection step.

The selected candidates augment the original inputs. Downstream predictors are trained independently of the feature-selection surrogate.

This repository contains the **V2 implementation**, experiment entry points, output-free historical benchmark notebooks, and recorded aggregate results used during manuscript preparation. The manuscript is in preparation; no publication DOI is assigned here.

## Installation

Use Python 3.11 or later in a virtual environment.

```bash
git clone https://github.com/ChenZiXu-svg/NE-AFE.git
cd NE-AFE
```

Run the remaining commands from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core requires NumPy, pandas and scikit-learn. To run all five forecasting models and export experiment outputs:

```bash
python -m pip install -e ".[experiments]"
```

`requirements-tested.txt` records the environment used to validate this release. It is not an exact environment lockfile of the original experiments. Computation is CPU-based in the supplied model runner; runtimes depend on hardware and library versions.

## Quick start without station data

```bash
python -m examples.quickstart
python -m unittest discover -s tests -v
```

The example creates a synthetic wind-like time series, fits PAFA and GWBP on a development segment, and evaluates Raw and augmented Ridge predictors on a later held-out segment. It demonstrates the API and makes no claim to reproduce the paper's numerical results.

```python
from neafe_v2 import PAFAConfig, PAFAFeatureGenerator, GWBPConfig, GWBPSelector

generator = PAFAFeatureGenerator(PAFAConfig(
    mode="physics", allocation="uniform", budget=120,
    windows=(4, 8, 16), soft_constraints=False, adaptive_strength=0.0,
))
development_candidates = generator.fit_transform(development_x, development_y)
selector = GWBPSelector(GWBPConfig(
    n_splits=4, calibration_fraction=0.08, validation_fraction=0.08,
    gap=16, max_features=12, score_mode="mean", base_max_iter=60,
))
selected_development = selector.fit_transform(
    development_x, development_candidates, development_y,
)
test_candidates = generator.transform(test_x, history=development_x)
selected_test = selector.transform(test_candidates)
# Concatenate selected_* with the corresponding original inputs, then train a predictor.
```

The class defaults also support exploratory configurations. **Use the explicit settings above or the full experiment runner for the main paper configuration.** In particular, the paper uses hard operator eligibility and **uniform group allocation**, rather than the adaptive allocation used by the bare PAFA default.

## Reproduce the experiments

The original station measurements are not redistributed. Provide the six chronologically ordered Feather files per station described in [data/README.md](data/README.md). Targets must never be included in the input feature columns.

### Five forecasting models over 14 stations

```bash
python -m experiments.run_full --data-root data --output-dir results/neafe_v2_full_final
```

The default runner evaluates Ridge, XGBoost, MLP, LSTM and Transformer with candidate budget 120, feature cap 12, four ordered folds and seeds 42, 2024 and 2003. Ridge uses the first seed only. The initial train and validation files are concatenated into development data; the test files are held out until final evaluation.

For a smaller execution check:

```bash
python -m experiments.run_full --data-root data --output-dir results/smoke --stations wind1 --models Ridge XGBoost --seeds 42
```

### Ablation and validation-only sensitivity

```bash
python -m experiments.run_ablation --data-root data --output-dir results/neafe_v2_mvp_final_strict --n-splits 4
python -m experiments.run_sensitivity --data-root data --output-dir results/neafe_v2_sensitivity_final
```

The ablation runner compares generic/physics, hard/soft, uniform/adaptive and alternative scoring settings. These are experimental variants, not separate external feature-engineering benchmarks. The sensitivity runner uses solar1, solar2, wind1 and wind2 for configuration development.

Use a **new output directory for a new data version or experiment configuration**. The runners resume existing rows/caches and should not be pointed at outputs from an unrelated run.

### Figure data

```bash
python -m experiments.export_figure_data --data-root data --full-results results/neafe_v2_full_final --mvp-results results/neafe_v2_mvp_final_strict --output-dir results/figure_data
```

This command includes additional ablation and runtime measurements. Runtime-memory sampling uses the native Windows API on Windows and `psutil` on Unix. See [reproducibility notes](docs/REPRODUCIBILITY.md) for protocol details and [recorded results](reference_results/README.md) for the included CSVs.

## External feature-engineering benchmarks

The station-specific Featuretools, TS-Min and TS-Eff notebooks are retained in [baselines/legacy](baselines/README.md). Their outputs and original machine paths have been removed; workbook paths are supplied with `NEAFE_DATA_FILE`. Their historical tuning and windowing procedures are preserved. They are not rerun automatically by the V2 model runner. The benchmark documentation identifies these protocol differences explicitly.

## Repository layout

```text
neafe_v2/          Feature generation, metadata rules, temporal selection and predictors
experiments/       Full evaluation, ablation, sensitivity and figure-data export
examples/          Synthetic quick start and demo input generation
tests/             Chronological-fold and feature-selection checks
configs/paper.json Recorded main experimental configuration with portable paths
data/README.md     Input schema and chronology requirements
reference_results/Recorded metrics and feature-selection summaries
baselines/legacy/  Historical benchmark notebooks with outputs removed
docs/              Method and reproducibility notes
```

## License and citation

The code is released under the [MIT License](LICENSE). Data obtained from external providers remain subject to their own terms. Software citation metadata are provided in `CITATION.cff`. When reporting results, cite the immutable release or commit used, together with the paper once its bibliographic details are available.
