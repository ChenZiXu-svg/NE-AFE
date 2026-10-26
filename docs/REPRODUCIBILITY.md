# Reproducibility notes

## Main configuration

`configs/paper.json` records the configuration from the completed 14-station V2 run. Original absolute machine paths have been replaced by portable relative paths.

| Setting | Main value |
|---|---|
| Candidate generator | Physics mode, hard eligibility, uniform allocation |
| Candidate budget | 120 before numerical filtering |
| Historical windows | 4, 8, 16 observations |
| Maximum selected candidates | 12 |
| Selection folds | 4, expanding base segment |
| Calibration / validation size | Each targets 8% of development observations per fold, subject to minimum-size rules |
| Gaps | 16 observations between base/calibration and calibration/validation |
| Gain aggregation | Mean relative residual-correction gain |
| Required positive-fold fraction | At least 0.5 |
| Correction shrinkage / ridge | 0.5 / 0.0001 |
| Surrogate | HistGradientBoostingRegressor, 60 iterations, 15 leaves |
| Downstream models | Ridge, XGBoost, MLP, LSTM, Transformer |
| Repeated seeds | 42, 2024, 2003; Ridge uses seed 42 only |
| Model validation fraction | Final 15% of development for XGBoost/neural training |
| Neural training | AdamW, learning rate 0.001, batch 256, at most 10 epochs, patience 2 |
| Neural training sample cap | 8000 |
| Sequence length | 12 |
| CPU thread count | 4 in the full model configuration |

PAFA pilot scores are retained as diagnostics. Uniform allocation does not adapt group quotas to those scores. Bare class defaults include other experimental options; they do not all equal the main configuration.

## What can be reproduced without original data

The synthetic example, existing chronology tests and recorded-result checks run without station measurements. The CSVs in `reference_results` permit checking reported aggregations. Reproducing station predictions requires the original prepared input files. This release does not claim bit-for-bit reproduction without those files and the original software/hardware environment.

```bash
python -m experiments.verify_reference_results
```

## Metrics and aggregation

The supplied evaluator defines nRMSE as RMSE divided by the observed target range of the evaluation segment. Constant-range segments yield NaN. This normalization is an evaluation calculation and does not fit the forecasting model to test labels.

Average repeated seeds within each station before averaging across stations. Mean station-level relative improvement is the average of `(Raw - NE-AFE) / Raw` computed separately for each station; it differs from the relative decrease in the overall mean nRMSE.

Selection scores summarize the sequential residual-correction proxy. They are not downstream model feature importances or additive contributions to final test error. Finite temporal folds provide empirical evidence, not a guarantee of future gains under arbitrary distribution shifts.

## Implementation boundaries

The seven files under `neafe_v2/` are copied unchanged from the research implementation for this release. `docs/source_manifest.json` records the source hashes. Experiment paths were made portable; runtime-memory export adds Unix support. Algorithmic behavior, configuration construction and fitting routines remain unchanged.

PAFA metadata inference, numerical screening and some GWBP candidate sanitization use the development data supplied to `fit`. The final test segment is excluded from those fitted decisions. The chronological proxy evaluation should not be interpreted as a guarantee that every preliminary transformation is independently refitted inside every inner fold. For a different deployment or nested-validation protocol, audit this distinction explicitly.

Each new experiment should use a fresh output directory. The existing runners resume cached features and completed model rows; their cache keys do not fingerprint arbitrary changes to input data or all model settings. Keep the release/commit, input data version and configuration together when archiving a run.

## Historical external benchmarks

The Featuretools and tsfresh notebooks preserve the historical experiments, including their own window construction, sample selection and Optuna searches. Their results should be interpreted using those actual procedures. They do not implement the same end-to-end runner as V2. See [the baseline notes](../baselines/README.md).

## Software citation

Use the repository URL for availability, and a version tag or full commit hash to identify the exact implementation. `CITATION.cff` describes the software. Replace or supplement it with the paper's final citation once accepted; this release does not assign a journal status or DOI.
