# Recorded research results

These CSVs are copied from recorded research outputs; they are not newly rerun by the release preparation process. They contain evaluation metrics, runtimes and selected expression metadata. Original sample-level observations, labels, predictions and fitted models are not included.

| Directory | Content |
|---|---|
| `neafe_v2_full_final` | Five-model evaluation over 14 stations, paired comparisons and selected-expression summaries |
| `neafe_v2_sensitivity_final` | Validation sensitivity for candidate budget, feature cap and adaptive allocation |
| `figure_data` | Exported forecasting metrics, ablations, selection traces and runtime measurements |

Some figure-data ablations and runtime refreshes use additional configuration-development runs. These are separate measurements and should not be merged blindly into the full five-model timing protocol. `model_results.csv` records downstream model-pipeline time; `model_results_with_total_runtime.csv` additionally accounts for feature engineering. Feature engineering is performed once per station and can be reused by downstream models and seeds.

```bash
python -m experiments.verify_reference_results
```

Source paths and hashes are listed in `docs/source_manifest.json`. The MIT code license does not grant rights to any external source dataset.
