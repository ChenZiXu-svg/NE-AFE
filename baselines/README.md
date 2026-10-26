# Historical external feature-engineering benchmarks

`legacy/featuretools`, `legacy/ts_min` and `legacy/ts_eff` contain the original station notebooks for Featuretools, tsfresh MinimalFCParameters and tsfresh EfficientFCParameters. There are 14 notebooks for each method. Notebook outputs, execution counts and machine-specific metadata are removed. The original workbook path assignment now reads `os.environ["NEAFE_DATA_FILE"]`; the remaining code is preserved.

Install the optional dependencies:

```bash
python -m pip install -e ".[experiments,baselines]"
```

Set `NEAFE_DATA_FILE` to the original station workbook in your shell before starting Jupyter. For example, PowerShell:

```powershell
$env:NEAFE_DATA_FILE = 'D:\data\station.xlsx'
python -m jupyterlab
```

Run each notebook in a separate output working directory, because it writes station-specific Feather files using relative filenames. Select the notebook corresponding to the station and workbook sheet. The notebooks retain their original English and Chinese comments.

These are archived benchmark procedures, rather than a newly unified comparison pipeline. They include station-specific details and Optuna tuning that differ from the fixed V2 forecasting configuration. The tsfresh templates may operate on historical power windows, use training subsets and limit retained columns. Check forecast-time alignment, target-derived feature availability and sample alignment for the intended task before treating a new run as a matched comparison. Optional dependencies and original spreadsheets are required; the archived notebooks are not part of the core smoke-test suite.

The research project's older generic helper classes and NE-AFE V1 scripts are omitted to keep the main implementation unambiguous. The retained notebooks are the concrete historical benchmark experiments used in the project.
