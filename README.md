# Formula 1 Podium Prediction

Milestone 1: predict whether a driver finishes on the podium using information available after qualifying and before race start. The unit is one driver entering one race; the target is `positionOrder <= 3`.

The pipeline audits 14 relational tables, resolves historical duplicate entries, constructs past-only features, compares logistic regression, random forest and a shallow Keras network, runs controlled experiments, and explains predictions with permutation importance, SHAP and LIME.

## Start here

- [Executed notebook](F1_Podium_Prediction.ipynb): code, saved outputs and all three modeling attempts.
- [Analytical report - Part I](Deliverables/8-%20Report/Analytical%20Report.pdf): three historical questions, selected-feature rationale and data limitations.
- [Technical report - Part II](Deliverables/8-%20Report/Technical%20Report.pdf): EDA, models/metrics, experiments, XAI, inference and evaluation limitations.
- [Reports guide](Deliverables/8-%20Report/Reports%20Guide.md): the two reports collectively contain all 24 notebook plots.
- [Deliverables](Deliverables): arranged results, native models and inference code.
- [Run guide](docs/RUN_GUIDE.md): setup, methodology and limitations.

## Dataset

Download the [Formula 1 World Championship archive](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). Place the 14 CSVs directly in `Dataset/` at the repository root. The supplied project snapshot covers 1950-2024; modeling is restricted to 1994-2024. A newer archive may change historical records and resulting scores.

Required tables: circuits, constructors, constructor_results, constructor_standings, drivers, driver_standings, races, results, qualifying, lap_times, pit_stops, sprint_results, seasons and status. Raw CSVs are never overwritten and are excluded from Git.

## Run locally

Use Python 3.12 (64-bit):

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
& .\.venv\Scripts\python.exe -m pip install jupyterlab
& .\.venv\Scripts\python.exe -m jupyterlab
```

Select this environment and Run All in the root notebook or `Deliverables/1- Notebook/F1 Podium Prediction.ipynb`. Dataset discovery searches the working directory and its ancestors, so both locations work without editing cells. For a custom location, set `F1_DATA_DIR` before launching the kernel. An explicit directory must contain all 14 CSVs.

The notebook is self-contained; editable source sections under `code/notebook_sections/` are for maintenance. Outputs are written under `output/` in the working directory. To rebuild and execute from the source sections:

```powershell
& .\.venv\Scripts\python.exe code\build_notebook.py --execute
```

## Kaggle

The upload-ready version is [submission/kaggle/F1_Podium_Prediction.ipynb](submission/kaggle/F1_Podium_Prediction.ipynb). It adds a visible dependency-setup cell before the same analytical code. Enable Internet, attach the F1 archive, and Save & Run All on CPU. Generated results appear under `/kaggle/working/output/`.

Published project URLs are recorded in `Deliverables/links.txt` after publication.

## Methods and results

All models share 19 engineered pre-race features. Driver/team/circuit histories use earlier races; standings are from the previous Grand Prix in the same season. Entire races stay together: training 1994-2018, validation 2019-2021, test 2022-2024. Imputation and scaling fit training only. Validation AP selects model/checkpoint/hyperparameters; validation F1 selects thresholds.

The delivered local run selects a one-hidden-layer neural network. Its 2022-2024 ROC-AUC is 0.938333, AP 0.711050 and F1 0.668235. Full results for every model/split are in [Metrics.csv](Deliverables/4-%20Model%20Comparison/Metrics.csv).

These are corrected retrospective results on a reused holdout after earlier test inspection, not a new blind test. Era drift, sparse historical qualifying, omitted weather/tyre/strategy information and repeated validation trials limit transfer. Independent probabilities do not enforce exactly three podium calls per race. XAI and historical comparisons describe associations, not causal effects.

## Inference

```powershell
& .\.venv\Scripts\python.exe "Deliverables/7- Inference/Inference.py" --record "Deliverables/7- Inference/Inference Demo Inputs.json" --example partial
```

Inputs are numeric engineered pre-race features, not raw driver/race names. Missing fields use frozen training medians/flags and are reported in the response. Complete, partial and missing-grid demonstrations are included with the saved models.
