# Formula 1 Podium Prediction - lab-aligned implementation

This is a new implementation of Milestone 1. It reads the supplied 14-table F1 archive, constructs one row per `(raceId, driverId)`, and predicts `positionOrder <= 3` using information available after qualifying and final-grid publication, before race start. This is the sole current implementation. The superseded version and temporary material were removed at your request; the supplied dataset and course references are retained.

## Open the results

- `F1_Podium_Prediction.ipynb`: self-contained notebook with the completed Run All outputs. Source sections are provided for maintenance; the notebook does not import them.
- `Deliverables/8- Report/Analytical Report.pdf` and `Technical Report.pdf`: Part I contains the three answers and feature validity; Part II contains the remaining computational evidence. Together they include all 24 notebook plots.
- `docs/REQUIREMENTS_CHECKLIST.md`: requirements matrix with evidence and verified/pending status.
- `docs/LAB_METHOD_MAPPING.md`: what changed and how each method relates to the manuals.

The final execution and visual-review evidence is in `output/verification.json` and the checker's output. A notebook generated without `--execute` must still be run before it is treated as a verified result.

The delivered checks are recorded in `output/delivery_checks.json`; `output/execution_run.json` records the successful fresh-kernel execution, and `output/pdf_visual_qa.json` identifies the exact visually reviewed PDF by SHA256. If you regenerate the report, its timestamp changes its bytes: repeat the visual review and refresh that QA record before claiming the regenerated PDF was visually verified. The checker detects a stale QA hash.

## Run locally on Windows

Use **Python 3.12, 64-bit**, and run commands from the project root directory (`ACL Project`). The notebook searches the working directory and its ancestors for the complete `Dataset/` folder, including from `Deliverables/1- Notebook/`. An explicit `F1_DATA_DIR` must contain all 14 CSVs. This command creates a normal isolated environment for a new installation:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe code\build_notebook.py --execute
& .\.venv\Scripts\python.exe code\check_deliverables.py
```

The builder runs in a fresh kernel and saves executed outputs. It creates a temporary kernelspec pointing to the Python executable used above; delivered notebook metadata uses `python3`. Model fitting is CPU-compatible. All three models are trained afresh; no original fitted models or scores are loaded.

For interactive use:

```powershell
& .\.venv\Scripts\python.exe -m pip install jupyterlab
& .\.venv\Scripts\python.exe -m jupyterlab
```

Open `F1_Podium_Prediction.ipynb`, select the environment containing the requirements, and choose **Restart Kernel and Run All**. No cell edits are needed. For another dataset location, set the variable before starting the builder or Jupyter:

```powershell
$env:F1_DATA_DIR = 'C:\path\to\14_csv_files'
```

`requirements-tested.txt`, generated with the delivered run, records exact direct-package versions. It can be used instead of `requirements.txt` to reproduce that environment more closely. Seeds and deterministic split boundaries are fixed; exact floating-point results can still vary with platform/library versions.

The delivered local run used a dedicated `.venv` created with `--system-site-packages` to reuse the read-only bundled analysis libraries, then installed TensorFlow 2.20.0, SHAP 0.53.0 and LIME 0.2.0.1 into that environment. The ordinary isolated setup above is suitable for another computer; it does not depend on the Codex runtime paths.

## Run on Kaggle

1. Create a Kaggle notebook and import `submission/kaggle/F1_Podium_Prediction.ipynb`. This variant adds two visible setup cells before the unchanged analytical notebook.
2. Attach the milestone's [Formula 1 archive](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), or upload the supplied 14 CSV files as a dataset. The delivered evidence uses the supplied archive, which ends in 2024. Analyses are capped at 2024 even if a newer archive is attached; changed historical records can change results.
3. Select a Python CPU session and explicitly enable Internet after import. The visible setup cell retains Kaggle's preloaded NumPy/pandas/sklearn/TensorFlow/Keras stack and installs only missing extras without dependencies. Do not replace preloaded numerical libraries inside a running Kaggle kernel: their in-memory binaries can disagree with newly installed files. The actual package versions are recorded in the outputs; they can differ from the pinned Python 3.12 local environment and slightly change fitted results. No analytical cells need changes.
4. The notebook discovers the attached folder containing all 14 tables. If there are multiple matching inputs, set `F1_DATA_DIR` to the intended folder in the session environment before execution.
5. Choose **Run All**, inspect the runtime verification and generated report, then save a version containing outputs. Kaggle writes generated evidence to `/kaggle/working/output`; local execution writes to this directory's `output`.

Uploading the notebook is enough to run it: source sections, the builder and standalone inference script are not notebook runtime dependencies. The published notebook URL is recorded in `Deliverables/links.txt`; `submission/kaggle/` contains a version with its visible dependency-setup cell.

## Pipeline and evidence

| Requirement | Implementation and evidence |
|---|---|
| Audit and cleaning | All 14 tables, sentinels/blanks, semantic types and parsing losses, primary/foreign keys, validated joins, before/after plots and immutable raw SHA256 hashes |
| Unique grain | Smallest positive grid then smallest resultId; uses no finish/target information; duplicate audit exported |
| Three relational questions | Matched qualifying/grid front rows before/after 2014; exact-grid/era standardized home/away comparison with race bootstrap; mechanical failures per race start with early-end gate and sensitivity |
| Pre-race feature engineering | Same 19 engineered features for all models; shifted driver/constructor/circuit histories, previous-round season standings, cutoff catalog and independent lineage checks |
| Partitions | Whole races; train 1994-2018, validation 2019-2021, test 2022-2024 |
| Preprocessing | Explicit training medians, fixed training-derived missing indicators, StandardScaler fit on training only, same input representation for all main models |
| Three visible attempts | sklearn logistic baseline, sklearn forest, shallow TensorFlow/Keras Sequential with Dense/ReLU/sigmoid, Adam and binary cross-entropy |
| Early stopping | Keras fit/History and EarlyStopping restoring the best exact validation AP checkpoint; AP callback precedes the stopper |
| Evaluation | Every model on train/validation/test; ROC-AUC, exact AP, trapezoidal PR-AUC, precision, recall, F1, Brier and frozen threshold; ROC/PR and confusion plots |
| Experiments | All five feature groups individually ablated; two paired preprocessing-order comparisons with protected binary flags and measured transformed-value effects; training-window sensitivity |
| Explainability | Selected-model validation permutation importance, global SHAP, local waterfall and same-record LIME, SHAP reconstruction and LIME fidelity |
| Inference | Complete, partial and missing-grid demos; schema checking; frozen native models/preprocessing; reloaded prediction parity |
| Report and verification | Report generated from actual evidence; complete figure manifest; fresh-kernel execution, runtime checks and visual PDF review |

The brief lists ROC-AUC, PR-AUC and F1 but later refers to five metrics. Precision and recall are also supplied, and AP is explicitly distinguished from trapezoidal PR-AUC. Both local SHAP and LIME cover the prose/checklist variation.

## Output files

- `output/modeling_table.csv`: unique driver-race features, metadata and label.
- `output/tables/`: audits, data-engineering answers, feature catalog/evidence/lineage, candidate comparisons, metrics, ablations, preprocessing effects, sensitivity, explanations and inference demonstrations.
- `output/figures/`: all notebook plots, collectively embedded across the two PDFs in `Deliverables/8- Report/`.
- `output/models/`: sklearn estimators, shared `preprocessing.joblib`, native `shallow_neural_network.keras` and model/schema/threshold/version metadata.
- `output/inference_demo_inputs.json`: exact complete, partial and missing-grid example records.
- `output/lime_local_explanation.html`: local interactive explanation.
- `output/analytical_report.md`, `output/pdf/analytical_report.pdf` and `output/report_figure_manifest.json`: generated report and figure inventory.
- `output/verification.json`: runtime assertions and hashes; `check_deliverables.py` checks the completed artifacts independently.

After notebook execution, standalone inference uses the persisted artifacts:

```powershell
& .\.venv\Scripts\python.exe code\inference.py --record output/inference_demo_inputs.json --example partial
& .\.venv\Scripts\python.exe code\inference.py --help
```

Inputs are the **engineered features** in the catalog. Historical quantities must already be calculated using strictly earlier races. This interface does not fetch features from a driver's name or race ID. Omitted/null values use frozen training medians and flags; missing or nonpositive grid derives `grid_unassigned=1`, while positive grid derives `0`. Unknown fields, nonfinite inputs and contradictory flags raise errors. Returned data includes probability, classification, threshold and imputed fields. Keras loading uses its native model format rather than a manual NumPy network.

## Interpretation and limitations

Evaluation is sequential: an earlier completed validation/test race can inform the next race's historical features, while training medians, scaling and model weights stay frozen. The final grid comes from a retrospective archive and is assumed to represent the published pre-start grid; a live system must acquire an actual cutoff snapshot. Missing qualifying coverage, scoring/regulation changes and archive entry definitions limit transfer across eras.

Model choice, checkpoints, hyperparameters and thresholds use validation only. Test scores are reported after decisions are frozen and do not trigger redesign. Repeated validation experiments can still overfit that window. Binary predictions do not enforce exactly three podium selections per race. Home-country comparisons and XAI show associations, not causal effects; correlated inputs and unrealistic perturbations limit interpretation. Missing weather, tyres and practice/strategy data leave substantial uncertainty.

Race-start classification is shared by Q1, Q3 and historical experience. Positive recorded laps establish participation even when W/F or Withdrew labels conflict; those rows remain starts. Without positive laps, explicit withdrawal/nonqualifying labels are excluded. N means unclassified and remains a start; zero grid/laps alone does not prove DNS. The positive-lap conflicts are exported in `start_classification_conflicts.csv`. This audit correction changes 516 prior-start values and restores three Q3 denominator entries relative to the preceding delivered package. Nationality/country strings are normalized consistently, correcting nine Colapinto home flags. Global SHAP uses a seeded, season-proportional validation sample, exports membership/counts, and excludes any separately appended local-only case from global summaries.

This is a corrective rerun after the examiner audit inspected earlier test results. Split boundaries, all 19 feature definitions, candidate grids and validation selection rules are unchanged. Reported test scores are a corrected evaluation on the same 2022-2024 holdout, not a new blind test. All models and evidence are regenerated from the corrected inputs.

## Submission and evaluation steps still requiring people/accounts

Publication URLs are recorded in `Deliverables/links.txt`. Course-form submission and evaluation arrangements are separate steps.

- [ ] Publish and verify an accessible Kaggle notebook version with successful Run All outputs.
- [ ] Publish and verify an accessible GitHub repository with the code and dependency instructions.
- [ ] Submit both accessible links and the PDF through the [course form](https://forms.gle/esDxHsvfzffYHVAZ9).
- [ ] Confirm every team member can explain cleaning, lineage, model mechanics, primary/secondary metrics, experiments and limitations using the actual evidence.
- [ ] Check the evaluation announcement and reserve a slot when available.

The milestone deadline is **18 October 2026 at 11:59 PM**; the brief does not specify a timezone. Evaluation dates and reservation details are to be announced separately.

For GitHub, include the notebook, code/ source sections and tools, requirements, README, docs/ lab mapping and checklist. Include the PDF and useful evidence tables. Keep `.venv`, caches and temporary QA renders out of the repository; attach the archive on Kaggle instead of duplicating all raw data in Git.

## Organized project locations

The current notebook is at the project root. Editable source sections are in `code/notebook_sections/`; the builder, delivery checker and standalone inference script are in `code/`. Course materials are in `references/`. Review the PDF at `output/pdf/analytical_report.pdf`; the current code/results ZIP is in `submission/`.

Current verification checks dataset hashes, native models, source parity, corrected start/home features, SHAP sampling and PDF evidence. The latest fresh-kernel execution was performed from this project root. Temporary correction backups, execution files and PDF review renders are kept under ignored `tmp/` and excluded from the submission ZIP.
