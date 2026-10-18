# Requirements checklist and traceability matrix

Reviewed against all 8 pages of references/Milestone 1.pdf and the full attached implementation request. The original planning matrix was created before implementation; the rows below now record actual evidence. Final review: **16 October 2026**.

**Result:** All 27 local technical/documentation rows are verified. Team preparedness and external publication/submission arrangements remain unverified or pending. No missing local technical deliverable was identified.

Checked means implemented and verified by the stated evidence; it does not certify a grade, human understanding, or external submission.

Evidence names are relative to the project root. Editable code is in `code/`; review documents are in `docs/`. All evidence filenames below are under `output/tables/`, `output/`, or `output/models/` unless they name a root document or notebook.

| ID | Requirement | Final section / artifact | Actual evidence and verification | Status |
|---|---|---|---|---|
| R01 | 14-table raw/clean EDA with plots | 1 | raw_audit.csv, clean_audit.csv, cleaning_comparison.csv, clean_numeric_summary.csv; all 14 tables and before/after figures 01-06 | [x] VERIFIED |
| R02 | Sentinels, semantic types, parsing losses | 1 | typing_audit.csv and semantic_types.csv; zero unexpected parsing losses | [x] VERIFIED |
| R03 | PK/FK, cardinality, immutable raw data | 1, 6 | primary_key_audit.csv, foreign_key_audit.csv, raw_hashes_before.json, verification.json; 14 PK and 23 FK audits pass | [x] VERIFIED |
| R04 | Unique driver-race grain and declared duplicate rule | 1 | duplicate_driver_race_audit.csv and modeling_table.csv; 91 extra rows across 85 pairs resolved; 12,358 unique modeling rows | [x] VERIFIED |
| R05 | Q1 circuit front-row conversion / pre-post 2014 | 2.1 | q1_frontrow_rates.csv, q1_era_changes.csv, q1_qualifying_grid_disagreement.csv; 490 matched eligible races; corrected nonstarter count 138; positive-lap conflicts remain starts; notebook/report answer and figures | [x] VERIFIED |
| R06 | Q2 home association controlling grid and era | 2.2 | q2_standardization_pivot.csv, q2_grid_strata.csv, q2_race_bootstrap_draws.csv, q2_home_advantage.csv; 40 common strata and 11,043 retained rows; 1,000 race resamples, 964 valid adjusted draws | [x] VERIFIED |
| R07 | Q3 constructor mechanical rates by required eras | 2.3 | q3_status_mapping.csv, q3_excluded_nonstarters.csv, q3_constructor_reliability.csv, start_classification_conflicts.csv; positive-lap withdrawals retained; McLaren/Haas/Aston Martin denominators 316/241/134; 28 entries excluded under the declared nonstart policy; sensitivities and plot | [x] VERIFIED |
| R08 | 19 pre-race features with EDA support | 3 | feature_catalog.csv and feature_evidence.csv; all 19 definitions, motivations and cutoff policies; normalized nationality/country corrects nine home flags; feature plots and correlation heatmap | [x] VERIFIED |
| R09 | Past-only driver/team/circuit form and standings | 3 | feature_lineage.csv and feature_reference_checks.csv; 480 sample references and every one of 12,358 prior-start/home feature rows independently reconciled to raw records; positive-lap evidence overrides exclusion labels | [x] VERIFIED |
| R10 | Whole-race chronological train/validation/test | 3, 4 | split_summary.csv; train 9,799 / validation 1,200 / test 1,359 rows; whole-race sets disjoint | [x] VERIFIED |
| R11 | Train-only median/flags/standardization | 4.1 | fitted_preprocessing.csv and preprocessing.joblib; 19 original inputs plus 11 training-derived indicators; medians and scaler training count independently checked | [x] VERIFIED |
| R12 | Logistic baseline + forest + shallow Keras, distinct attempts | 4.1.1, 4.1.2, 4.2 | tuning.csv and validation_attempt_comparison.csv; 6 logistic + 4 forest + 3 Keras candidates; separate fitted attempts and baseline comparisons | [x] VERIFIED |
| R13 | Keras fit, AP callback, EarlyStopping restore best | 4.2 | neural_training_histories.csv; exact AP callback precedes stopper; restored checkpoint asserted; selected 32-unit network at epoch 140 of 160; training/validation loss and AP curves | [x] VERIFIED |
| R14 | Validation-only tuning/model/threshold choice | 4 | selection_frozen.json written before corrected holdout scoring; validation AP chooses model and validation F1 chooses thresholds; original split/candidate grids/rules retained; prior test inspection disclosed | [x] VERIFIED |
| R15 | All model/split metrics and rationale | 4.4 | metrics.csv; 9 complete model/split rows, ROC-AUC/AP/PR-AUC/F1/precision/recall/Brier/threshold; holdout ROC/PR and confusion plots | [x] VERIFIED |
| R16 | Every feature-group ablation retrained | 4.3.1 | ablation.csv; reference plus all 5 group removals; fixed logistic configuration and same rows; positive removal effects reported honestly | [x] VERIFIED |
| R17 | Two preprocessing-order comparisons | 4.3.2 | preprocessing.csv and preprocessing_value_effects.csv; 4 fitted variants; all 13 binary/indicator columns protected; A plots qualifying_gap_pct with 1,987 missing training values | [x] VERIFIED |
| R18 | Historical qualifying/training-window sensitivity | 4.3.3 | training_window_sensitivity.csv and plot; fixed configuration/validation; train-only preprocessing refit on shorter window | [x] VERIFIED |
| R19 | Global permutation + SHAP | 4.5 | permutation_importance.csv, shap_importance.csv, shap_sample.csv and shap_sample_by_year.csv; season-proportional global sample (2019:21 / 2020:17 / 2021:22); training background; local-only row excluded from global averages/plots; approximation disclosed | [x] VERIFIED |
| R20 | Same-prediction local SHAP + LIME | 4.5 | local_shap.csv, lime_local.csv, waterfall/LIME plots and explanation_checks.json; same race 1027 / driver 830; SHAP reconstruction error 1.1e-16; LIME R-squared 0.3993 explicitly limited | [x] VERIFIED |
| R21 | Complete/partial/missing-grid inference and validation | 4.6 | inference.py, inference_demo_inputs.json and inference_examples.csv; complete/partial/missing-grid demos; unknown keys, NaN/infinity, malformed values and contradictory flags rejected | [x] VERIFIED |
| R22 | Native Keras/sklearn model persistence | 4.6 | models/*.joblib, native shallow_neural_network.keras, metadata.json and reload_parity.csv; all native reloads checked; all 9 split scores independently recomputed | [x] VERIFIED |
| R23 | Self-contained Run All notebook, visible outputs | all | F1_Podium_Prediction.ipynb, execution_run.json and delivery_checks.json; 55 cells, all 27 code cells run 1-27 in fresh kernel; no saved errors; exact source parity | [x] VERIFIED |
| R24 | Every plot and all 3 written answers in notebook AND PDF | 5 | Deliverables/8- Report contains Part I (7 pages, 6 figures) and Part II (17 pages, 18 figures); all three answers and all 24 notebook plots are included across the two visually reviewed PDFs | [x] VERIFIED |
| R25 | Report feature validity, comparisons, limits and XAI | 5 | Part I covers historical answers and feature validity; Part II covers model mechanics/metrics, experiments, global/local XAI, inference and evaluation limitations | [x] VERIFIED |
| R26 | Readable lab mapping, run instructions and dependencies | README, LAB_METHOD_MAPPING | README.md, LAB_METHOD_MAPPING.md, requirements.txt and requirements-tested.txt; local execution verified; Kaggle instructions supplied but actual Kaggle run pending | [x] VERIFIED |
| R27 | Evaluation preparation / defend choices | README, report | Notebook explanations, report rationale and lab mapping supplied; individual understanding and oral defense cannot be verified from files | [ ] HUMAN CHECK |
| R28 | Kaggle/GitHub/form/PDF submission and evaluation slot | checklist | PDF ready locally; accessible Kaggle/GitHub links, course form and evaluation-slot arrangements need account/human actions and were not performed | [ ] PENDING |
| R29 | Retain final work and raw data after requested cleanup | Project organization | The lab-aligned notebook, report, models, figures, tables and all 14 raw CSVs were preserved during reorganization; obsolete implementation and temporary files were removed at your request | [x] VERIFIED |

## Execution and delivery checks

- [x] Fresh-kernel Run All completed all 27 code cells in 143.64 seconds. See [execution_run.json](../output/execution_run.json).
- [x] The repeat corrective training reproduced all 13 tuning results exactly and all nine metric rows within numerical precision (maximum difference 5.6e-17).
- [x] Notebook sources match all six editable source sections; the notebook runs independently of those files.
- [x] All 14 raw CSV hashes, key checks, joins, unique grain, chronological race groups and past-only lineage checks pass.
- [x] Independent raw-data recounts check all 12,358 prior-start values, all Q3 denominators and all 12,358 normalized home flags. Positive-lap participants cannot be excluded as DNS.
- [x] SHAP sample membership and season counts are independently checked against validation rows; the global sample is proportional across all three seasons.
- [x] All three native models reload; all nine model/split scores and validation-F1 threshold optima are independently checked. Complete/partial/grid inference and schema rejection pass.
- [x] The six figures selected for the focused analytical report match PDF-embedded pixels. The report was visually reviewed after the scope revision. See [pdf_visual_qa.json](../output/pdf_visual_qa.json).

## Measured selected-model results

The validation-selected model is **Shallow neural network**: one 32-unit ReLU hidden layer, one sigmoid output, L2 0.01 and Adam learning rate 0.003. Exact validation AP is 0.748718; the restored checkpoint is epoch 140. The frozen validation-F1 threshold is 0.347110.

| Holdout metric, 2022-2024 | Result |
|---|---|
| ROC-AUC | 0.938333 |
| Average precision (AP) | 0.711050 |
| Trapezoidal PR-AUC | 0.709687 |
| F1 | 0.668235 |
| Precision | 0.642534 |
| Recall | 0.696078 |
| Brier | 0.069970 |

Scores come from the corrective rerun. Model choice, hyperparameters, checkpoint and thresholds use validation only. This follows an examiner audit that inspected earlier test results: the same holdout is reused, rather than claimed as a new blind test. Full evidence: [metrics.csv](../output/tables/metrics.csv) and [delivery_checks.json](../output/delivery_checks.json).

## Lab alignment and justified additions

- [x] Lab 1: Pandas missing values, explicit aggregation/pivots, correlations and plots.
- [x] Lab 2: sklearn logistic regression and classification metrics.
- [x] Lab 3: explicit StandardScaler, Keras Sequential/Dense, compile/fit, History and EarlyStopping restoring best weights. A short callback supplies exact AP before the stopper.
- [x] Lab 4: sklearn forest, global SHAP, local waterfall and same-record LIME with correct representations.
- [x] Lab 5 reviewed; Neo4j is unnecessary for the required relational questions. Separate graded lab assignments are outside this implementation.
- [x] Necessary additions are explained: chronological splits, past-only histories/standings, key checks, whole-race bootstrap, controlled experiments, clipping and native model reloads.

## Correctness changes and limitations

- Shared race-start rule: positive recorded laps override W/F/withdrawal labels; without positive laps, explicit withdrawal/nonqualifying labels exclude an entry. N remains an unclassified starter. Zero grid/laps alone does not prove DNS. All 24 conflicting raw driver-race labels are exported in start_classification_conflicts.csv.
- Relative to the preceding delivered ZIP, 516 driver_prior_starts values change (310 training / 45 validation / 161 test); three Q3 starts are restored; Q1 retains the same 490 eligible races but its nonstarter count changes from 145 to 138.
- Nationality/country normalization corrects nine Colapinto home flags; there are no missing home flags in the current modeling table.
- Global SHAP uses a seeded season-proportional random sample, with 21/17/22 rows from 2019/2020/2021. Membership/counts are exported; any local-only addition does not affect the global beeswarm or mean absolute importance.
- Kernel SHAP approximates the selected network with a training background and limited coalition budget. Units are podium probability; local reconstruction is checked. Local LIME R-squared is 0.3993; its weights remain approximate and noncausal.
- This corrective rerun changes data semantics and sampling, not split boundaries, feature definitions, candidate grids or selection rules. Earlier test results were inspected; the corrected scores are not a new independent blind evaluation.
- Final grid remains an archive assumption requiring a genuine pre-start snapshot in live use. Nationality is a home proxy; uncertain zero-lap starts, mechanical status and historical qualifying/era drift limit transfer.
- Independent driver predictions do not enforce exactly three podium calls. Repeated validation diagnostics may overfit that era; scores do not guarantee future performance.

## Human/account steps still pending

- [ ] Run and save an accessible Kaggle notebook version with outputs; actual Kaggle execution is not verified here.
- [ ] Publish and verify an accessible GitHub repository with the new code and instructions.
- [ ] Submit the Kaggle link, GitHub link and PDF through the [course form](https://forms.gle/esDxHsvfzffYHVAZ9).
- [ ] Confirm every team member can defend all choices and results using evidence.
- [ ] Check evaluation announcements and reserve the required slot when available.

The brief deadline is **18 October 2026, 11:59 PM**; it gives no timezone. No external publishing or submission was performed.

## Organization update

Only the lab-aligned implementation remains in the project folder. The executed notebook is at the root, tools and source sections are under `code/`, review documents are under `docs/`, and course PDFs are under `references/`. The current code/results ZIP is under `submission/`. Old temporary files and superseded preservation/package manifests were removed. The examiner fixes subsequently regenerated the notebook, models, scores, explanations and report together; the submission ZIP contains the corrected evidence. Temporary backups and visual renders are excluded from submission.
