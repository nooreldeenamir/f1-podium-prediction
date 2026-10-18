# Lab methods and changes from the original implementation

The milestone brief defines what must be delivered. Lab techniques guide the implementation where they suit a temporal, relational F1 prediction task. The separate graded assignments in the manuals are outside this milestone.

| Area | New implementation | Lab basis | Necessary adaptation |
|---|---|---|---|
| Data exploration | Pandas dimensions, missingness, summary/correlation calculations, Matplotlib and Seaborn plots | Lab 1, guided data inspection/visualization | All 14 tables, before/after programmatic evidence, immutable raw files |
| Missing values and typing | Replace sentinels on copies; numeric/date/duration conversion; explicit median `fillna` for model inputs | Lab 1, missing values and type conversion | Parse-loss audit; learn medians from training only; preserve raw analysis missingness |
| Aggregation | Readable groupby/pivot intermediate tables for circuit, home/grid/era and constructor comparisons | Lab 1, grouping and pivots | Validated multi-hop joins, documented denominators/support and race-cluster uncertainty |
| Classification baseline | sklearn LogisticRegression `.fit()`/`.predict_proba()` | Labs 2 and 3 | Regularized binary podium probability, chronological validation tuning and threshold selection |
| Scaling | Explicit StandardScaler `fit_transform(train)` and `transform(validation/test)` | Lab 3, page 6 | Identical 19 original features plus training-derived missing indicators for all models; frozen inference transform |
| Random forest | sklearn RandomForestClassifier | Lab 4, pages 8-9 | Validation candidates and same-row baseline comparison; no independent random-row split |
| Neural architecture | TensorFlow/Keras Sequential, Input, Dense ReLU hidden layer, Dense sigmoid output | Lab 3, pages 7-9 | One hidden layer keeps the network shallow; binary cross-entropy replaces the lab's multiclass loss |
| Neural training | `.compile(Adam, binary_crossentropy)` and `.fit()` with explicit chronological `validation_data` | Lab 3, pages 9-11 | Replaces PyTorch class/manual batch-loop management; no `validation_split` creating another partition |
| Early stopping | Keras EarlyStopping, `restore_best_weights=True` | Lab 3, page 10 | A short preceding callback publishes exact sklearn validation AP; max-mode AP replaces accuracy as the selection criterion |
| Training plots | Keras History for train/validation loss and exact validation AP | Lab 3, pages 11-12 | Plot actual checkpoint evidence, preserving AP definition |
| Classification evaluation | sklearn precision, recall, F1, ROC/PR measures and confusion plot | Lab 2, classification evaluation | Explicit AP versus trapezoidal PR-AUC; Brier diagnostic; every model/split and validation-selected threshold |
| SHAP | Appropriate fitted-model explainer, global summary/beeswarm and local waterfall | Lab 4, pages 9-11 | Correct transformed representation and positive class; training background, seeded validation sample, reconstruction check and stated units |
| LIME | LimeTabularExplainer plus fitted-model probability function | Lab 4, pages 10-11 | Same local record as SHAP, imputed original racing units and binary flags, wrapper applying the frozen scaler exactly once, local probability parity and fidelity reported |
| Knowledge graphs | No Neo4j deployment | Lab 5 | Pandas relational joins directly satisfy the milestone; graph infrastructure would add no required evidence |

## Additional F1 methods retained because the brief needs them

- Deduplicate at `(raceId, driverId)` with smallest positive grid then resultId; whole-row `drop_duplicates()` does not fix shared drives.
- Count positive-lap participants as starts even when W/F/Withdrew labels conflict. Without positive laps, explicit withdrawal/nonqualifying labels exclude an entry; N remains an unclassified starter. One shared rule serves Q1, Q3 and prior-start histories, with exported conflicts and independent raw-data recounts. The examiner correction changes 516 experience values and restores three Q3 starts relative to the preceding package, without changing the 19 definitions.
- Strip nationality/country strings before mapping in both analysis and features. Sample global SHAP proportionally across validation seasons using a fixed seed without labels; export membership/counts and exclude any local-only addition from global summaries.
- Use whole-race chronological train/validation/test splits rather than the labs' generic random split.
- Shift driver histories by earlier entries and constructor/circuit histories by earlier **races**; aggregate constructor-race outcomes before shifting to prevent teammate leakage.
- Use standings after the previous round of the same season; current-race standings/results/status/points/laps are not predictors.
- Keep explicit nationality-to-country mappings, common exact-grid/era standardization, pooled-support weighting and whole-race bootstrap intervals for the observational home-country question.
- Require a recorded component-failure cause, a started entry and an early-end proxy for mechanical retirement; record ambiguous cases and sensitivity definitions.
- Report each feature group's ablation, including improvements when a redundant group is removed. Fit on the same training rows with a fixed estimator configuration.
- Compare both requested preprocessing orders using training-only statistics. Percentile clipping is an extra technique; binary and missingness flags are protected from it.
- Preserve historical qualifying/training-window sensitivity, temporal feature recomputations, immutable hashes and serialized-model prediction parity.
- Explain importance as model reliance rather than causality. Keep archive-publication, correlation, era-shift and repeated-validation limitations visible.

## Persistence and student-readable code

The new notebook uses explicit Pandas/NumPy preprocessing with a small shared helper and ordinary sklearn estimators. It stores the medians, indicator schema and scaler once in `preprocessing.joblib`, and saves the neural model in native `.keras` format. The original PyTorch checkpoint and manually exported NumPy neural weights are not used. Notebook and standalone inference apply the same schema and frozen transforms.

Candidate tables, controlled experiments and report text are generated from this run's measured outputs. The superseded implementation has been removed from the project folder at your request. The current notebook/models/results are the lab-aligned version. `docs/REQUIREMENTS_CHECKLIST.md` records evidence after execution; use that matrix and the runtime verification for completion status rather than treating this design mapping as proof of successful execution.
