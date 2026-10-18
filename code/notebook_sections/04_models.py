# %% [markdown]
# ## 4. Lab-aligned models, experiments, explanation, and inference
#
# We predict podium versus no podium. All three attempts use the same original
# 19 features and driver-race rows. Entire races stay together: 1994-2018 trains,
# 2019-2021 validates, and 2022-2024 is the future-season holdout, isolated from
# fitting and selection. This corrective rerun follows an audit of earlier test
# results; it is a corrected evaluation on the same holdout, not a new blind test.
# Split boundaries, candidate grids and selection rules are unchanged. Random
# row splitting from the classroom examples would let teammates and nearby
# events cross partitions, and would not represent predicting future seasons.
#
# Lab 1 supplies median filling; Lab 3 supplies explicit StandardScaler,
# Sequential/Dense, compile(), fit(), EarlyStopping, and History plots. Lab 2
# supplies sklearn classification and its metrics; Lab 4 supplies forest, SHAP,
# and LIME. The temporal split, historical shifts, exact AP callback, and
# preprocessing experiments adapt those methods to this milestone.
#
# **Declared selection rule:** maximize validation average precision (AP), a
# useful minority-class ranking measure. AP is the recall-weighted sum of
# precision, while trapezoidal PR-AUC linearly interpolates the PR curve. We
# report both, alongside ROC-AUC (overall ranking), precision, recall and F1
# (decisions at a threshold). Brier score additionally describes probability
# error. Thresholds maximize validation F1, with larger thresholds breaking ties.
# Accuracy alone could hide poor podium detection. No holdout value is used for
# hyperparameters, checkpoint selection, thresholds, features, or experiments.

# %%
import json
import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    roc_curve, auc, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, brier_score_loss,
)

TABLES = OUT / "tables"
MODELS_DIR = OUT / "models"
TABLES.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_figures = []
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

train_rows = model_table.loc[model_table["year"].between(1994, 2018)].copy()
val_rows = model_table.loc[model_table["year"].between(2019, 2021)].copy()
test_rows = model_table.loc[model_table["year"].between(2022, 2024)].copy()
X_train = train_rows[FEATURES].astype(float)
X_val = val_rows[FEATURES].astype(float)
X_test = test_rows[FEATURES].astype(float)
y_train = train_rows["y"].to_numpy(dtype=int)
y_val = val_rows["y"].to_numpy(dtype=int)
y_test = test_rows["y"].to_numpy(dtype=int)
assert len(X_train) and len(X_val) and len(X_test)
assert set(train_rows.raceId).isdisjoint(val_rows.raceId)
assert set(train_rows.raceId).isdisjoint(test_rows.raceId)
assert set(val_rows.raceId).isdisjoint(test_rows.raceId)
assert not X_train.isna().all().any()
assert not np.isinf(model_table[FEATURES].to_numpy(dtype=float)).any()
assert not set(FEATURES) & {"y", "positionOrder", "points", "statusId", "laps", "milliseconds"}
assert all(frame.columns.tolist() == FEATURES for frame in [X_train, X_val, X_test])
for frame in [X_train, X_val, X_test]:
    assert np.array_equal(frame.grid_unassigned.to_numpy(), frame.grid.isna().astype(int).to_numpy())

split_summary = pd.DataFrame([
    {"split": name, "years": years, "rows": len(rows), "races": rows.raceId.nunique(),
     "podiums": int(rows.y.sum()), "podium_rate": rows.y.mean()}
    for name, years, rows in [("train", "1994-2018", train_rows),
                              ("validation", "2019-2021", val_rows),
                              ("test", "2022-2024", test_rows)]
])
split_summary.to_csv(TABLES / "split_summary.csv", index=False)
display(split_summary)

# %% [markdown]
# ### 4.1 Training-only preprocessing shared by all attempts
#
# We retain raw missingness in X. Fit medians on training only, fill with Pandas
# fillna(), append a missingness indicator for each column missing in training,
# and fit StandardScaler to that completed training frame. Validation, holdout,
# and inference call transform() with these same stored statistics and column
# order. The dictionary below stores ordinary objects instead of a nested
# preprocessing pipeline or custom transformer. Binary flags are standardized
# consistently with the other inputs; scaling does not erase their two values.
# The already derived grid_unassigned flag is never median-filled in inference.

# %%
def append_missing_flags(filled_frame, original_frame, indicator_features):
    """Keep the missingness observed before filling, in an explicit column order."""
    completed = filled_frame.copy()
    for feature in indicator_features:
        completed["missing_" + feature] = original_frame[feature].isna().astype(float)
    return completed


def fit_preprocessing(training_frame):
    """Fit only on the supplied training rows; return portable library objects."""
    features = list(training_frame.columns)
    medians = training_frame.median()
    assert medians.notna().all(), "A training feature has no observed value."
    indicator_features = [f for f in features if training_frame[f].isna().any()]
    completed = append_missing_flags(training_frame.fillna(medians), training_frame, indicator_features)
    scaler = StandardScaler()
    scaler.fit(completed)
    return {"features": features, "medians": medians,
            "indicator_features": indicator_features,
            "transformed_features": list(completed.columns), "scaler": scaler}


def transform_frame(frame, bundle):
    """Apply frozen training medians, indicator schema, and scaler; never refit."""
    original = frame[bundle["features"]].astype(float)
    completed = append_missing_flags(original.fillna(bundle["medians"]), original,
                                     bundle["indicator_features"])
    assert list(completed.columns) == bundle["transformed_features"]
    transformed = bundle["scaler"].transform(completed)
    assert np.isfinite(transformed).all()
    return transformed


preprocessing = fit_preprocessing(X_train)
Z_train = transform_frame(X_train, preprocessing)
Z_val = transform_frame(X_val, preprocessing)
transformed_names = preprocessing["transformed_features"]
assert len(transformed_names) == len(FEATURES) + len(preprocessing["indicator_features"])
preprocessing_audit = pd.DataFrame({
    "feature": FEATURES,
    "training_median": preprocessing["medians"].reindex(FEATURES).values,
    "training_missing_rows": X_train.isna().sum().values,
    "has_missing_indicator": [f in preprocessing["indicator_features"] for f in FEATURES],
})
preprocessing_audit.to_csv(TABLES / "fitted_preprocessing.csv", index=False)
display(preprocessing_audit)


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def metric_row(y, probabilities, threshold):
    precision, recall, _ = precision_recall_curve(y, probabilities)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {"ROC_AUC": float(roc_auc_score(y, probabilities)),
            "AP": float(average_precision_score(y, probabilities)),
            "PR_AUC": float(auc(recall, precision)),
            "F1": float(f1_score(y, predictions, zero_division=0)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "Brier": float(brier_score_loss(y, probabilities)), "threshold": float(threshold)}


models, best_configs, validation_probabilities = {}, {}, {}
tuning_records = []
model_start = time.perf_counter()


def show_attempt_comparison(model_name):
    baseline_ap = average_precision_score(y_val, validation_probabilities["Logistic regression"])
    comparison = pd.DataFrame([
        {"model": name, **metric_row(y_val, p, choose_threshold(y_val, p)),
         "delta_AP_vs_baseline": float(average_precision_score(y_val, p) - baseline_ap)}
        for name, p in validation_probabilities.items()
    ])
    display(comparison.round(4))
    current = comparison.set_index("model").loc[model_name]
    display(Markdown(f"**{model_name}:** validation AP **{current.AP:.4f}**, change versus "
                     f"logistic baseline **{current.delta_AP_vs_baseline:+.4f}**; "
                     f"F1 **{current.F1:.4f}** at validation-selected threshold "
                     f"**{current.threshold:.4f}**. Same rows and feature inputs in every attempt."))
    return comparison

# %% [markdown]
# #### 4.1.1 Attempt 1 - Logistic regression baseline (Labs 2-3)
#
# LogisticRegression.fit() learns an additive score and sigmoid probability.
# Regularization controls large coefficients among correlated form, qualifying,
# and standings features. Its limitation is the lack of automatically learned
# nonlinear interactions. Six small candidate combinations compare C (inverse
# regularization) and class weighting on validation AP. Balanced weights alter
# the learning objective and may hurt calibration; Brier is reported as well.

# %%
logistic_candidates = []
logistic_configs = [{"C": c, "class_weight": weight}
                    for c in [0.1, 1.0, 10.0] for weight in [None, "balanced"]]
for config in logistic_configs:
    estimator = LogisticRegression(**config, max_iter=2000, random_state=SEED)
    estimator.fit(Z_train, y_train)
    probabilities = estimator.predict_proba(Z_val)[:, 1]
    score = float(average_precision_score(y_val, probabilities))
    tuning_records.append({"model": "Logistic regression", "config": json.dumps(config),
                           "validation_AP": score, "epochs": np.nan, "trained_epochs": np.nan})
    logistic_candidates.append((score, estimator, config, probabilities))
chosen_logistic = max(logistic_candidates, key=lambda result: result[0])
models["Logistic regression"] = chosen_logistic[1]
best_configs["Logistic regression"] = chosen_logistic[2]
validation_probabilities["Logistic regression"] = chosen_logistic[3]
display(pd.DataFrame([r for r in tuning_records if r["model"] == "Logistic regression"]))
attempt_comparison = show_attempt_comparison("Logistic regression")

# %% [markdown]
# #### 4.1.2 Attempt 2 - Random forest versus the baseline (Lab 4)
#
# RandomForestClassifier.fit() averages trees trained on bootstrap samples with
# random feature subsets. It can learn nonlinear starting-grid/form interactions.
# Depth and minimum leaf size restrain overfitting, though historical rule changes
# remain a limitation. Four configurations receive exactly the same Z_train and
# Z_val as logistic regression. Trees do not need scaling, but receive it so the
# comparison and the saved preprocessing remain consistent.

# %%
forest_candidates = []
forest_configs = [{"max_depth": depth, "min_samples_leaf": leaf}
                  for depth in [8, None] for leaf in [4, 12]]
for config in forest_configs:
    estimator = RandomForestClassifier(**config, n_estimators=200,
                                       max_features="sqrt", n_jobs=2, random_state=SEED)
    estimator.fit(Z_train, y_train)
    probabilities = estimator.predict_proba(Z_val)[:, 1]
    score = float(average_precision_score(y_val, probabilities))
    tuning_records.append({"model": "Random forest", "config": json.dumps(config),
                           "validation_AP": score, "epochs": np.nan, "trained_epochs": np.nan})
    forest_candidates.append((score, estimator, config, probabilities))
chosen_forest = max(forest_candidates, key=lambda result: result[0])
models["Random forest"] = chosen_forest[1]
best_configs["Random forest"] = chosen_forest[2]
validation_probabilities["Random forest"] = chosen_forest[3]
display(pd.DataFrame([r for r in tuning_records if r["model"] == "Random forest"]))
attempt_comparison = show_attempt_comparison("Random forest")

# %% [markdown]
# ### 4.2 Attempt 3 - Shallow Keras network versus the baseline (Lab 3)
#
# Sequential contains one hidden Dense ReLU layer and one sigmoid output for
# binary classification. compile() selects binary cross-entropy and Adam;
# fit() manages batches, gradients, validation, and History. Small L2 kernel
# penalties discourage large weights. Three candidates test hidden width and L2
# strength. This network learns a continuous nonlinear representation, but can
# overfit the limited historical sample and temporal drift. Its flexibility must
# earn a measured improvement over the baseline.
#
# The lab monitors validation accuracy. Here a short callback computes **exact
# sklearn AP** after every epoch. It writes val_ap into logs BEFORE EarlyStopping
# reads it. EarlyStopping(mode='max', restore_best_weights=True) restores the
# highest-AP epoch; no manual checkpoint management or batch-training loop is used.
# Explicit validation_data uses 2019-2021; no random validation_split is introduced.
# The callback is the only extra Keras class needed for this task's selection rule.

# %%
class ExactValidationAP(keras.callbacks.Callback):
    """Publish the task's exact validation metric before the stopping callback."""
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model(self.inputs, training=False).numpy().ravel()
        logs["val_ap"] = float(average_precision_score(self.labels, probabilities))


neural_candidates, neural_history_records = [], []
neural_configs = [{"hidden_dim": 16, "l2": 0.001},
                  {"hidden_dim": 32, "l2": 0.001},
                  {"hidden_dim": 32, "l2": 0.01}]
for config in neural_configs:
    keras.utils.set_random_seed(SEED)
    network = keras.Sequential([
        layers.Input(shape=(Z_train.shape[1],)),
        layers.Dense(config["hidden_dim"], activation="relu",
                     kernel_regularizer=keras.regularizers.l2(config["l2"])),
        layers.Dense(1, activation="sigmoid"),
    ])
    network.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003),
                    loss="binary_crossentropy", metrics=["accuracy"])
    exact_ap = ExactValidationAP(Z_val, y_val)
    early_stop = EarlyStopping(monitor="val_ap", mode="max", patience=20,
                               min_delta=0.0, restore_best_weights=True)
    history = network.fit(Z_train.astype(np.float32), y_train.astype(np.float32),
                          validation_data=(Z_val.astype(np.float32), y_val.astype(np.float32)),
                          epochs=180, batch_size=256, shuffle=True, verbose=0,
                          callbacks=[exact_ap, early_stop])
    probabilities = network(Z_val.astype(np.float32), training=False).numpy().ravel()
    score = float(average_precision_score(y_val, probabilities))
    best_epoch = int(np.argmax(history.history["val_ap"]) + 1)
    assert np.isclose(score, max(history.history["val_ap"])), "Best AP weights were not restored."
    candidate_config = {**config, "learning_rate": 0.003, "batch_size": 256,
                        "best_epoch": best_epoch}
    history_table = pd.DataFrame({"epoch": np.arange(1, len(history.history["loss"]) + 1),
                                  "training_loss": history.history["loss"],
                                  "validation_loss": history.history["val_loss"],
                                  "training_accuracy": history.history["accuracy"],
                                  "validation_accuracy": history.history["val_accuracy"],
                                  "validation_AP": history.history["val_ap"]})
    tuning_records.append({"model": "Shallow neural network", "config": json.dumps(candidate_config),
                           "validation_AP": score, "epochs": best_epoch,
                           "trained_epochs": len(history_table)})
    neural_history_records.append(history_table.assign(hidden_dim=config["hidden_dim"],
                                                        l2=config["l2"], selected_epoch=best_epoch))
    neural_candidates.append((score, network, candidate_config, history_table, probabilities))
    print(f"Neural candidate {config}: validation AP={score:.4f}, "
          f"restored epoch={best_epoch}, trained epochs={len(history_table)}")

chosen_nn = max(neural_candidates, key=lambda result: result[0])
nn_model = chosen_nn[1]
nn_preprocessor = preprocessing
models["Shallow neural network"] = nn_model
best_configs["Shallow neural network"] = chosen_nn[2]
nn_history = chosen_nn[3]
validation_probabilities["Shallow neural network"] = chosen_nn[4]
pd.concat(neural_history_records, ignore_index=True).to_csv(TABLES / "neural_training_histories.csv", index=False)
display(pd.DataFrame([r for r in tuning_records if r["model"] == "Shallow neural network"]))
attempt_comparison = show_attempt_comparison("Shallow neural network")
attempt_comparison.to_csv(TABLES / "validation_attempt_comparison.csv", index=False)
tuning_table = pd.DataFrame(tuning_records)
tuning_table.to_csv(TABLES / "tuning.csv", index=False)
assert len(tuning_table) == 13
display(tuning_table)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(nn_history.epoch, nn_history.training_loss, label="Training")
axes[0].plot(nn_history.epoch, nn_history.validation_loss, label="Validation")
axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy + L2 penalty", title="Keras History: loss")
axes[0].legend()
axes[1].plot(nn_history.epoch, nn_history.validation_AP, label="Exact validation AP")
axes[1].axvline(chosen_nn[2]["best_epoch"], color="tab:red", ls="--", label="Restored checkpoint")
axes[1].set(xlabel="Epoch", ylabel="Validation AP", title="EarlyStopping on validation AP")
axes[1].legend()
savefig("neural_learning_curve.png", "Lab-style Keras History plots: training/validation loss and exact validation AP. EarlyStopping restores the maximum validation-AP weights.")
model_figures.append("neural_learning_curve.png")

# %% [markdown]
# ### 4.3 Controlled experiments before opening the holdout
# #### 4.3.1 Every feature group ablated
#
# Remove one declared group at a time and retrain logistic regression with the
# selected configuration fixed. Same training/validation rows; preprocessing is
# refitted on training only for the retained columns. Full-feature logistic is
# the reference. AP is threshold free; the F1 column uses each candidate's
# validation-selected threshold. A negative delta means removal hurts. Positive
# deltas are retained honestly: correlated inputs may substitute for one another,
# and a hypothesis-supported feature need not improve every fitted model.

# %%
full_lr_ap = float(average_precision_score(y_val, validation_probabilities["Logistic regression"]))
full_lr_threshold = choose_threshold(y_val, validation_probabilities["Logistic regression"])
ablation_records = [{"group_removed": "None (all features)", "feature_count": len(FEATURES),
                     "AP": full_lr_ap, "delta_AP": 0.0,
                     "F1": float(f1_score(y_val, validation_probabilities["Logistic regression"] >= full_lr_threshold)),
                     "threshold": full_lr_threshold}]
for group, group_features in FEATURE_GROUPS.items():
    kept = [f for f in FEATURES if f not in group_features]
    assert 0 < len(kept) < len(FEATURES)
    ablated_preprocessing = fit_preprocessing(X_train[kept])
    ablated = LogisticRegression(**best_configs["Logistic regression"], max_iter=2000, random_state=SEED)
    ablated.fit(transform_frame(X_train[kept], ablated_preprocessing), y_train)
    probabilities = ablated.predict_proba(transform_frame(X_val[kept], ablated_preprocessing))[:, 1]
    threshold = choose_threshold(y_val, probabilities)
    score = float(average_precision_score(y_val, probabilities))
    ablation_records.append({"group_removed": group, "feature_count": len(kept),
                             "AP": score, "delta_AP": score - full_lr_ap,
                             "F1": float(f1_score(y_val, probabilities >= threshold)), "threshold": threshold})
ablation_table = pd.DataFrame(ablation_records)
assert set(ablation_table.group_removed.iloc[1:]) == set(FEATURE_GROUPS)
ablation_table.to_csv(TABLES / "ablation.csv", index=False)
display(ablation_table)
fig, ax = plt.subplots(figsize=(9, 4.8))
plot_ablation = ablation_table.iloc[1:].sort_values("delta_AP")
ax.barh(plot_ablation.group_removed, plot_ablation.delta_AP,
        color=np.where(plot_ablation.delta_AP < 0, "#d55e00", "#0072b2"))
ax.axvline(0, color="black", lw=1)
ax.set(xlabel="Validation AP after removal - full-feature AP",
       title="Every declared feature group: fixed logistic configuration")
savefig("feature_group_ablation.png", "Validation-only ablation of every feature group, with fixed logistic hyperparameters and unchanged rows. Improvements after removal are also shown.")
model_figures.append("feature_group_ablation.png")

# %% [markdown]
# #### 4.3.2 Two preprocessing-order comparisons using explicit operations
#
# **A:** median filling then StandardScaler versus StandardScaler fitted to
# observed values then median filling in scaled units. StandardScaler ignores
# NaNs when fitting. Filling before fitting its variance can therefore change
# values and regularized model behavior. Flags follow the same schema and are
# standardized in both variants.
#
# **B:** training 1st/99th-percentile clipping then filling/scaling versus filling
# and scaling then training 1st/99th-percentile clipping. Percentile clipping
# (winsorization) is an additional technique beyond the labs: it bounds extreme
# continuous inputs instead of deleting rows. Only continuous columns are
# clipped. Binary original columns and appended missing indicators retain both
# values, including the rare grid_unassigned flag. All medians, scales and bounds
# come exclusively from training; validation rows and logistic settings stay fixed.
#
# These are diagnostic experiments. The primary three attempts retain the
# predeclared impute-then-scale preprocessing. A uses a feature actually missing
# in training; B uses a continuous feature with a long range. The table measures
# changed values, range and dispersion as well as validation performance.

# %%
binary_features = [f for f in FEATURES if set(X_train[f].dropna().unique()).issubset({0.0, 1.0})]
continuous_features = [f for f in FEATURES if f not in binary_features]
continuous_indices = [transformed_names.index(f) for f in continuous_features]
protected_indices = [transformed_names.index(f) for f in binary_features]
protected_indices += list(range(len(FEATURES), len(transformed_names)))
assert "grid_unassigned" in binary_features

# A0 is exactly the already fitted primary preprocessing.
transformed_orders = {"A0: impute → scale (baseline)": Z_train}
validation_orders = {"A0: impute → scale (baseline)": Z_val}

# A1: fit the scale to observed values first, then fill with training scaled medians.
flagged_train = append_missing_flags(X_train, X_train, preprocessing["indicator_features"])
flagged_val = append_missing_flags(X_val, X_val, preprocessing["indicator_features"])
scale_first = StandardScaler()
scaled_train = pd.DataFrame(scale_first.fit_transform(flagged_train), columns=transformed_names, index=X_train.index)
scaled_val = pd.DataFrame(scale_first.transform(flagged_val), columns=transformed_names, index=X_val.index)
scaled_medians = scaled_train.median()
transformed_orders["A1: scale → impute"] = scaled_train.fillna(scaled_medians).to_numpy()
validation_orders["A1: scale → impute"] = scaled_val.fillna(scaled_medians).to_numpy()

# B0: bounds on observed raw continuous values; missing flags still come from raw X.
raw_lower = X_train[continuous_features].quantile(0.01)
raw_upper = X_train[continuous_features].quantile(0.99)
clipped_train, clipped_val = X_train.copy(), X_val.copy()
clipped_train[continuous_features] = X_train[continuous_features].clip(raw_lower, raw_upper, axis=1)
clipped_val[continuous_features] = X_val[continuous_features].clip(raw_lower, raw_upper, axis=1)
clip_medians = clipped_train.median()
clip_completed_train = append_missing_flags(clipped_train.fillna(clip_medians), X_train, preprocessing["indicator_features"])
clip_completed_val = append_missing_flags(clipped_val.fillna(clip_medians), X_val, preprocessing["indicator_features"])
clip_first_scaler = StandardScaler()
transformed_orders["B0: clip → impute → scale"] = clip_first_scaler.fit_transform(clip_completed_train)
validation_orders["B0: clip → impute → scale"] = clip_first_scaler.transform(clip_completed_val)

# B1: impute and scale first; bounds on completed/scaled training continuous values.
scaled_lower = np.quantile(Z_train[:, continuous_indices], 0.01, axis=0)
scaled_upper = np.quantile(Z_train[:, continuous_indices], 0.99, axis=0)
clip_last_train, clip_last_val = Z_train.copy(), Z_val.copy()
clip_last_train[:, continuous_indices] = np.clip(clip_last_train[:, continuous_indices], scaled_lower, scaled_upper)
clip_last_val[:, continuous_indices] = np.clip(clip_last_val[:, continuous_indices], scaled_lower, scaled_upper)
transformed_orders["B1: impute → scale → clip"] = clip_last_train
validation_orders["B1: impute → scale → clip"] = clip_last_val

preprocessing_records, preprocessing_effect_records = [], []
for order_name, inputs in transformed_orders.items():
    assert np.isfinite(inputs).all() and np.isfinite(validation_orders[order_name]).all()
    # Clipping must preserve every original binary column and missingness flag.
    if order_name.startswith("B"):
        np.testing.assert_allclose(inputs[:, protected_indices], Z_train[:, protected_indices], atol=1e-12)
        np.testing.assert_allclose(validation_orders[order_name][:, protected_indices], Z_val[:, protected_indices], atol=1e-12)
    estimator = LogisticRegression(**best_configs["Logistic regression"], max_iter=2000, random_state=SEED)
    estimator.fit(inputs, y_train)
    probabilities = estimator.predict_proba(validation_orders[order_name])[:, 1]
    threshold = choose_threshold(y_val, probabilities)
    preprocessing_records.append({"order": order_name, **metric_row(y_val, probabilities, threshold)})
    for j, feature in enumerate(transformed_names):
        values = inputs[:, j]
        preprocessing_effect_records.append({"order": order_name, "feature": feature,
            "training_missing_rows": int(X_train[feature].isna().sum()) if feature in FEATURES else 0,
            "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "standard_deviation": float(values.std()),
            "changed_vs_A0": int((~np.isclose(values, Z_train[:, j], atol=1e-10, rtol=1e-10)).sum()),
            "maximum_absolute_change_vs_A0": float(np.max(np.abs(values - Z_train[:, j]))),
            "binary_or_indicator_protected": j in protected_indices})
preprocessing_table = pd.DataFrame(preprocessing_records)
preprocessing_table.to_csv(TABLES / "preprocessing.csv", index=False)
preprocessing_value_effects = pd.DataFrame(preprocessing_effect_records)
preprocessing_value_effects.to_csv(TABLES / "preprocessing_value_effects.csv", index=False)
display(preprocessing_table)

missing_continuous = [f for f in continuous_features if X_train[f].isna().any()]
assert missing_continuous, "Experiment A needs an actually missing continuous feature."
distribution_feature = "qualifying_gap_pct" if "qualifying_gap_pct" in missing_continuous else missing_continuous[0]
distribution_feature_b = "driver_prior_starts" if "driver_prior_starts" in continuous_features else continuous_features[0]
display(preprocessing_value_effects.loc[preprocessing_value_effects.feature.isin([distribution_feature, distribution_feature_b])])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for order_name in transformed_orders:
    is_a = order_name.startswith("A")
    feature = distribution_feature if is_a else distribution_feature_b
    axes[0 if is_a else 1].hist(transformed_orders[order_name][:, transformed_names.index(feature)],
        bins=45, density=True, histtype="step", lw=2, label=order_name)
axes[0].set(title=f"A: {int(X_train[distribution_feature].isna().sum())} missing training values",
            xlabel=f"Transformed {distribution_feature}", ylabel="Training density")
axes[1].set(title="B: continuous-feature percentile clipping",
            xlabel=f"Transformed {distribution_feature_b}", ylabel="Training density")
for ax in axes:
    ax.legend(fontsize=8)
savefig("preprocessing_order_distributions.png", "Training distributions for A (a feature with actual missing values) and B (a continuous range). All binary/missingness flags are protected from clipping; all statistics are training-only.")
model_figures.append("preprocessing_order_distributions.png")
order_a_value_change = float(np.max(np.abs(transformed_orders["A1: scale → impute"][:, transformed_names.index(distribution_feature)] - Z_train[:, transformed_names.index(distribution_feature)])))
assert order_a_value_change > 1e-6
display(Markdown(f"**Observed transformation effect:** {distribution_feature} has "
                 f"{int(X_train[distribution_feature].isna().sum()):,} training missing values. "
                 f"Its maximum absolute A1-vs-A0 transformed change is **{order_a_value_change:.4f}**. "
                 f"All {len(protected_indices)} binary/indicator columns match the unclipped baseline "
                 "in both clipping variants. Range/value evidence is exported for every feature and order."))

# %% [markdown]
# #### 4.3.3 Historical qualifying-coverage sensitivity
#
# Refit the same logistic configuration on 2003-2018, rather than 1994-2018,
# while retaining the same 2019-2021 validation rows and feature definitions.
# Its preprocessing is fitted only on its shorter training window. Earlier known
# events may still supply historical lagged features. Changing training era also
# changes sample size, entrants, cars, and rules; this does not isolate a causal
# effect of missing qualifying. The primary training window stays predeclared.

# %%
recent_train_mask = train_rows["year"].ge(2003).to_numpy()
recent_preprocessing = fit_preprocessing(X_train.loc[recent_train_mask])
recent_lr = LogisticRegression(**best_configs["Logistic regression"], max_iter=2000, random_state=SEED)
recent_lr.fit(transform_frame(X_train.loc[recent_train_mask], recent_preprocessing), y_train[recent_train_mask])
recent_probabilities = recent_lr.predict_proba(transform_frame(X_val, recent_preprocessing))[:, 1]
recent_threshold = choose_threshold(y_val, recent_probabilities)
coverage_sensitivity = pd.DataFrame([
    {"training_window": "1994-2018", "training_rows": len(X_train),
     "qualifying_missing_rate": float(X_train.qualifying_position.isna().mean()),
     **metric_row(y_val, validation_probabilities["Logistic regression"], full_lr_threshold)},
    {"training_window": "2003-2018", "training_rows": int(recent_train_mask.sum()),
     "qualifying_missing_rate": float(X_train.loc[recent_train_mask].qualifying_position.isna().mean()),
     **metric_row(y_val, recent_probabilities, recent_threshold)},
])
coverage_sensitivity_table = coverage_sensitivity
coverage_sensitivity.to_csv(TABLES / "training_window_sensitivity.csv", index=False)
coverage_ap_delta = float(coverage_sensitivity.iloc[1].AP - coverage_sensitivity.iloc[0].AP)
display(coverage_sensitivity)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(coverage_sensitivity.training_window, coverage_sensitivity.AP, color=["#0072b2", "#009e73"])
axes[0].set(ylabel="Validation AP", title="Fixed logistic configuration: training period", ylim=(0, 1))
axes[1].bar(coverage_sensitivity.training_window, coverage_sensitivity.qualifying_missing_rate, color=["#0072b2", "#009e73"])
axes[1].set(ylabel="Training qualifying-position missing fraction", title="Coverage and sample-size tradeoff")
savefig("training_window_sensitivity.png", "Validation-only 1994 versus 2003 training-window sensitivity, using fixed logistic hyperparameters and validation rows. Era, sample size and qualifying coverage all change together.")
model_figures.append("training_window_sensitivity.png")

# %% [markdown]
# ### 4.4 Freeze choices, then score the future-season holdout
#
# Choose each threshold using validation F1 and the final model using validation
# AP. Save those decisions BEFORE computing holdout predictions. Models retain
# their training-only weights so these thresholds refer to the same models.
# Refitting on train plus validation would require a new calibration/checkpoint
# design. The holdout is scored once below; no later code tunes from it.

# %%
thresholds = {name: choose_threshold(y_val, p) for name, p in validation_probabilities.items()}
final_model_name = max(validation_probabilities,
    key=lambda name: average_precision_score(y_val, validation_probabilities[name]))
final_threshold = thresholds[final_model_name]
selection_frozen = {"selected_model": final_model_name, "thresholds": thresholds,
                    "best_configs": best_configs, "features": FEATURES,
                    "training_rows": len(X_train), "validation_rows": len(X_val),
                    "selection_rule": "Validation AP; thresholds maximize validation F1; primary preprocessing unchanged",
                    "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
(OUT / "selection_frozen.json").write_text(json.dumps(selection_frozen, indent=2), encoding="utf-8")


def probability_for_model(name, frame):
    inputs = transform_frame(frame, preprocessing)
    if name == "Shallow neural network":
        return models[name](inputs.astype(np.float32), training=False).numpy().ravel()
    return models[name].predict_proba(inputs)[:, 1]


metrics_records, test_probabilities = [], {}
for name in models:
    for split_name, frame, labels in [("train", X_train, y_train),
                                      ("validation", X_val, y_val), ("test", X_test, y_test)]:
        probabilities = validation_probabilities[name] if split_name == "validation" else probability_for_model(name, frame)
        if split_name == "test":
            test_probabilities[name] = probabilities
        metrics_records.append({"model": name, "split": split_name,
                                **metric_row(labels, probabilities, thresholds[name])})
metrics_table = pd.DataFrame(metrics_records)
assert len(metrics_table) == 9 and metrics_table.notna().all().all()
metrics_table.to_csv(TABLES / "metrics.csv", index=False)
display(metrics_table.round(4))
display(Markdown(f"**Validation-selected model:** {final_model_name}. Frozen threshold "
                 f"**{final_threshold:.4f}**. Scores below are final holdout evaluation; "
                 "the selected model is retained even if another model scores higher on test."))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for name, probabilities in test_probabilities.items():
    false_positive, true_positive, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[0].plot(false_positive, true_positive, label=name)
    axes[1].plot(recall, precision, label=name)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1].axhline(y_test.mean(), color="black", ls="--", label="Test prevalence")
axes[0].set(title="Holdout ROC curves", xlabel="False positive rate", ylabel="True positive rate")
axes[1].set(title="Holdout precision-recall curves", xlabel="Recall", ylabel="Precision")
for ax in axes:
    ax.legend(fontsize=8)
savefig("holdout_curves.png", "2022-2024 ROC and precision-recall curves of all three frozen models; model choice uses validation AP only.")
model_figures.append("holdout_curves.png")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, test_probabilities[final_model_name] >= final_threshold,
    display_labels=["No podium", "Podium"], ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"{final_model_name}: frozen-threshold holdout")
savefig("holdout_confusion_matrix.png", "Holdout confusion matrix for the validation-selected model and its frozen validation-F1 threshold.")
model_figures.append("holdout_confusion_matrix.png")

# %% [markdown]
# ### 4.5 Explain the selected model globally and locally (Lab 4)
#
# Permutation importance uses sklearn's library implementation: shuffle one
# original validation feature and measure the loss in AP, over eight repeats.
# SHAP uses a seeded training background and a validation sample stratified by
# season in proportion to each season's row count, without consulting labels.
# The local case is appended only for explanation, not to global averages/plots.
# Sampling membership and season counts are exported. TreeExplainer
# explains forest podium probability; LinearExplainer explains logistic log-odds;
# if the network wins, KernelExplainer approximates its podium probability.
# Only the branch for the actual selected model is run. The input representation
# always matches the fitted estimator: imputed, indicator-augmented and scaled.
#
# The local case is selected without its outcome: validation probability nearest
# 0.5. SHAP waterfall and LIME explain that SAME row. SHAP units are stated and
# its additive reconstruction is checked. LIME's local surrogate R-squared is
# reported. SHAP displays standardized inputs. LIME displays imputed values in
# original racing units and applies the frozen scaler inside its probability
# wrapper. Binary inputs and missing flags are categorical 0/1 values for LIME,
# so their individual states remain valid when perturbed.
#
# **Interpretation:** beeswarm sign indicates an increase/decrease in the model's
# output relative to its background; red means a larger standardized input, blue
# a smaller one. Waterfall adds local contributions to the background. Neither
# SHAP, permutation, nor LIME establishes causal effects. Correlated qualifying,
# grid, form and standings share information. Independent perturbations can
# create unrealistic feature combinations (including a grid/flag mismatch), and
# importance may understate redundant feature groups. LIME fidelity measures
# local approximation quality, not predictive correctness or causality.

# %%
import shap
from lime.lime_tabular import LimeTabularExplainer

explained_model_name = final_model_name
selected_estimator = models[final_model_name]
selected_preprocessor = preprocessing
selected_probability = lambda frame: probability_for_model(final_model_name, frame)


def selected_ap_scorer(estimator, original_frame, labels):
    inputs = transform_frame(original_frame, preprocessing)
    if final_model_name == "Shallow neural network":
        positive = estimator(inputs.astype(np.float32), training=False).numpy().ravel()
    else:
        positive = estimator.predict_proba(inputs)[:, 1]
    return average_precision_score(labels, positive)


importance = permutation_importance(selected_estimator, X_val, y_val,
    scoring=selected_ap_scorer, n_repeats=8, random_state=SEED, n_jobs=1)
permutation_table = pd.DataFrame({"feature": FEATURES,
    "AP_drop_mean": importance.importances_mean,
    "AP_drop_std": importance.importances.std(axis=1, ddof=1)}).sort_values("AP_drop_mean", ascending=False)
permutation_table.to_csv(TABLES / "permutation_importance.csv", index=False)
display(permutation_table)
fig, ax = plt.subplots(figsize=(9, 6))
ordered = permutation_table.iloc[::-1]
ax.barh(ordered.feature, ordered.AP_drop_mean, xerr=ordered.AP_drop_std, color="#0072b2", alpha=0.85)
ax.axvline(0, color="black", lw=1)
ax.set(xlabel="Validation AP decrease (mean ± SD of 8 shuffles)",
       title=f"Permutation importance: {final_model_name}")
savefig("permutation_importance.png", "Global validation permutation importance of original inputs, scored through the frozen preprocessing/model. Bars are eight seeded AP-loss shuffles; error bars are sample SD.")
model_figures.append("permutation_importance.png")

explanation_train, explanation_val = Z_train, Z_val
xai_rng = np.random.default_rng(SEED)
background_indices = xai_rng.choice(len(X_train), size=min(80, len(X_train)), replace=False)
sample_budget = min(60 if final_model_name == "Shallow neural network" else 120, len(X_val))
validation_years = val_rows["year"].to_numpy(dtype=int)
year_counts = pd.Series(validation_years).value_counts().sort_index()
ideal_year_counts = sample_budget * year_counts / year_counts.sum()
sample_year_counts = np.floor(ideal_year_counts).astype(int)
remainder = sample_budget - int(sample_year_counts.sum())
for sample_year in (ideal_year_counts - sample_year_counts).sort_values(
        ascending=False, kind="stable").index[:remainder]:
    sample_year_counts.loc[sample_year] += 1
global_sample_indices = np.concatenate([
    xai_rng.choice(np.flatnonzero(validation_years == sample_year), size=int(count), replace=False)
    for sample_year, count in sample_year_counts.items()
])
assert len(global_sample_indices) == sample_budget
local_index = int(np.argmin(np.abs(validation_probabilities[final_model_name] - 0.5)))
sample_indices = np.unique(np.r_[global_sample_indices, local_index])
global_sample_mask = np.isin(sample_indices, global_sample_indices)
background = explanation_train[background_indices]
explanation_sample = explanation_val[sample_indices]
shap_sample = val_rows.iloc[sample_indices][["raceId", "driverId", "year", "date"]].copy()
shap_sample.insert(0, "validation_position", sample_indices)
shap_sample["included_global"] = global_sample_mask
shap_sample["included_local"] = sample_indices == local_index
shap_sample.to_csv(TABLES / "shap_sample.csv", index=False)
shap_sampling_table = pd.DataFrame({"year": year_counts.index,
    "validation_rows": year_counts.values, "global_sample_rows": sample_year_counts.values})
shap_sampling_table["validation_fraction"] = shap_sampling_table.validation_rows / len(X_val)
shap_sampling_table["global_sample_fraction"] = shap_sampling_table.global_sample_rows / sample_budget
shap_sampling_table.to_csv(TABLES / "shap_sample_by_year.csv", index=False)
display(shap_sampling_table)
shap_sampling_description = ", ".join(
    f"{int(row.year)}: {int(row.global_sample_rows)}" for row in shap_sampling_table.itertuples())


def transformed_probability(array):
    """LIME/SHAP already supply the fitted estimator's transformed inputs."""
    if final_model_name == "Shallow neural network":
        positive = selected_estimator(np.asarray(array, dtype=np.float32), training=False).numpy().ravel()
        return np.column_stack([1 - positive, positive])
    return selected_estimator.predict_proba(np.asarray(array, dtype=float))


if final_model_name == "Random forest":
    shap_explainer = shap.TreeExplainer(selected_estimator, data=background,
        model_output="probability", feature_perturbation="interventional")
    values = shap_explainer.shap_values(explanation_sample, check_additivity=False)
    # Older SHAP returns one array per class; newer releases append the class axis.
    shap_values = np.asarray(values[1] if isinstance(values, list) else values[:, :, 1])
    shap_base = float(np.asarray(shap_explainer.expected_value)[1])
    shap_units = "podium probability"
elif final_model_name == "Logistic regression":
    shap_explainer = shap.LinearExplainer(selected_estimator, background)
    shap_values = np.asarray(shap_explainer.shap_values(explanation_sample))
    shap_base = float(np.asarray(shap_explainer.expected_value).ravel()[0])
    shap_units = "log-odds"
else:
    shap_explainer = shap.KernelExplainer(lambda inputs: transformed_probability(inputs)[:, 1], background)
    shap_values = np.asarray(shap_explainer.shap_values(explanation_sample,
        nsamples=250, l1_reg="num_features(12)", silent=True))
    shap_base = float(shap_explainer.expected_value)
    shap_units = "podium probability (Kernel SHAP approximation)"

global_shap_values = shap_values[global_sample_mask]
global_explanation_sample = explanation_sample[global_sample_mask]
shap_table = pd.DataFrame({"feature": transformed_names,
    "mean_abs_SHAP": np.mean(np.abs(global_shap_values), axis=0)}).sort_values("mean_abs_SHAP", ascending=False)
shap_table.to_csv(TABLES / "shap_importance.csv", index=False)
display(shap_table.head(12))
plt.figure(figsize=(10, 6))
shap.summary_plot(global_shap_values, global_explanation_sample, feature_names=transformed_names, show=False, max_display=15, rng=np.random.default_rng(SEED))
plt.title(f"{final_model_name}: global SHAP ({shap_units})", fontsize=11)
savefig("shap_global.png", f"Global SHAP for the validation-selected {final_model_name}; season-proportional seeded validation sample ({shap_sampling_description}). The separately selected local case does not enter global summaries. Attribution units: {shap_units}; background is training-only.")
model_figures.append("shap_global.png")

local_sample_index = int(np.flatnonzero(sample_indices == local_index)[0])
local_values = shap_values[local_sample_index]
local_explanation_object = shap.Explanation(values=local_values, base_values=shap_base,
    data=explanation_val[local_index], feature_names=transformed_names)
shap.plots.waterfall(local_explanation_object, max_display=12, show=False)
plt.title(f"Local prediction: {shap_units}", fontsize=11)
savefig("shap_local_waterfall.png", "Local SHAP waterfall for the validation probability nearest 0.5, selected without consulting its true label. The same case is explained with LIME.")
model_figures.append("shap_local_waterfall.png")
local_shap_table = pd.DataFrame({"feature": transformed_names,
    "value_transformed": explanation_val[local_index], "SHAP": local_values})
local_shap_table = local_shap_table.sort_values("SHAP", key=lambda series: series.abs(), ascending=False)
local_shap_table.to_csv(TABLES / "local_shap.csv", index=False)

lime_train = append_missing_flags(X_train.fillna(preprocessing["medians"]), X_train,
                                  preprocessing["indicator_features"]).to_numpy()
lime_local = append_missing_flags(X_val.fillna(preprocessing["medians"]), X_val,
                                  preprocessing["indicator_features"]).iloc[local_index].to_numpy()


def lime_probability(imputed_inputs):
    """Scale LIME's imputed, original-unit features exactly once before prediction."""
    completed = pd.DataFrame(imputed_inputs, columns=transformed_names)
    scaled_inputs = preprocessing["scaler"].transform(completed)
    return transformed_probability(scaled_inputs)


np.testing.assert_allclose(lime_probability(lime_local.reshape(1, -1))[0, 1],
                            validation_probabilities[final_model_name][local_index], atol=1e-6)
lime_explainer = LimeTabularExplainer(lime_train, feature_names=transformed_names,
    class_names=["No podium", "Podium"], mode="classification", discretize_continuous=True,
    categorical_features=protected_indices, random_state=SEED)
lime_explanation = lime_explainer.explain_instance(lime_local, lime_probability,
    labels=(1,), num_features=10, num_samples=3000)
lime_table = pd.DataFrame(lime_explanation.as_list(label=1), columns=["condition", "local_weight"])
lime_table.to_csv(TABLES / "lime_local.csv", index=False)
lime_explanation.save_to_file(str(OUT / "lime_local_explanation.html"))
fig = lime_explanation.as_pyplot_figure(label=1)
plt.title(f"LIME local surrogate: {final_model_name} (R²={lime_explanation.score:.3f})")
savefig("lime_local.png", "LIME on the same validation case as local SHAP. Conditions are imputed original-unit values with binary flags; a wrapper applies frozen scaling. The title reports local surrogate R-squared.")
model_figures.append("lime_local.png")
local_metadata = val_rows.iloc[local_index][["raceId", "driverId", "year", "round"]].to_dict()
local_probability = float(validation_probabilities[final_model_name][local_index])
local_actual = int(y_val[local_index])
shap_reconstruction = float(shap_base + local_values.sum())
if final_model_name == "Logistic regression":
    shap_reconstruction = float(1 / (1 + np.exp(-shap_reconstruction)))
shap_reconstruction_error = abs(shap_reconstruction - local_probability)
assert shap_reconstruction_error < (0.03 if final_model_name == "Shallow neural network" else 0.005)
display(Markdown(f"Local case: **{local_metadata}**; predicted podium probability "
    f"**{local_probability:.3f}**, observed label **{local_actual}**, frozen threshold "
    f"**{final_threshold:.3f}**. SHAP reconstruction **{shap_reconstruction:.3f}** "
    f"(absolute error {shap_reconstruction_error:.6f}); LIME surrogate R² "
    f"**{lime_explanation.score:.3f}**. Explanations describe model association, not causality."))
(OUT / "explanation_checks.json").write_text(json.dumps({"model": final_model_name,
    "raceId": int(local_metadata["raceId"]), "driverId": int(local_metadata["driverId"]),
    "local_probability": local_probability, "shap_reconstruction": shap_reconstruction,
    "shap_absolute_error": shap_reconstruction_error, "shap_units": shap_units,
    "lime_R2": float(lime_explanation.score), "background_split": "train",
    "explanation_split": "validation", "seed": SEED,
    "global_sample_method": "Season-proportional stratified random sample without replacement; labels unused",
    "global_sample_rows": int(sample_budget),
    "global_sample_by_year": {str(year): int(count) for year, count in sample_year_counts.items()},
    "local_only_excluded_from_global": True}, indent=2), encoding="utf-8")

# %% [markdown]
# ### 4.6 Inference, missing inputs, and native persistence
#
# predict_podium(record) consumes numeric **engineered pre-race features**, in
# the feature catalog's units. It is not a raw-name/history lookup: rolling
# features and standings must already exclude the requested race. Missing fields
# and nulls use frozen training medians and indicators, and are reported back.
# Omitted/null/nonpositive grid becomes missing and derives grid_unassigned=1;
# positive grid derives 0. Supplied contradictory flags, unknown keys, malformed
# numbers and nonfinite values, including NaN, are rejected. Omitted keys and
# None represent missing external inputs. Extensive imputation reduces reliability.
#
# Demos use a complete historical validation record, then remove qualifying and
# standings fields, then remove grid and its flag. These do not select models.
# Export uses ordinary joblib preprocessing/sklearn artifacts and Keras's native
# .keras serialization. Predictions from EVERY reloaded model are checked
# against the in-memory fitted model; neural inference uses Keras load_model().

# %%
def predict_podium(record):
    """Predict one podium probability from documented engineered pre-race inputs."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary of numeric pre-race features.")
    unknown = sorted(set(record) - set(FEATURES))
    if unknown:
        raise ValueError(f"Unknown feature keys: {unknown}. Allowed keys: {FEATURES}")
    normalized = dict(record)
    raw_grid = normalized.get("grid")
    if raw_grid is None:
        normalized_grid = None
    else:
        try:
            normalized_grid = float(raw_grid)
        except (ValueError, TypeError) as error:
            raise ValueError("grid must be a number or null.") from error
        if not np.isfinite(normalized_grid):
            raise ValueError("grid must be finite or null.")
        if normalized_grid <= 0:
            normalized_grid = None
    derived_flag = int(normalized_grid is None)
    supplied_flag = normalized.get("grid_unassigned")
    if supplied_flag is not None:
        try:
            supplied_flag_number = float(supplied_flag)
        except (TypeError, ValueError) as error:
            raise ValueError("grid_unassigned must be 0 or 1, matching grid.") from error
        if not np.isfinite(supplied_flag_number):
            raise ValueError("grid_unassigned must be finite or null.")
        if supplied_flag_number != derived_flag:
            raise ValueError(f"grid_unassigned={supplied_flag} contradicts grid={raw_grid}; expected {derived_flag}.")
    normalized["grid"], normalized["grid_unassigned"] = normalized_grid, derived_flag
    values, missing = {}, []
    for feature in FEATURES:
        value = normalized.get(feature)
        if value is None:
            values[feature] = np.nan
            missing.append(feature)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{feature} must be a number or null.") from error
            if not np.isfinite(number):
                raise ValueError(f"{feature} must be finite or null.")
            values[feature] = number
    probability = float(selected_probability(pd.DataFrame([values], columns=FEATURES))[0])
    return {"model": final_model_name, "podium_probability": probability,
            "podium_prediction": int(probability >= final_threshold),
            "threshold": float(final_threshold), "imputed_features": missing,
            "note": "Missing inputs use training medians and missingness flags."
                    if missing else "All engineered inputs supplied."}


complete_positions = np.flatnonzero(~X_val.isna().any(axis=1))
assert len(complete_positions), "Complete demo must be an actual fully observed record."
complete_record = {f: float(v) for f, v in X_val.iloc[int(complete_positions[0])].items()}
partial_record = {key: value for key, value in complete_record.items()
                  if "qualifying" not in key and "standing" not in key}
missing_grid_record = {key: value for key, value in complete_record.items()
                       if key not in ["grid", "grid_unassigned"]}
inference_complete = predict_podium(complete_record)
inference_partial = predict_podium(partial_record)
inference_missing_grid = predict_podium(missing_grid_record)
inference_examples = pd.DataFrame([{"example": "complete", **inference_complete},
    {"example": "partially missing", **inference_partial},
    {"example": "missing grid", **inference_missing_grid}])
display(inference_examples)
inference_examples.to_csv(TABLES / "inference_examples.csv", index=False)
(OUT / "inference_demo_inputs.json").write_text(json.dumps({"complete": complete_record,
    "partial": partial_record, "missing_grid": missing_grid_record}, indent=2, allow_nan=False), encoding="utf-8")
zero_grid_result = predict_podium({**complete_record, "grid": 0, "grid_unassigned": 1})
negative_grid_result = predict_podium({**complete_record, "grid": -1, "grid_unassigned": 1})
null_grid_result = predict_podium({**complete_record, "grid": None, "grid_unassigned": None})
for result in [zero_grid_result, negative_grid_result, null_grid_result]:
    np.testing.assert_allclose(result["podium_probability"], inference_missing_grid["podium_probability"])
    assert result["imputed_features"] == ["grid"]
assert inference_missing_grid["imputed_features"] == ["grid"]
assert not predict_podium({f: v for f, v in complete_record.items() if f != "grid_unassigned"})["imputed_features"]
invalid_records = [({**complete_record, "grid": 0, "grid_unassigned": 0}, "contradicts"),
                   ({**complete_record, "unrecognized": 1}, "Unknown"),
                   ({**complete_record, "grid": np.inf}, "finite"),
                   ({**complete_record, "grid": np.nan}, "finite"),
                   ({**complete_record, "grid_unassigned": np.inf}, "finite"),
                   ({**complete_record, "grid_unassigned": np.nan}, "finite"),
                   ({**complete_record, "driver_age": np.inf}, "finite"),
                   ({**complete_record, "driver_age": np.nan}, "finite"),
                   ({**complete_record, "grid": "bad"}, "number")]
for invalid_record, expected_message in invalid_records:
    try:
        predict_podium(invalid_record)
    except (ValueError, TypeError) as error:
        assert expected_message in str(error)
    else:
        raise AssertionError("Invalid inference record was accepted.")
print("Inference checks passed: complete/partial/grid cases and invalid-input rejection.")

# Native library artifacts share one explicit training-fitted preprocessing bundle.
joblib.dump(preprocessing, MODELS_DIR / "preprocessing.joblib")
joblib.dump(models["Logistic regression"], MODELS_DIR / "logistic_regression.joblib")
joblib.dump(models["Random forest"], MODELS_DIR / "random_forest.joblib")
nn_model.save(MODELS_DIR / "shallow_neural_network.keras")
import sklearn
metadata = {"selected_model": final_model_name, "features": FEATURES,
    "threshold": float(final_threshold), "all_thresholds": thresholds,
    "transformed_features": transformed_names,
    "indicator_features": preprocessing["indicator_features"],
    "feature_groups": FEATURE_GROUPS, "training_years": [1994, 2018],
    "validation_years": [2019, 2021], "test_years": [2022, 2024],
    "best_configs": best_configs, "seed": SEED,
    "feature_semantics": "Engineered post-qualifying pre-race inputs; histories exclude current race. Omitted/null/grid<=0 is missing; grid_unassigned derived; contradictory flags fail.",
    "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                 "tensorflow": tf.__version__, "keras": keras.__version__, "shap": shap.__version__},
    "training_medians": {f: float(preprocessing["medians"][f]) for f in FEATURES},
    "model_files": {"Logistic regression": "logistic_regression.joblib",
                    "Random forest": "random_forest.joblib", "Shallow neural network": "shallow_neural_network.keras"}}
(MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

reloaded_preprocessing = joblib.load(MODELS_DIR / "preprocessing.joblib")
reload_inputs = transform_frame(X_val.iloc[:25], reloaded_preprocessing)
reload_checks = []
for name, filename in [("Logistic regression", "logistic_regression.joblib"),
                       ("Random forest", "random_forest.joblib")]:
    reloaded = joblib.load(MODELS_DIR / filename)
    probabilities = reloaded.predict_proba(reload_inputs)[:, 1]
    reference = validation_probabilities[name][:len(reload_inputs)]
    np.testing.assert_allclose(probabilities, reference, rtol=1e-8, atol=1e-9)
    reload_checks.append({"model": name, "rows": len(reload_inputs),
                          "maximum_absolute_error": float(np.max(np.abs(probabilities - reference))), "passed": True})
reloaded_nn = keras.models.load_model(MODELS_DIR / "shallow_neural_network.keras", compile=False)
reloaded_neural_probability = reloaded_nn(reload_inputs.astype(np.float32), training=False).numpy().ravel()
neural_reference = validation_probabilities["Shallow neural network"][:len(reload_inputs)]
np.testing.assert_allclose(reloaded_neural_probability, neural_reference, rtol=1e-6, atol=1e-7)
reload_checks.append({"model": "Shallow neural network", "rows": len(reload_inputs),
    "maximum_absolute_error": float(np.max(np.abs(reloaded_neural_probability - neural_reference))), "passed": True})
reload_parity_table = pd.DataFrame(reload_checks)
reload_parity_table.to_csv(TABLES / "reload_parity.csv", index=False)
display(reload_parity_table)
print("Native export verification passed for all three models and the shared preprocessing.")

# %% [markdown]
# ### 4.7 Measured findings and practical limits
#
# The summary below is generated from this run's actual results. Holdout
# performance describes 2022-2024 retrospectively, not a future accuracy promise.
# Weather, tyres, race conditions and strategy are absent. Repeated experiments
# can overfit one validation era. Binary probabilities do not enforce exactly
# three predicted podiums per race, and are not externally calibrated. SHAP and
# LIME explain predictive associations, not interventions or causal racing effects.

# %%
winner_val_ap = float(average_precision_score(y_val, validation_probabilities[final_model_name]))
test_metric_dict = metric_row(y_test, test_probabilities[final_model_name], final_threshold)
worst_ablation = ablation_table.iloc[1:].sort_values("delta_AP").iloc[0]
improved_groups = ablation_table.loc[(ablation_table.group_removed != "None (all features)") & ablation_table.delta_AP.gt(0), "group_removed"].tolist()
prep_lookup = preprocessing_table.set_index("order")
order_a_delta = float(prep_lookup.loc["A1: scale → impute", "AP"] - prep_lookup.loc["A0: impute → scale (baseline)", "AP"])
order_b_delta = float(prep_lookup.loc["B1: impute → scale → clip", "AP"] - prep_lookup.loc["B0: clip → impute → scale", "AP"])
top_permutation = ", ".join(permutation_table.head(3).feature)
top_shap = ", ".join(shap_table.head(3).feature)
local_positive = local_shap_table.loc[local_shap_table.SHAP > 0].head(2).feature.tolist()
local_negative = local_shap_table.loc[local_shap_table.SHAP < 0].head(2).feature.tolist()
modeling_summary = {
    "winner": final_model_name, "threshold": float(final_threshold),
    "validation_AP": winner_val_ap, "test_metrics": test_metric_dict, "best_configs": best_configs,
    "ablation_conclusion": f"Removing {worst_ablation['group_removed']} yielded the lowest AP difference ({worst_ablation['delta_AP']:+.4f}) relative to full logistic regression. Groups whose removal improved AP: {', '.join(improved_groups) or 'none'}. Correlation, regularization and sampling affect these descriptive comparisons.",
    "preprocessing_conclusion": f"Scaling before imputation changed validation AP by {order_a_delta:+.4f}. Impute → scale → clip changed AP by {order_b_delta:+.4f} versus clip → impute → scale. For {distribution_feature} ({int(X_train[distribution_feature].isna().sum())} training missing values), the maximum absolute A1/A0 transformed-value difference was {order_a_value_change:.4f}. All {len(protected_indices)} binary/indicator columns were preserved in both clipping orders. The primary comparison retained impute → scale.",
    "coverage_sensitivity_conclusion": f"Training logistic regression on 2003-2018 instead of 1994-2018 changed validation AP by {coverage_ap_delta:+.4f}, with the same estimator configuration and validation rows. Qualifying-position missingness changed from {coverage_sensitivity.iloc[0]['qualifying_missing_rate']:.1%} to {coverage_sensitivity.iloc[1]['qualifying_missing_rate']:.1%}. Sample size and era change too; this does not isolate the effect of qualifying coverage. Primary training remains 1994-2018.",
    "xai_conclusion": f"Leading permutation features: {top_permutation}. Leading mean absolute SHAP features: {top_shap}. SHAP units: {shap_units}; global validation sample size {sample_budget}, stratified in proportion to season counts ({shap_sampling_description}). Sampling uses no outcomes. A separately appended local-only case is excluded from global averages and beeswarm plots. Correlated inputs can share or substitute information; independent perturbations may form unrealistic combinations. None of these measures establishes causality.",
    "local_explanation": f"Same local case for both explainers: validation raceId={int(local_metadata['raceId'])}, driverId={int(local_metadata['driverId'])}; probability={local_probability:.3f}, actual podium={local_actual}, threshold={final_threshold:.3f}. Positive SHAP contributors: {', '.join(local_positive) or 'none'}; negative contributors: {', '.join(local_negative) or 'none'}. Reconstruction={shap_reconstruction:.3f} (absolute error {shap_reconstruction_error:.6f}); LIME local surrogate R²={lime_explanation.score:.3f}. SHAP displays scaled values; LIME displays imputed original-unit values and binary flags, with frozen scaling inside its prediction wrapper.",
    "inference_conclusion": f"Complete-record probability={inference_complete['podium_probability']:.3f}; partially missing={inference_partial['podium_probability']:.3f}, with {len(inference_partial['imputed_features'])} fields filled using training medians. Missing-grid probability={inference_missing_grid['podium_probability']:.3f}; only grid is imputed and its flag derives as 1. Null/zero/negative grid agree; contradictory flags, unknown keys and nonfinite numbers are rejected. All three native reloaded models match the original predictions. Inputs are engineered pre-race features, not raw names or lookup queries.",
    "limitations": "One chronological validation period; repeated validation experiments can overfit; historical regulation/scoring/qualifying drift; no external probability calibration; independent binary calls need not produce exactly three podiums; absent weather, tyre and strategy inputs; observational importance and home association are not causal.",
    "explained_model": explained_model_name, "shap_units": shap_units,
    "lime_R2": float(lime_explanation.score), "model_seconds": float(time.perf_counter() - model_start),
    "neural_implementation": "Keras Sequential, one hidden ReLU layer, one sigmoid output; binary cross-entropy, Adam, exact validation-AP callback before EarlyStopping with best-weight restoration; native .keras persistence",
}
modeling_summary['local_explanation'] += (
    f" The local surrogate explains about {100 * lime_explanation.score:.1f}% of the weighted "
    "variation in its sampled neighborhood. "
    + ("Substantial unexplained variation limits this LIME explanation; treat its feature weights as approximate."
       if lime_explanation.score < 0.7 else
       "Its neighborhood fit does not establish predictive correctness or causal effects.")
)
(OUT / "modeling_summary.json").write_text(json.dumps(modeling_summary, indent=2), encoding="utf-8")
display(Markdown(f"**Validation winner:** {final_model_name}, AP {winner_val_ap:.3f}. "
    f"**2022-2024 holdout:** ROC-AUC {test_metric_dict['ROC_AUC']:.3f}, AP {test_metric_dict['AP']:.3f}, "
    f"trapezoidal PR-AUC {test_metric_dict['PR_AUC']:.3f}, F1 {test_metric_dict['F1']:.3f}, "
    f"precision {test_metric_dict['precision']:.3f}, recall {test_metric_dict['recall']:.3f}.\n\n" +
    "\n\n".join(modeling_summary[key] for key in ["ablation_conclusion", "preprocessing_conclusion",
        "coverage_sensitivity_conclusion", "xai_conclusion", "local_explanation", "inference_conclusion"])))
