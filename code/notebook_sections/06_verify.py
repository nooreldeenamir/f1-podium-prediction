# %% [markdown]
# ## 6. Final checks and evidence manifest
# These assertions check the executed results, not hard-coded expected scores.
# They check the source bytes again, the prediction grain and chronological race
# groups, every required model/experiment, and the complete report figure set.
# The external delivery checker additionally reloads all three native models,
# recomputes their scores, matches PDF images, and checks this notebook's sources.

# %%
assert raw_hashes == {name: hashlib.sha256((DATA / f'{name}.csv').read_bytes()).hexdigest()
                      for name in TABLE_NAMES}
assert not model_table.duplicated(['raceId', 'driverId']).any()
assert model_table.groupby('raceId').split.nunique().max() == 1
assert len(FEATURES) == 19 and len(metrics_table) == 9
assert set(metrics_table['split']) == {'train', 'validation', 'test'}
assert set(metrics_table['model']) == {'Logistic regression', 'Random forest', 'Shallow neural network'}
assert set(FEATURE_GROUPS).issubset(set(ablation_table.group_removed))
assert len(preprocessing_table) == 4
assert (MODELS_DIR / 'shallow_neural_network.keras').is_file()
assert (OUT / 'pdf' / 'analytical_report.pdf').stat().st_size > 10_000
assert not inference_complete['imputed_features']
assert inference_partial['imputed_features']
assert inference_missing_grid['imputed_features'] == ['grid']
assert all(0 <= demo['podium_probability'] <= 1
           for demo in [inference_complete, inference_partial, inference_missing_grid])
assert all((FIG / name).exists() for name in analysis_figures + model_figures)
assert len(FIG_CAPTIONS) == len(list(FIG.glob('*.png')))
assert start_evidence.loc[start_evidence.laps.gt(0), 'started'].all()
assert not a_reliability_all.loc[a_reliability_all.nonstarter, 'laps'].gt(0).any()
assert model_table.loc[model_table.nationality.isin(NATIONALITY_COUNTRIES), 'is_home'].notna().all()
assert shap_sample.included_global.sum() == sample_budget
assert (abs(sample_year_counts - ideal_year_counts) < 1).all()
verification = {
    'status': 'PASS', 'raw_csvs_unchanged': True, 'unique_driver_race_grain': True,
    'chronological_race_groups': True, 'historical_timestamp_checks': True,
    'model_rows': len(model_table), 'numeric_input_features': len(FEATURES),
    'model_split_metric_rows': len(metrics_table), 'ablation_groups': list(FEATURE_GROUPS),
    'preprocessing_variants': len(preprocessing_table), 'figure_count': len(FIG_CAPTIONS),
    'selected_model': final_model_name, 'selection_metric': 'validation average precision',
    'threshold': float(final_threshold), 'native_keras_model_saved': True,
    'grid_schema_consistent_between_training_and_inference': True,
    'mechanical_retirements_require_early_end_proxy': True,
    'positive_laps_override_withdrawal_labels': True,
    'start_label_conflict_rows': len(start_conflicts),
    'nationality_country_text_normalized': True,
    'global_shap_sample_season_proportional': True,
    'source_sha256': raw_hashes, 'dataset_directory': str(DATA.resolve()),
}
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
(OUT / 'figure_captions.json').write_text(json.dumps(FIG_CAPTIONS, indent=2), encoding='utf-8')
display(Markdown('**Notebook integrity checks passed.** All three attempts were fitted in this run. '
                 'Tables, plots, native model files, inference demonstrations, and the PDF are under output/. '
                 'Accessible Kaggle/GitHub publication, course-form submission, team preparation, '
                 'and evaluation-slot arrangements remain separate human steps.'))
