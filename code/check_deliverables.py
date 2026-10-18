"""Independently verify the executed notebook, native artifacts and report."""
from pathlib import Path
import ast
import hashlib
import json
import re
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import fitz
import joblib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             auc, f1_score, precision_score, recall_score, brier_score_loss)
from inference import PodiumPredictor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'


def expected_cells():
    result = []
    for path in sorted((ROOT / 'code' / 'notebook_sections').glob('*.py')):
        source = path.read_text(encoding='utf-8')
        ast.parse(source)
        chunks = re.split(r'^# %%([^\n]*)\n', source, flags=re.M)
        for index in range(1, len(chunks), 2):
            kind, body = chunks[index], chunks[index + 1].rstrip()
            if '[markdown]' in kind:
                body = '\n'.join(line[2:] if line.startswith('# ') else
                                 line[1:] if line.startswith('#') else line for line in body.splitlines())
                result.append(('markdown', body))
            elif body.strip():
                result.append(('code', body))
    return result


def scores(y, probabilities, threshold):
    precision, recall, _ = precision_recall_curve(y, probabilities)
    predicted = probabilities >= threshold
    return {'ROC_AUC': roc_auc_score(y, probabilities), 'AP': average_precision_score(y, probabilities),
            'PR_AUC': auc(recall, precision), 'F1': f1_score(y, predicted),
            'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0),
            'Brier': brier_score_loss(y, probabilities)}


def main():
    notebook = nbformat.read(ROOT / 'F1_Podium_Prediction.ipynb', as_version=4)
    assert [(c.cell_type, c.source) for c in notebook.cells] == expected_cells(), 'Notebook/source mismatch.'
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not [o for c in code for o in c.get('outputs', []) if o.output_type == 'error']
    for cell in code:
        ast.parse(cell.source)
    all_source = '\n'.join(c.source for c in code)
    assert 'import torch' not in all_source and 'from torch' not in all_source
    assert 'QuantileClipper' not in all_source and 'BCEWithLogitsLoss' not in all_source
    assert 'EarlyStopping' in all_source and 'restore_best_weights=True' in all_source
    assert 'validation_split=' not in all_source
    for number in [1, 2, 3]:
        assert any(c.cell_type == 'markdown' and f'Attempt {number}' in c.source for c in notebook.cells)
    run_file = OUT / 'execution_run.json'
    if not run_file.exists():
        run_file = ROOT / 'tmp/fresh_run.json'
    run = json.loads(run_file.read_text())
    assert run['status'] == 'PASS' and run['fresh_kernel']

    verification = json.loads((OUT / 'verification.json').read_text())
    assert verification['status'] == 'PASS'
    candidates = [Path(os.environ['F1_DATA_DIR'])] if os.environ.get('F1_DATA_DIR') else []
    candidates += [ROOT / 'Dataset', ROOT.parent / 'Dataset', Path(verification['dataset_directory'])]
    data_dir = next((p for p in candidates if all((p / f'{name}.csv').is_file()
                                                for name in verification['source_sha256'])), None)
    assert data_dir is not None, 'Set F1_DATA_DIR to the 14 supplied CSV tables.'
    for name, checksum in verification['source_sha256'].items():
        assert hashlib.sha256((data_dir / f'{name}.csv').read_bytes()).hexdigest() == checksum
    for name, cols in [('primary_key_audit.csv', ['null_key_rows', 'duplicate_key_rows']),
                       ('foreign_key_audit.csv', ['null_references', 'unmatched_references'])]:
        audit = pd.read_csv(OUT / 'tables' / name)
        assert audit[cols].to_numpy().sum() == 0

    metadata = json.loads((OUT / 'models/metadata.json').read_text())
    bundle = joblib.load(OUT / 'models/preprocessing.joblib')
    table = pd.read_csv(OUT / 'modeling_table.csv', float_precision='round_trip')
    assert not table.duplicated(['raceId', 'driverId']).any()
    assert len(metadata['features']) == 19
    train = table.loc[table.year.le(2018), metadata['features']]
    pd.testing.assert_series_equal(bundle['medians'], train.median(), check_names=False)
    assert bundle['indicator_features'] == train.columns[train.isna().any()].tolist()
    assert int(bundle['scaler'].n_samples_seen_) == len(train)
    for feature, flag in zip(table.grid, table.grid_unassigned):
        assert int(flag) == int(pd.isna(feature))
    keys = [set(table.loc[table.year.between(lo, hi), 'raceId'])
            for lo, hi in [(1994, 2018), (2019, 2021), (2022, 2024)]]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    references = pd.read_csv(OUT / 'tables/feature_reference_checks.csv')
    assert len(references) >= 480 and references.passed.all()
    lineage = pd.read_csv(OUT / 'tables/feature_lineage.csv', parse_dates=['date'])
    for column in [c for c in lineage if c.endswith('_history_date')]:
        previous = pd.to_datetime(lineage[column])
        assert (previous.dropna() < lineage.loc[previous.notna(), 'date']).all()

    # Recount all starts from raw records, rather than trusting the exported
    # reference checks or assuming every withdrawal happened before the start.
    raw_results = pd.read_csv(data_dir / 'results.csv', na_values=['\\N'])
    raw_results['_priority'] = raw_results.grid.where(raw_results.grid.gt(0), 10000).fillna(10000)
    raw_results = raw_results.sort_values(['raceId', 'driverId', '_priority', 'resultId']).drop_duplicates(
        ['raceId', 'driverId'])
    raw_races = pd.read_csv(data_dir / 'races.csv', parse_dates=['date'])
    raw_history = raw_results.merge(raw_races[['raceId', 'date', 'year', 'circuitId']],
                                    on='raceId', validate='many_to_one').merge(
        pd.read_csv(data_dir / 'status.csv'), on='statusId', validate='many_to_one')
    raw_history = raw_history.sort_values(['date', 'raceId', 'driverId'])
    exclusion_labels = (raw_history.positionText.astype('string').str.strip().str.upper().isin(['W', 'F'])
        | raw_history.status.isin(['Withdrew', 'Did not qualify', 'Did not prequalify', '107% Rule']))
    raw_history['started_reference'] = raw_history.laps.gt(0) | ~exclusion_labels
    assert raw_history.loc[raw_history.laps.gt(0), 'started_reference'].all()
    raw_history['prior_starts_reference'] = raw_history.groupby('driverId').started_reference.transform(
        lambda values: values.astype(int).cumsum().shift(1).fillna(0))
    checked_starts = table.merge(raw_history[['raceId', 'driverId', 'prior_starts_reference']],
        on=['raceId', 'driverId'], validate='one_to_one')
    np.testing.assert_array_equal(checked_starts.driver_prior_starts, checked_starts.prior_starts_reference)
    conflict_keys = raw_history.loc[exclusion_labels & raw_history.laps.gt(0), ['raceId', 'driverId']]
    conflict_export = pd.read_csv(OUT / 'tables/start_classification_conflicts.csv')
    assert set(map(tuple, conflict_export[['raceId', 'driverId']].to_numpy())) == set(
        map(tuple, conflict_keys.to_numpy()))
    reliability_reference = raw_history.loc[raw_history.year.between(2014, 2024)
        & raw_history.started_reference].copy()
    reliability_reference['era'] = np.where(reliability_reference.year.le(2021), '2014–2021', '2022–2024')
    expected_denominators = reliability_reference.groupby(['constructorId', 'era']).size().sort_index()
    actual_denominators = pd.read_csv(OUT / 'tables/q3_constructor_reliability.csv').set_index(
        ['constructorId', 'era']).starts.sort_index()
    pd.testing.assert_series_equal(actual_denominators, expected_denominators, check_names=False)
    excluded = pd.read_csv(OUT / 'tables/q3_excluded_nonstarters.csv')
    assert not excluded.laps.gt(0).any(), 'A positive-lap participant was excluded as DNS.'

    # Reconstruct home flags from normalized raw dimensions, independently of the
    # feature table and training medians (which can hide an avoidable missing flag).
    home_reference = table[['raceId', 'driverId', 'circuitId', 'is_home']].merge(
        pd.read_csv(data_dir / 'drivers.csv')[['driverId', 'nationality']],
        on='driverId', validate='many_to_one').merge(
        pd.read_csv(data_dir / 'circuits.csv')[['circuitId', 'country']],
        on='circuitId', validate='many_to_one')
    home_reference['nationality'] = home_reference.nationality.str.strip()
    home_reference['country'] = home_reference.country.str.strip().replace({'United States': 'USA'})
    nationality_map = pd.read_csv(OUT / 'tables/q2_nationality_country_mapping.csv').set_index(
        'nationality').mapped_countries.str.split(' | ', regex=False).to_dict()
    expected_home = np.array([float(country in nationality_map[nationality])
        if nationality in nationality_map and pd.notna(country) else np.nan
        for nationality, country in zip(home_reference.nationality, home_reference.country)])
    np.testing.assert_allclose(home_reference.is_home, expected_home, equal_nan=True)

    metrics = pd.read_csv(OUT / 'tables/metrics.csv', float_precision='round_trip')
    assert len(metrics) == 9
    model_names = ['Logistic regression', 'Random forest', 'Shallow neural network']
    for name in model_names:
        predictor = PodiumPredictor(OUT / 'models', name)
        for split, low, high in [('train', 1994, 2018), ('validation', 2019, 2021), ('test', 2022, 2024)]:
            rows = table.loc[table.year.between(low, high)]
            frame = rows[metadata['features']]
            filled = frame.fillna(bundle['medians']).copy()
            for feature in bundle['indicator_features']:
                filled['missing_' + feature] = frame[feature].isna().astype(float)
            transformed = bundle['scaler'].transform(filled[bundle['transformed_features']])
            if name == 'Shallow neural network':
                probabilities = np.asarray(predictor.model(transformed.astype(np.float32), training=False)).ravel()
            else:
                probabilities = predictor.model.predict_proba(transformed)[:, 1]
            expected = metrics.loc[metrics.model.eq(name) & metrics.split.eq(split)].iloc[0]
            actual = scores(rows.y.to_numpy(), probabilities, predictor.threshold)
            for key, value in actual.items():
                np.testing.assert_allclose(value, expected[key], atol=2e-6, rtol=2e-6,
                                           err_msg=f'{name} {split} {key}')
            if split == 'validation':
                precision, recall, candidates = precision_recall_curve(rows.y, probabilities)
                candidate_f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
                    precision[:-1] + recall[:-1], 1e-12)
                optimal = candidates[np.flatnonzero(np.isclose(
                    candidate_f1, candidate_f1.max(), rtol=0, atol=1e-12))[-1]]
                np.testing.assert_allclose(predictor.threshold, optimal, rtol=0, atol=1e-12,
                                           err_msg=f'{name}: threshold is not validation-F1 optimum')

    comparison = pd.read_csv(OUT / 'tables/validation_attempt_comparison.csv').set_index('model')
    validation = metrics.loc[metrics.split.eq('validation')].set_index('model')
    for metric in ['ROC_AUC', 'AP', 'PR_AUC', 'F1', 'precision', 'recall', 'Brier', 'threshold']:
        np.testing.assert_allclose(comparison.loc[validation.index, metric], validation[metric])
    selected = validation.AP.idxmax()
    assert selected == metadata['selected_model']
    frozen = json.loads((OUT / 'selection_frozen.json').read_text())
    assert frozen['selected_model'] == selected and frozen['features'] == metadata['features']
    assert frozen['thresholds'] == metadata['all_thresholds']
    ablation = pd.read_csv(OUT / 'tables/ablation.csv')
    assert set(metadata['feature_groups']).issubset(set(ablation.group_removed))
    assert len(pd.read_csv(OUT / 'tables/preprocessing.csv')) == 4
    assert len(pd.read_csv(OUT / 'tables/training_window_sensitivity.csv')) == 2
    effects = pd.read_csv(OUT / 'tables/preprocessing_value_effects.csv')
    protected = effects.loc[effects['order'].str.startswith('B') & effects.binary_or_indicator_protected]
    assert (protected.changed_vs_A0 == 0).all()
    demonstrated = effects.loc[effects['order'].str.startswith('A1') & effects.training_missing_rows.gt(0)]
    assert demonstrated.maximum_absolute_change_vs_A0.max() > 1e-6
    explanation = json.loads((OUT / 'explanation_checks.json').read_text())
    assert explanation['model'] == selected
    shap_members = pd.read_csv(OUT / 'tables/shap_sample.csv')
    shap_by_year = pd.read_csv(OUT / 'tables/shap_sample_by_year.csv').set_index('year')
    assert not shap_members.validation_position.duplicated().any()
    assert shap_members.included_local.sum() == 1
    validation_rows = table.loc[table.year.between(2019, 2021)].reset_index(drop=True)
    member_reference = validation_rows.iloc[shap_members.validation_position.to_numpy()]
    np.testing.assert_array_equal(shap_members[['raceId', 'driverId', 'year']].to_numpy(),
                                  member_reference[['raceId', 'driverId', 'year']].to_numpy())
    global_year_counts = shap_members.loc[shap_members.included_global].groupby('year').size()
    expected_year_counts = validation_rows.groupby('year').size()
    pd.testing.assert_series_equal(shap_by_year.validation_rows, expected_year_counts, check_names=False,
                                    check_index_type=False)
    pd.testing.assert_series_equal(shap_by_year.global_sample_rows, global_year_counts, check_names=False,
                                    check_index_type=False)
    sample_size = int(global_year_counts.sum())
    assert sample_size == min(60 if selected == 'Shallow neural network' else 120, len(validation_rows))
    assert (abs(global_year_counts - sample_size * expected_year_counts / len(validation_rows)) < 1).all()
    assert explanation['global_sample_rows'] == sample_size
    assert explanation['local_only_excluded_from_global']

    predictor = PodiumPredictor(OUT / 'models')
    demos = json.loads((OUT / 'inference_demo_inputs.json').read_text())
    examples = pd.read_csv(OUT / 'tables/inference_examples.csv').set_index('example')
    for name, label in [('complete', 'complete'), ('partial', 'partially missing'), ('missing_grid', 'missing grid')]:
        result = predictor.predict_podium(demos[name])
        np.testing.assert_allclose(result['podium_probability'], examples.loc[label, 'podium_probability'], atol=1e-6)
    assert not predictor.predict_podium(demos['complete'])['imputed_features']
    assert predictor.predict_podium(demos['missing_grid'])['imputed_features'] == ['grid']
    np.testing.assert_allclose(predictor.predict_podium({**demos['complete'], 'grid': 0, 'grid_unassigned': 1})['podium_probability'],
                               predictor.predict_podium(demos['missing_grid'])['podium_probability'])
    for bad in [{**demos['complete'], 'grid': 0, 'grid_unassigned': 0},
                {**demos['complete'], 'grid': float('inf')},
                {**demos['complete'], 'grid': float('nan')},
                {**demos['complete'], 'grid_unassigned': float('nan')},
                {**demos['complete'], 'driver_age': float('nan')},
                {**demos['complete'], 'unknown_field': 1},
                {**demos['complete'], 'driver_age': 'invalid'}]:
        try:
            predictor.predict_podium(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Invalid inference record accepted.')

    inline = sum('image/png' in out.get('data', {}) for c in code for out in c.get('outputs', []))
    pngs = sorted((OUT / 'figures').glob('*.png'))
    assert inline == len(pngs)
    report_manifest = json.loads((OUT / 'report_figure_manifest.json').read_text(encoding='utf-8'))
    report_figure_names = set(report_manifest['figures'])
    available_pngs = {path.name: path for path in pngs}
    assert report_figure_names and report_figure_names.issubset(available_pngs)
    images = {}
    for name in report_figure_names:
        with Image.open(available_pngs[name]) as picture:
            images[hashlib.sha256(picture.convert('RGB').tobytes()).hexdigest()] = name
    with fitz.open(OUT / 'pdf/analytical_report.pdf') as pdf:
        text = '\n'.join(page.get_text() for page in pdf)
        numbers = set(map(int, re.findall(r'Figure\s+(\d+)\.', text)))
        assert numbers == set(range(1, len(report_figure_names) + 1))
        found = set()
        for page in pdf:
            for info in page.get_images(full=True):
                pixmap = fitz.Pixmap(pdf, info[0])
                if pixmap.n != 3:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                digest = hashlib.sha256(pixmap.samples).hexdigest()
                if digest in images:
                    found.add(images[digest])
            for block in page.get_text('dict')['blocks']:
                if block['type'] == 0:
                    x0, y0, x1, y1 = block['bbox']
                    assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width + 1 and y1 <= page.rect.height + 1
        assert found == report_figure_names, 'Missing or modified required report figure.'
        for phrase in ['Answers to the three data-engineering questions', 'Front-row',
                       'Home-country', 'mechanical retirements', 'Selected features',
                       'Known limitations', 'causal']:
            assert phrase.lower() in text.lower(), phrase
        pages = len(pdf)
    assert not (ROOT / 'lab_aligned_version').exists(), 'Superseded project layout remains.'
    for directory in ['code/notebook_sections', 'docs']:
        assert (ROOT / directory).is_dir(), directory
    qa_file = OUT / 'pdf_visual_qa.json'
    if qa_file.exists():
        qa = json.loads(qa_file.read_text())
        assert qa['status'] == 'PASS' and not qa['unresolved_material_layout_findings']
        assert qa['report_sha256'] == hashlib.sha256((OUT / 'pdf/analytical_report.pdf').read_bytes()).hexdigest()
        visual_status = 'PASS (recorded visual review matches current PDF hash)'
    else:
        visual_status = 'PENDING: visually inspect the report and record pdf_visual_qa.json'
    publication_file = OUT / 'publication.json'
    publication = json.loads(publication_file.read_text()) if publication_file.exists() else {}
    publication_status = ('Kaggle and GitHub published; course-form submission not performed'
                          if publication.get('status') == 'PUBLISHED' else 'PENDING')
    results = {'status': 'PASS', 'fresh_kernel_run': run, 'executed_code_cells': len(code),
               'notebook_sources_match': True, 'no_saved_errors': True,
               'project_layout': 'organized lab-aligned implementation at workspace root',
               'previous_implementation': 'removed at user request; supplied dataset retained',
               'raw_csv_hashes_match': True, 'model_rows': len(table), 'native_models_reloaded': model_names,
               'all_nine_model_split_metrics_recomputed': True, 'training_preprocessing_verified': True,
               'all_thresholds_verified_against_validation_f1': True,
               'independent_feature_reference_checks': len(references),
               'all_prior_starts_recounted_from_raw': len(checked_starts),
               'positive_lap_label_conflicts_retained_as_starts': len(conflict_export),
               'q3_start_denominators_reconciled': True,
               'normalized_home_flags_recomputed': len(home_reference),
               'shap_global_season_counts_verified': {str(int(year)): int(count)
                                                     for year, count in global_year_counts.items()},
               'history_timestamps_verified': True, 'protected_clipping_flags_verified': True,
               'inference_schema_and_demos_verified': True, 'inline_figures': inline,
               'pdf_figures_exact_pixel_matches': len(found), 'report_figures': sorted(report_figure_names), 'pdf_pages': pages,
               'external_publication_and_submission': publication_status, 'pdf_visual_review': visual_status}
    (OUT / 'delivery_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
