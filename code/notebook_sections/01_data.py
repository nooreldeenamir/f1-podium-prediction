# %% [markdown]
# # Formula 1 Podium Prediction: lab-aligned implementation
# **Milestone 1 | reproducible Pandas analysis, sklearn models and Keras network**
#
# One prediction means one driver entering one race. The label is `positionOrder <= 3`.
# The cutoff is after qualifying and after the final starting grid is known, immediately
# before the Grand Prix starts. Current-race results, points, status, lap times, pit stops
# and post-race standings cannot be input features. This is an observational project;
# associations and explanations do not establish causation.
#
# **Navigation:** 1. Setup, audit and cleaning; 2. Three relational analyses;
# 3. Feature evidence and chronological dataset; 4. Models, experiments, explanations
# and inference; 5. Analytical report and final verification.
#
# The supplied brief lists three evaluation metrics but later refers to five. We report
# ROC-AUC, average precision (AP), trapezoidal PR-AUC, precision, recall and F1, explicitly
# distinguishing the two precision-recall summaries. Local LIME and local SHAP are both
# included. No performance numbers in this notebook are hand-entered.

# %% [markdown]
# ## 1. Setup and reproducibility
# Use **Run All** after installing the documented requirements. The setup cell checks
# dependencies and gives a clear installation message if anything is absent.
# Locally, search the working directory and its ancestors for the complete `Dataset/`
# folder, including when this notebook is opened from `Deliverables/1- Notebook/`.
# On Kaggle, discover the complete 14-table archive under `/kaggle/input`.
# An explicit `F1_DATA_DIR` environment variable overrides discovery. All output goes to
# `output/` or `/kaggle/working/output/`. Raw CSVs are never overwritten.

# %%
import importlib.util
_required = {'pandas': 'pandas', 'numpy': 'numpy', 'matplotlib': 'matplotlib',
             'seaborn': 'seaborn', 'sklearn': 'scikit-learn', 'tensorflow': 'tensorflow',
             'shap': 'shap', 'lime': 'lime', 'joblib': 'joblib', 'reportlab': 'reportlab',
             'PIL': 'pillow', 'pygments': 'pygments', 'fitz': 'pymupdf'}
_missing = [package for module, package in _required.items() if importlib.util.find_spec(module) is None]
if _missing:
    raise ModuleNotFoundError('Missing dependencies: ' + ', '.join(_missing) +
        '. Install requirements.txt in this kernel environment, then Restart and Run All. '
        'On Kaggle, install the missing packages in the environment before this notebook.')
from pathlib import Path
import os, sys, json, hashlib, platform, importlib.metadata, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')

SEED = 42
np.random.seed(SEED)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 170, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
sns.set_theme(style='whitegrid', palette='colorblind')

def locate_data():
    required = ['circuits', 'constructors', 'constructor_results', 'constructor_standings',
                'drivers', 'driver_standings', 'races', 'results', 'qualifying', 'lap_times',
                'pit_stops', 'sprint_results', 'seasons', 'status']
    configured = os.environ.get('F1_DATA_DIR')
    if configured:
        candidate = Path(configured).expanduser().resolve()
        missing = [name + '.csv' for name in required if not (candidate / (name + '.csv')).is_file()]
        if missing:
            raise FileNotFoundError('F1_DATA_DIR is missing required CSVs: ' + ', '.join(missing))
        return candidate
    working = Path.cwd().resolve()
    candidates = [working]
    for parent in [working, *working.parents]:
        candidates.extend([parent / 'Dataset', parent / 'dataset'])
    if Path('/kaggle/input').is_dir():
        candidates.extend(p.parent for p in sorted(Path('/kaggle/input').rglob('results.csv')))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / (name + '.csv')).is_file() for name in required):
            return candidate
    raise FileNotFoundError('No complete F1 archive found. Place all 14 CSVs in the project Dataset/ '
                            'folder, attach the archive on Kaggle, or set F1_DATA_DIR.')

DATA = locate_data()
OUT = Path('/kaggle/working/output') if Path('/kaggle/working').exists() else Path('output')
FIG = OUT / 'figures'
for directory in [OUT, FIG, OUT / 'tables', OUT / 'models', OUT / 'pdf']:
    directory.mkdir(parents=True, exist_ok=True)
FIG_CAPTIONS = {}
def savefig(name, caption=None):
    filename = name if name.endswith('.png') else name + '.png'
    plt.gcf().savefig(FIG / filename, bbox_inches='tight', facecolor='white')
    FIG_CAPTIONS[filename] = caption or filename.replace('.png', '').replace('_', ' ')
    plt.show()
    plt.close()

TABLE_NAMES = ['circuits', 'constructors', 'constructor_results', 'constructor_standings',
               'drivers', 'driver_standings', 'races', 'results', 'qualifying', 'lap_times',
               'pit_stops', 'sprint_results', 'seasons', 'status']
assert all((DATA / f'{t}.csv').exists() for t in TABLE_NAMES), 'Dataset is missing required tables.'
raw_hashes = {t: hashlib.sha256((DATA / f'{t}.csv').read_bytes()).hexdigest() for t in TABLE_NAMES}
versions = {m: importlib.metadata.version(m) for m in
            ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'tensorflow', 'keras', 'shap', 'lime', 'reportlab']}
display(pd.Series({'Python': platform.python_version(), **versions}, name='version').to_frame())
(OUT / 'environment.json').write_text(json.dumps(versions, indent=2))
(OUT / 'raw_hashes_before.json').write_text(json.dumps(raw_hashes, indent=2))

# %% [markdown]
# ### 1.1 Initial EDA: inspect the data programmatically before cleaning
# Load strings with automatic missing-value conversion disabled so the sentinel audit
# counts the literal `\N` rather than silently losing it. Report dimensions, duplicate
# rows, sentinel counts and empty strings for every table. These are implemented checks,
# not conclusions from manually opening a few rows.

# %%
raw_tables = {t: pd.read_csv(DATA / f'{t}.csv', dtype=str, keep_default_na=False) for t in TABLE_NAMES}
raw_audit = pd.DataFrame([{'table': t, 'rows': len(df), 'columns': len(df.columns),
                         'sentinel_cells': int(df.eq('\\N').sum().sum()),
                         'blank_cells': int(df.apply(lambda s: s.str.fullmatch(r'\s*')).sum().sum()),
                         'duplicate_full_rows': int(df.duplicated().sum())}
                        for t, df in raw_tables.items()])
display(raw_audit)
raw_audit.to_csv(OUT / 'tables' / 'raw_audit.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 4))
raw_audit.set_index('table')[['sentinel_cells', 'blank_cells']].plot.bar(ax=ax, stacked=True)
ax.set(ylabel='Cells', title='Raw placeholder missing values across 14 tables')
plt.xticks(rotation=45, ha='right')
savefig('01_raw_placeholders', 'Initial EDA: literal sentinel and blank counts before cleaning.')

raw_results = raw_tables['results']
raw_dates = raw_tables['races'][['raceId', 'year']].copy()
raw_year_counts = raw_results.merge(raw_dates, on='raceId', validate='many_to_one').groupby('year').size()
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(raw_year_counts.index.astype(int), raw_year_counts.values)
ax.set(xlabel='Season', ylabel='Result records', title='Raw result records by season')
savefig('02_raw_season_volume', 'Initial EDA: calendar length and entry counts change across seasons.')

# %% [markdown]
# ### 1.2 Cleaning policy and semantic types
# Convert `\N` and blanks into missing values on **copies**. IDs and counts become
# nullable integers; measurement columns become numeric; race/session dates and driver
# birth dates become datetime objects. Race clock times remain clock strings.
# Durations such as `1:23.456` become seconds; race result `+5.432` means a gap, not an
# elapsed race duration, so it is intentionally not coerced as a total duration.
# Qualifying Q2/Q3 absence frequently means elimination, rather than a corrupt record.
# Those values remain missing and are audited; they are not filled with zero lap times.

# %%
def duration_seconds(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.startswith('+'):
        return np.nan
    try:
        pieces = [float(p) for p in text.split(':')]
        if len(pieces) > 3 or any(p < 0 for p in pieces):
            return np.nan
        return sum(p * (60 ** i) for i, p in enumerate(reversed(pieces)))
    except ValueError:
        return np.nan

int_cols = {'year', 'round', 'number', 'grid', 'position', 'positionOrder', 'laps',
            'fastestLap', 'rank', 'wins', 'lap', 'stop', 'milliseconds'}
float_cols = {'points', 'fastestLapSpeed', 'lat', 'lng', 'alt'}
tables, coercion_log = {}, []
for t, raw in raw_tables.items():
    df = raw.replace({'\\N': np.nan, '': np.nan}).replace(r'^\s*$', np.nan, regex=True).copy(deep=True)
    for col in df.columns:
        if col.endswith('Id') or col in int_cols or col in float_cols:
            before = df[col].notna()
            numeric = pd.to_numeric(df[col], errors='coerce')
            newly_missing = int((before & numeric.isna()).sum())
            coercion_log.append({'table': t, 'column': col, 'invalid_nonmissing_values': newly_missing})
            df[col] = numeric.astype('Int64') if col.endswith('Id') or col in int_cols else numeric
        elif col == 'dob' or col == 'date' or col.endswith('_date'):
            before = df[col].notna()
            df[col] = pd.to_datetime(df[col], errors='coerce')
            coercion_log.append({'table': t, 'column': col,
                                 'invalid_nonmissing_values': int((before & df[col].isna()).sum())})
    duration_cols = {'qualifying': ['q1', 'q2', 'q3'], 'lap_times': ['time'],
                     'pit_stops': ['duration'], 'results': ['fastestLapTime'],
                     'sprint_results': ['fastestLapTime']}.get(t, [])
    for col in duration_cols:
        converted = df[col].map(duration_seconds)
        coercion_log.append({'table': t, 'column': col,
                             'invalid_nonmissing_values': int((df[col].notna() & converted.isna()).sum())})
        df[col + '_seconds'] = converted
    tables[t] = df
coercions = pd.DataFrame(coercion_log)
display(coercions.loc[coercions.invalid_nonmissing_values.gt(0)])
coercions.to_csv(OUT / 'tables' / 'typing_audit.csv', index=False)
assert coercions.invalid_nonmissing_values.sum() == 0, 'Investigate unexpected parsing loss.'
semantic_types = pd.DataFrame([
    {'table': table, 'column': column, 'dtype': str(frame[column].dtype),
     'observed_values': int(frame[column].notna().sum()),
     'missing_values': int(frame[column].isna().sum())}
    for table, frame in tables.items() for column in frame.columns
])
semantic_types.to_csv(OUT / 'tables' / 'semantic_types.csv', index=False)

# %% [markdown]
# ### 1.3 Relational integrity and duplicate-driver grain
# Validate table primary keys and every supplied ID relationship, including race season
# references. Merge validation prevents silently multiplying rows during joins.
# Some historical result records describe shared drives or alternate entries.
# For duplicate `(raceId, driverId)` pairs, retain the row with the smallest **positive
# grid position**, then the smallest `resultId`; if all grids are zero/missing, retain
# the smallest `resultId`. This deterministic rule uses entry/grid metadata, not the
# podium label or finishing position. It is an imperfect proxy for the original entry;
# we export every affected row and compare label counts before/after rather than hiding
# the historical ambiguity. All analyses use this declared driver-race grain.

# %%
PRIMARY_KEYS = {'circuits': ['circuitId'], 'constructors': ['constructorId'],
                'constructor_results': ['constructorResultsId'],
                'constructor_standings': ['constructorStandingsId'], 'drivers': ['driverId'],
                'driver_standings': ['driverStandingsId'], 'races': ['raceId'],
                'results': ['resultId'], 'qualifying': ['qualifyId'],
                'lap_times': ['raceId', 'driverId', 'lap'],
                'pit_stops': ['raceId', 'driverId', 'stop'], 'sprint_results': ['resultId'],
                'seasons': ['year'], 'status': ['statusId']}
key_audit_rows = []
for t, keys in PRIMARY_KEYS.items():
    df = tables[t]
    key_audit_rows.append({'table': t, 'keys': ','.join(keys),
                          'null_key_rows': int(df[keys].isna().any(axis=1).sum()),
                          'duplicate_key_rows': int(df.duplicated(keys).sum())})
key_audit = pd.DataFrame(key_audit_rows)
assert key_audit[['null_key_rows', 'duplicate_key_rows']].to_numpy().sum() == 0
foreign_rows = []
for t, df in tables.items():
    for col, parent in {'raceId': 'races', 'driverId': 'drivers', 'constructorId': 'constructors',
                        'circuitId': 'circuits', 'statusId': 'status', 'year': 'seasons'}.items():
        if col in df and t != parent:
            unmatched = df[col].notna() & ~df[col].isin(tables[parent][col])
            foreign_rows.append({'table': t, 'foreign_key': col, 'parent': parent,
                                 'null_references': int(df[col].isna().sum()),
                                 'unmatched_references': int(unmatched.sum())})
fk_audit = pd.DataFrame(foreign_rows)
assert fk_audit[['null_references', 'unmatched_references']].to_numpy().sum() == 0
display(key_audit)
display(fk_audit)
key_audit.to_csv(OUT / 'tables' / 'primary_key_audit.csv', index=False)
fk_audit.to_csv(OUT / 'tables' / 'foreign_key_audit.csv', index=False)

rr = tables['results'].copy()
rr['_grid_priority'] = rr.grid.where(rr.grid.gt(0), 10_000).fillna(10_000)
rr = rr.sort_values(['raceId', 'driverId', '_grid_priority', 'resultId'], kind='stable')
rr['_kept'] = ~rr.duplicated(['raceId', 'driverId'])
dedup_audit = rr.loc[rr.duplicated(['raceId', 'driverId'], keep=False)].drop(columns='_grid_priority')
dedup_audit = dedup_audit.merge(tables['races'][['raceId', 'year']], on='raceId', validate='many_to_one')
dedup_audit.to_csv(OUT / 'tables' / 'duplicate_driver_race_audit.csv', index=False)
results_unique = rr.loc[rr._kept].drop(columns=['_grid_priority', '_kept']).copy()
assert not results_unique.duplicated(['raceId', 'driverId']).any()
results_unique['y'] = results_unique.positionOrder.le(3).astype(int)
race_lookup = tables['races']

# A withdrawal/nonqualifying label is not sufficient evidence of a DNS when
# recorded laps show participation. Keep the same rule in analyses and histories.
START_EXCLUSION_STATUSES = {'Withdrew', 'Did not qualify', 'Did not prequalify', '107% Rule'}

def classify_race_starts(frame, status_column):
    positive_laps = pd.to_numeric(frame['laps'], errors='coerce').gt(0)
    exclusion_label = (frame['positionText'].astype('string').str.strip().str.upper().isin(['W', 'F'])
                       | frame[status_column].isin(START_EXCLUSION_STATUSES))
    return positive_laps | ~exclusion_label

start_evidence = results_unique.merge(tables['status'], on='statusId', validate='many_to_one')
start_evidence['started'] = classify_race_starts(start_evidence, 'status')
start_evidence['exclusion_label'] = (
    start_evidence.positionText.astype('string').str.strip().str.upper().isin(['W', 'F'])
    | start_evidence.status.isin(START_EXCLUSION_STATUSES))
start_conflicts = start_evidence.loc[
    start_evidence.exclusion_label & start_evidence.laps.gt(0),
    ['raceId', 'driverId', 'constructorId', 'grid', 'laps', 'positionText', 'status', 'started']
].merge(tables['races'][['raceId', 'year', 'date', 'name']], on='raceId', validate='many_to_one')
assert start_conflicts.started.all()
start_conflicts.to_csv(OUT / 'tables' / 'start_classification_conflicts.csv', index=False)
display(Markdown(f"**Start-evidence audit:** {len(start_conflicts)} unique driver-race records "
                 "have withdrawal/nonqualifying labels despite positive recorded laps. "
                 "They remain starts. Zero-lap W/F or explicit nonstart labels are excluded; "
                 "N is unclassified, and zero grid/laps alone does not prove a DNS. "
                 "This retrospective rule is used only for descriptive counts and earlier-race histories, "
                 "never as a current-race predictor."))
after_audit = pd.DataFrame([{'table': t, 'rows': len(df), 'columns': len(df.columns),
                            'missing_cells': int(df.isna().sum().sum()),
                            'duplicate_full_rows': int(df.duplicated().sum())}
                           for t, df in tables.items()])
display(after_audit)
after_audit.to_csv(OUT / 'tables' / 'clean_audit.csv', index=False)
before_rate = tables['results'].positionOrder.le(3).mean()
after_rate = results_unique.y.mean()
clean_comparison = pd.DataFrame({'stage': ['Raw result records', 'Unique driver-race rows'],
                                'rows': [len(tables['results']), len(results_unique)],
                                'podium_rate': [before_rate, after_rate]})
clean_comparison.to_csv(OUT / 'tables' / 'cleaning_comparison.csv', index=False)
display(clean_comparison)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
sns.barplot(data=clean_comparison, x='stage', y='rows', ax=axes[0])
sns.barplot(data=clean_comparison, x='stage', y='podium_rate', ax=axes[1])
for ax in axes:
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=10)
axes[0].set_title('Effect of grain correction on row count')
axes[1].set_title('Effect on the target distribution')
savefig('03_duplicate_cleaning_effect', 'Before/after EDA: deterministic driver-race deduplication and target prevalence.')

missing_compare = []
for col in ['grid', 'position', 'points', 'fastestLapTime', 'milliseconds']:
    missing_compare.append({'field': col, 'stage': 'Raw placeholders',
                            'fraction': raw_tables['results'][col].isin(['\\N', '']).mean()})
    missing_compare.append({'field': col, 'stage': 'Clean NA',
                            'fraction': tables['results'][col].isna().mean()})
fig, ax = plt.subplots(figsize=(9, 3.5))
sns.barplot(data=pd.DataFrame(missing_compare), x='field', y='fraction', hue='stage', ax=ax)
ax.set(title='Sentinel conversion preserves the missingness pattern', ylabel='Missing fraction', xlabel='')
savefig('04_missingness_cleaning_effect', 'Before/after EDA: parsing changes representation, without fabricating observed values.')

qcoverage = results_unique[['raceId', 'driverId']].merge(
    tables['qualifying'][['raceId', 'driverId', 'position']], on=['raceId', 'driverId'],
    how='left', validate='one_to_one').merge(tables['races'][['raceId', 'year']], on='raceId', validate='many_to_one')
coverage = qcoverage.groupby('year').agg(entries=('driverId', 'size'),
                                        qualifying_observed=('position', 'count'))
coverage['qualifying_coverage'] = coverage.qualifying_observed / coverage.entries
coverage.to_csv(OUT / 'tables' / 'qualifying_coverage.csv')
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(coverage.index, coverage.qualifying_coverage, marker='.', label='Observed qualifying position')
ax.axvline(1994, color='grey', ls='--', label='Modeling period begins')
ax.axvline(2003, color='orange', ls=':', label='More consistent coverage')
ax.set(xlabel='Season', ylabel='Fraction of entries', ylim=(0, 1.05), title='Qualifying coverage changes sharply over time')
ax.legend(loc='upper left', fontsize=8)
savefig('05_qualifying_coverage', 'Coverage evidence: historical qualifying absence motivates period restriction and missing-value indicators.')

fig, ax = plt.subplots(figsize=(9, 3.5))
dur = tables['qualifying']['q1_seconds'].dropna()
sns.histplot(dur.loc[dur.le(dur.quantile(.99))], bins=50, ax=ax)
ax.set(title='Typed qualifying Q1 times (display truncated at 99th percentile)', xlabel='Seconds')
savefig('06_typed_qualifying_duration', 'Semantic typing: lap-time strings converted to seconds; tail truncation is display-only, no raw records removed.')

# Lab 1's describe() provides observed-data range evidence after semantic typing.
# These are audit summaries; no imputation has been performed at this stage.
clean_numeric_summary = tables['results'][['grid', 'positionOrder', 'points', 'laps']].describe().T
clean_numeric_summary.to_csv(OUT / 'tables' / 'clean_numeric_summary.csv')
display(clean_numeric_summary)

data_summary = {'raw_results': len(tables['results']), 'unique_results': len(results_unique),
                'duplicates_removed': len(tables['results']) - len(results_unique),
                'duplicate_pairs': dedup_audit[['raceId', 'driverId']].drop_duplicates().shape[0],
                'podium_rate': float(after_rate), 'min_year': int(tables['races'].year.min()),
                'max_year': int(tables['races'].year.max()), 'foreign_key_errors': 0}
display(Markdown(f"**Audit conclusion:** {data_summary['raw_results']:,} result records become "
                 f"{data_summary['unique_results']:,} unique driver-race rows after removing "
                 f"{data_summary['duplicates_removed']} extra rows across {data_summary['duplicate_pairs']} pairs. "
                 f"Podium prevalence changes from {before_rate:.2%} to {after_rate:.2%}. "
                 "All declared primary and foreign key checks pass. Qualifying starts in 1994, "
                 "but 1996-2002 coverage is incomplete; missingness is retained and modeled explicitly."))
