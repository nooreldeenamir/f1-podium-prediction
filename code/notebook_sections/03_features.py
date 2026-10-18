# %% [markdown]
# ## 3. Leakage-safe feature engineering and evidence
#
# **Prediction scenario:** sequential forecasting at each race's final pre-start cutoff.
# A later validation/test race may use results of earlier validation/test races once
# those races have completed. It may never use its own or a later race's outcomes.
# This simulates deployment with updating history, not a single forecast of a whole
# unseen season made on January 1. Model weights and preprocessing remain frozen.
#
# Build history on all cleaned years, then restrict model rows to **1994-2024**.
# This keeps qualifying-era observations while using older races as historical warm-up.
# A 2003+ sensitivity experiment later checks the effect of the sparse early qualifying
# years. Feature exploration below uses **training rows only**; the holdout is not used
# to choose features. Driver/constructor identifiers remain metadata, avoiding arbitrary
# numeric ID effects and improving behavior for new competitors.

# %%
history = results_unique.merge(tables['races'][['raceId', 'year', 'round', 'date', 'circuitId']],
                               on='raceId', how='left', validate='many_to_one')
history = history.merge(tables['status'].rename(columns={'status': 'status_text'}),
                         on='statusId', validate='many_to_one')
history = history.sort_values(['date', 'raceId', 'driverId']).reset_index(drop=True)
assert not history.date.isna().any()
NO_START = START_EXCLUSION_STATUSES
# Positive laps establish participation even when the archive says W/F/Withdrew.
# Without positive laps, exclude explicit W/F/nonstart labels. N remains a start;
# grid=0 or laps=0 alone does not prove DNS. Only earlier races feed experience.
history['_started'] = classify_race_starts(history, 'status_text')

def past_roll(frame, group, value, window=5):
    return frame.groupby(group, sort=False)[value].transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).mean())

# Driver histories are averages over the last five listed entries, not last five calendar races.
history['driver_form_podium_rate'] = past_roll(history, 'driverId', 'y')
history['driver_form_points'] = past_roll(history, 'driverId', 'points')
history['driver_form_finish'] = past_roll(history, 'driverId', 'positionOrder')
history['driver_prior_starts'] = history.groupby('driverId')['_started'].transform(
    lambda x: x.astype(int).cumsum().shift(1).fillna(0))
history['driver_history_date'] = history.groupby('driverId')['date'].shift(1)

# Aggregate constructor outcomes to ONE constructor-race row before shifting.
# Shifting individual driver rows would leak a teammate's result from the current race.
team_history = history.groupby(['constructorId', 'raceId', 'date'], as_index=False).agg(
    podium_rate=('y', 'mean'), points_per_entry=('points', 'mean'))
team_history = team_history.sort_values(['date', 'raceId', 'constructorId'])
team_history['constructor_form_podium_rate'] = past_roll(team_history, 'constructorId', 'podium_rate')
team_history['constructor_form_points'] = past_roll(team_history, 'constructorId', 'points_per_entry')
team_history['constructor_history_date'] = team_history.groupby('constructorId')['date'].shift(1)
history = history.merge(team_history[['constructorId', 'raceId', 'constructor_form_podium_rate',
                                     'constructor_form_points', 'constructor_history_date']],
                         on=['constructorId', 'raceId'], how='left', validate='many_to_one')

# Circuit history is a past-entry podium fraction averaged over its last five races.
# It mostly represents historical field size, so its value is tested by context ablation.
circuit_history = history.groupby(['circuitId', 'raceId', 'date'], as_index=False).agg(rate=('y', 'mean'))
circuit_history = circuit_history.sort_values(['date', 'raceId', 'circuitId'])
circuit_history['circuit_prior_podium_rate'] = past_roll(circuit_history, 'circuitId', 'rate')
circuit_history['circuit_history_date'] = circuit_history.groupby('circuitId')['date'].shift(1)
history = history.merge(circuit_history[['circuitId', 'raceId', 'circuit_prior_podium_rate', 'circuit_history_date']],
                         on=['circuitId', 'raceId'], validate='many_to_one')

# Standings after previous Grand Prix in the SAME season: never join current raceId snapshots.
schedule = tables['races'][['raceId', 'year', 'round', 'date']].sort_values(['year', 'round'])
schedule['previous_raceId'] = schedule.groupby('year')['raceId'].shift(1)
schedule['standings_history_date'] = schedule.groupby('year')['date'].shift(1)
history = history.merge(schedule[['raceId', 'previous_raceId', 'standings_history_date']],
                         on='raceId', validate='many_to_one')
for source, entity, prefix in [('driver_standings', 'driverId', 'driver'),
                                ('constructor_standings', 'constructorId', 'constructor')]:
    snapshot = tables[source][['raceId', entity, 'points', 'position']].rename(
        columns={'raceId': 'previous_raceId', 'points': prefix + '_standing_points',
                 'position': prefix + '_standing_position'})
    assert not snapshot.duplicated(['previous_raceId', entity]).any()
    history = history.merge(snapshot, on=['previous_raceId', entity], how='left', validate='many_to_one')
    # New season or no standings entry: zero observed season points and undefined rank.
    history[prefix + '_standing_points'] = history[prefix + '_standing_points'].fillna(0)

quali = tables['qualifying'][['raceId', 'driverId', 'position', 'q1_seconds', 'q2_seconds', 'q3_seconds']].copy()
assert not quali.duplicated(['raceId', 'driverId']).any()
quali['best_seconds'] = quali[['q1_seconds', 'q2_seconds', 'q3_seconds']].min(axis=1)
quali['quali_session_best'] = quali.groupby('raceId')['best_seconds'].transform('min')
quali['qualifying_gap_pct'] = 100 * (quali.best_seconds / quali.quali_session_best - 1)
quali = quali.rename(columns={'position': 'qualifying_position'})
history = history.merge(quali[['raceId', 'driverId', 'qualifying_position', 'qualifying_gap_pct']],
                         on=['raceId', 'driverId'], how='left', validate='one_to_one')

history = history.merge(tables['drivers'][['driverId', 'forename', 'surname', 'dob', 'nationality']],
                         on='driverId', validate='many_to_one')
history = history.merge(tables['circuits'][['circuitId', 'country', 'name']].rename(columns={'name': 'circuit_name'}),
                         on='circuitId', validate='many_to_one')
history['driver_age'] = (history.date - history.dob).dt.days / 365.2425
# Nationality is a proxy for sporting home country, not residence or birth country.
NATIONALITY_COUNTRIES = {
    'British': {'UK'}, 'German': {'Germany'}, 'French': {'France'}, 'Italian': {'Italy'},
    'Spanish': {'Spain'}, 'Finnish': {'Finland'}, 'Brazilian': {'Brazil'}, 'Argentine': {'Argentina'},
    'Australian': {'Australia'}, 'Austrian': {'Austria'}, 'Belgian': {'Belgium'},
    'Canadian': {'Canada'}, 'Dutch': {'Netherlands'}, 'American': {'USA'}, 'Swiss': {'Switzerland'},
    'Swedish': {'Sweden'}, 'Danish': {'Denmark'}, 'Japanese': {'Japan'}, 'Mexican': {'Mexico'},
    'New Zealander': {'New Zealand'}, 'Polish': {'Poland'}, 'Russian': {'Russia'},
    'Portuguese': {'Portugal'}, 'South African': {'South Africa'}, 'Indian': {'India'},
    'Hungarian': {'Hungary'}, 'Malaysian': {'Malaysia'}, 'Indonesian': {'Indonesia'},
    'Chinese': {'China'}, 'Thai': {'Thailand'}, 'Monegasque': {'Monaco'}, 'Chilean': {'Chile'},
    'Czech': {'Czech Republic'}, 'Liechtensteiner': {'Liechtenstein'}, 'Rhodesian': {'Zimbabwe'},
    'Uruguayan': {'Uruguay'}, 'Venezuelan': {'Venezuela'}, 'Colombian': {'Colombia'},
    'Argentine-Italian': {'Argentina', 'Italy'}, 'East German': {'Germany'},
    'Irish': {'Ireland'}, 'American-Italian': {'USA', 'Italy'}, 'Argentinian': {'Argentina'}
}
history['country'] = history.country.replace({'United States': 'USA'})
history['country'] = history.country.astype('string').str.strip()
history['nationality'] = history.nationality.astype('string').str.strip()
history['is_home'] = [float(country in NATIONALITY_COUNTRIES[nat]) if nat in NATIONALITY_COUNTRIES else np.nan
                      for nat, country in zip(history.nationality, history.country)]
mapped_home = history.nationality.isin(NATIONALITY_COUNTRIES) & history.country.notna()
assert history.loc[mapped_home, 'is_home'].notna().all()
history['grid'] = history.grid.where(history.grid.gt(0), np.nan)
# Zero grid can mean a pit-lane/no-grid entry; do not pretend it is better than pole position.
history['grid_unassigned'] = history['grid'].isna().astype(float)

FEATURE_GROUPS = {
    'grid': ['grid', 'grid_unassigned'],
    'qualifying': ['qualifying_position', 'qualifying_gap_pct'],
    'form': ['driver_form_podium_rate', 'driver_form_points', 'driver_form_finish', 'driver_prior_starts',
             'constructor_form_podium_rate', 'constructor_form_points'],
    'standings': ['driver_standing_points', 'driver_standing_position',
                  'constructor_standing_points', 'constructor_standing_position'],
    'context': ['driver_age', 'is_home', 'circuit_prior_podium_rate', 'year', 'round']
}
FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]
model_table = history.loc[history.year.between(1994, 2024)].sort_values(['date', 'raceId', 'driverId']).reset_index(drop=True)
for col in FEATURES:
    model_table[col] = pd.to_numeric(model_table[col], errors='coerce').astype(float)
model_table['y'] = model_table.y.astype(int)
assert not model_table.duplicated(['raceId', 'driverId']).any()
assert not np.isinf(model_table[FEATURES].to_numpy()).any()
assert model_table[FEATURES].notna().any().all(), 'A feature is entirely missing.'
for col in ['driver_history_date', 'constructor_history_date', 'circuit_history_date', 'standings_history_date']:
    observed = model_table[col].notna()
    assert (model_table.loc[observed, col] < model_table.loc[observed, 'date']).all(), col
assert not set(FEATURES) & {'positionOrder', 'points', 'statusId', 'laps', 'milliseconds', 'y', 'rank', 'fastestLap'}

# Additional lineage checks: first observed driver row has no form, and previous-race joins never self-join.
first_driver = history.sort_values(['date', 'raceId']).groupby('driverId').head(1)
assert first_driver.driver_form_podium_rate.isna().all()
assert not model_table.previous_raceId.eq(model_table.raceId).any()
# Independently recompute sample references using filters on earlier raw outcomes
# and the calendar. This avoids validating shift/merge code with the same shift.
reference_checks = []
def check_reference(row, feature, reference):
    actual = row[feature]
    actual = np.nan if pd.isna(actual) else float(actual)
    reference = np.nan if pd.isna(reference) else float(reference)
    passed = bool(np.isclose(actual, reference, equal_nan=True))
    reference_checks.append({'raceId': int(row.raceId), 'driverId': int(row.driverId),
        'feature': feature, 'actual': actual, 'independent_reference': reference, 'passed': passed})
    assert passed, f'Independent reference failed: {feature}, race {row.raceId}.'

for _, row in model_table.sample(40, random_state=SEED).iterrows():
    driver_all_past = history.loc[
        history.driverId.eq(row.driverId) & history.date.lt(row.date)
    ].sort_values(['date', 'raceId'])
    driver_past = driver_all_past.tail(5)
    for feature, original_field in [('driver_form_podium_rate', 'y'),
                                     ('driver_form_points', 'points'),
                                     ('driver_form_finish', 'positionOrder')]:
        check_reference(row, feature, driver_past[original_field].mean())
    independent_started = (driver_all_past.laps.gt(0)
        | (~driver_all_past.status_text.isin(NO_START)
           & ~driver_all_past.positionText.astype('string').str.strip().str.upper().isin(['W', 'F'])))
    check_reference(row, 'driver_prior_starts', independent_started.sum())

    # Rebuild the constructor-race aggregation from earlier driver rows instead
    # of trusting the previously engineered team_history table.
    team_past = history.loc[
        history.constructorId.eq(row.constructorId) & history.date.lt(row.date)
    ].groupby(['date', 'raceId'], as_index=False).agg(
        podium_rate=('y', 'mean'), points_per_entry=('points', 'mean')
    ).sort_values(['date', 'raceId']).tail(5)
    check_reference(row, 'constructor_form_podium_rate', team_past.podium_rate.mean())
    check_reference(row, 'constructor_form_points', team_past.points_per_entry.mean())
    circuit_past = history.loc[
        history.circuitId.eq(row.circuitId) & history.date.lt(row.date)
    ].groupby(['date', 'raceId']).y.mean().sort_index().tail(5)
    check_reference(row, 'circuit_prior_podium_rate', circuit_past.mean())

    # Identify the immediately previous ROUND in this season independently of
    # the engineered previous_raceId column, then look up that exact snapshot.
    earlier_rounds = tables['races'].loc[
        tables['races'].year.eq(row.year) & tables['races']['round'].lt(row['round'])
    ].sort_values('round')
    previous_round_id = earlier_rounds.raceId.iloc[-1] if len(earlier_rounds) else np.nan
    check_reference(row, 'previous_raceId', previous_round_id)
    for source, entity, prefix in [('driver_standings', 'driverId', 'driver'),
                                    ('constructor_standings', 'constructorId', 'constructor')]:
        reference_snapshot = tables[source].loc[
            tables[source].raceId.eq(previous_round_id) & tables[source][entity].eq(row[entity])
        ]
        assert len(reference_snapshot) <= 1
        reference_points = reference_snapshot.points.iloc[0] if len(reference_snapshot) else 0.0
        reference_rank = reference_snapshot.position.iloc[0] if len(reference_snapshot) else np.nan
        check_reference(row, prefix + '_standing_points', reference_points)
        check_reference(row, prefix + '_standing_position', reference_rank)
feature_reference_checks = pd.DataFrame(reference_checks)
feature_reference_checks.to_csv(OUT / 'tables' / 'feature_reference_checks.csv', index=False)
display(feature_reference_checks.groupby('feature', sort=False).agg(
    references=('passed', 'size'), all_passed=('passed', 'all')))
availability_audit = model_table[['raceId', 'driverId', 'date', 'driver_history_date',
                                  'constructor_history_date', 'circuit_history_date', 'standings_history_date']]
availability_audit.to_csv(OUT / 'tables' / 'feature_lineage.csv', index=False)
export_columns = ['raceId', 'driverId', 'constructorId', 'circuitId', 'date', 'y'] + FEATURES
model_table[export_columns].to_csv(OUT / 'modeling_table.csv', index=False)

# %% [markdown]
# ### 3.1 A chronological split that respects race groups
# Train: 1994-2018; validation: 2019-2021; test: 2022-2024. Keep every driver from the
# same race in the same partition. A random driver-row split would mix race conditions
# and later seasons into training. Validation covers three seasons for tuning, and the
# test spans the 2022 regulation change, deliberately measuring temporal transfer.
# We retain the full historical qualifying era, and explicitly test a 2003+ training
# restriction rather than silently discarding missing qualifying rows.

# %%
model_table['split'] = np.select([model_table.year.le(2018), model_table.year.le(2021)],
                                 ['train', 'validation'], default='test')
split_summary = model_table.groupby('split', sort=False).agg(
    rows=('y', 'size'), races=('raceId', 'nunique'), first_year=('year', 'min'),
    last_year=('year', 'max'), podium_rate=('y', 'mean')).reset_index()
assert model_table.groupby('raceId').split.nunique().max() == 1
assert model_table.loc[model_table.split.eq('train'), 'date'].max() < model_table.loc[model_table.split.eq('validation'), 'date'].min()
assert model_table.loc[model_table.split.eq('validation'), 'date'].max() < model_table.loc[model_table.split.eq('test'), 'date'].min()
display(split_summary)
split_summary.to_csv(OUT / 'tables' / 'split_summary.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.barplot(data=split_summary, x='split', y='rows', ax=axes[0])
sns.barplot(data=split_summary, x='split', y='podium_rate', ax=axes[1])
axes[0].set_title('Chronological partition sizes')
axes[1].set_title('Podium prevalence by partition')
savefig('10_chronological_split', 'Race-grouped chronological split: same-race entries stay together; prevalence measures class imbalance.')

# %% [markdown]
# ### 3.2 Evidence behind feature choices (training rows only)
# Grid and qualifying describe starting opportunity. Shifted driver/team form measures
# recent competitiveness, while previous-round standings measure current-season form.
# Age, home-country flag, calendar and circuit history are exploratory context features.
# They are hypotheses, not guaranteed improvements: group ablation measures their actual
# validation contribution. A session-wide qualifying time gap normalizes circuit length
# but mixes Q1/Q2/Q3 conditions and formats; its interpretation has this limitation.
# Past points reflect changes in the points system; year/round partly represent this
# context but cannot fully remove era differences.

# %%
train_evidence = model_table.loc[model_table.split.eq('train')].copy()
feature_stats = pd.DataFrame({'feature': FEATURES,
                              'missing_fraction': train_evidence[FEATURES].isna().mean().values,
                              'spearman_with_podium': [train_evidence[f].corr(train_evidence.y, method='spearman')
                                                       for f in FEATURES]})
feature_stats.to_csv(OUT / 'tables' / 'feature_evidence.csv', index=False)
display(feature_stats.round(3))
fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained')
feature_stats.sort_values('missing_fraction').plot.barh(x='feature', y='missing_fraction', ax=axes[0], legend=False)
feature_stats.sort_values('spearman_with_podium').plot.barh(x='feature', y='spearman_with_podium', ax=axes[1], legend=False)
axes[0].set(title='Training feature missingness', xlabel='Fraction missing')
axes[1].set(title='Training association with podium', xlabel='Spearman correlation')
savefig('11_feature_evidence', 'Training-only feature evidence: missingness and univariate association are descriptive, not causal or independent effects.')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
grid_rates = train_evidence.groupby('grid').y.agg(['mean', 'size']).loc[lambda d: d['size'].ge(30)]
axes[0].plot(grid_rates.index, grid_rates['mean'], marker='o')
axes[0].set(title='Starting grid and podium frequency (n >= 30)', xlabel='Grid', ylabel='Podium fraction')
form_binned = train_evidence.assign(form_bin=pd.cut(train_evidence.driver_form_podium_rate,
                                [-.001, .2, .4, .6, .8, 1.001]))
form_rates = form_binned.groupby('form_bin', observed=True).y.agg(['mean', 'size'])
axes[1].bar(range(len(form_rates)), form_rates['mean'])
axes[1].set_xticks(range(len(form_rates)), [str(i) for i in form_rates.index], rotation=20)
axes[1].set(title='Prior-five-entry form and podium frequency', xlabel='Previous podium fraction', ylabel='Podium fraction')
savefig('12_feature_effects', 'Feature motivation: training podium rates by grid and strictly shifted driver form.')

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(train_evidence[FEATURES].corr(method='spearman'), cmap='coolwarm', vmin=-1, vmax=1,
            center=0, ax=ax, square=True, cbar_kws={'shrink': .7})
ax.set_title('Training feature correlations: form and standings are redundant')
savefig('13_feature_correlation', 'Correlated predictors motivate regularized logistic regression and grouped ablation; importance can be shared among correlated features.')

availability = {
    'grid': 'Final published starting grid before race start; zero is treated as missing.',
    'qualifying': 'Current qualifying positions and completed qualifying lap times before start.',
    'form': 'Only earlier completed Grand Prix results; driver entry rolling mean and team race aggregation shifted once.',
    'standings': 'Snapshot after previous Grand Prix in same season; round one has zero points and missing rank.',
    'context': 'Scheduled year/round, birth date, sporting nationality/circuit country, and earlier circuit races.'}
feature_definitions = {
    'grid': ('Final grid position; nonpositive values become missing.',
             'Starting opportunity is expected to constrain podium chances.',
             'Median learned on training rows; grid_unassigned retains absence.'),
    'grid_unassigned': ('1 when the final grid is missing or nonpositive; otherwise 0.',
             'Distinguishes pit-lane/unrecorded grid from an ordinary midfield start.',
             'Derived from grid before imputation; never supplied inconsistently.'),
    'qualifying_position': ('Recorded classification position in completed qualifying.',
             'Reflects weekend pace while allowing penalties to change the actual grid.',
             'No observed session row stays missing, with a training-derived indicator.'),
    'qualifying_gap_pct': ('100 * (driver best Q1/Q2/Q3 seconds / race session best - 1).',
             'Normalizes lap duration across circuits, while acknowledging session-condition differences.',
             'Missing timing stays missing until train-fitted imputation.'),
    'driver_form_podium_rate': ('Mean podium label across this driver\'s previous five listed entries.',
             'Recent results summarize competitiveness using only completed races.',
             'Fewer than five prior entries use available history; debut remains missing.'),
    'driver_form_points': ('Mean points across this driver\'s previous five listed entries.',
             'Provides a graded recent-performance measure beyond a binary podium.',
             'No prior entries remains missing; changing scoring rules are a limitation.'),
    'driver_form_finish': ('Mean positionOrder across this driver\'s previous five listed entries.',
             'Adds recent finishing consistency, including non-podium outcomes.',
             'No prior entries remains missing; no current outcome is included.'),
    'driver_prior_starts': ('Cumulative earlier recorded starts, excluding explicit nonstarters.',
             'Represents experience and distinguishes debut from a missing form value.',
              'A debut has zero starts; positive recorded laps override W/F/withdrawal labels. Without positive laps, explicit nonstart labels are excluded. N remains a start; zero grid/laps alone does not prove DNS.'),
    'constructor_form_podium_rate': ('Previous-five-constructor-race mean of within-race entry podium fractions.',
             'Captures recent team strength without leaking a current-race teammate outcome.',
             'First constructor race remains missing; available earlier races otherwise used.'),
    'constructor_form_points': ('Previous-five-constructor-race mean of within-race points per entry.',
             'Summarizes recent team competitiveness without favoring larger historical lineups.',
             'First constructor race remains missing; team IDs do not inherit renamed teams.'),
    'driver_standing_points': ('Driver points in the snapshot after the preceding round of the same season.',
             'Summarizes current-season performance available before the current race.',
             'Round one or no prior snapshot entry has zero observed season points.'),
    'driver_standing_position': ('Driver championship position after the preceding same-season round.',
             'Adds relative current-season competitiveness alongside absolute points.',
             'Round one or absent prior snapshot leaves the rank missing.'),
    'constructor_standing_points': ('Constructor points after the preceding round of the same season.',
             'Represents accumulated team strength before the current race.',
             'Round one or absent prior snapshot entry has zero observed season points.'),
    'constructor_standing_position': ('Constructor championship position after the preceding same-season round.',
             'Adds the team\'s relative position under the current scoring system.',
             'Round one or absent prior snapshot leaves rank missing.'),
    'driver_age': ('Days from driver birth date to race date divided by 365.2425.',
             'Exploratory experience/context hypothesis, tested by context ablation.',
             'Missing date of birth would remain missing until training imputation.'),
    'is_home': ('1 if mapped sporting nationality includes the circuit country; otherwise 0.',
             'Tests the observed home-country association without treating it as causal.',
             'Unmapped nationality remains missing rather than being classified away.'),
    'circuit_prior_podium_rate': ('Mean entry podium fraction across the circuit\'s previous five races.',
             'Historical field-size/circuit context is an exploratory hypothesis.',
             'First observed circuit race remains missing.'),
    'year': ('Scheduled calendar season.',
             'Represents broad eras and scoring/format changes, with extrapolation limits.',
             'Required calendar metadata; present in every modeling row.'),
    'round': ('Scheduled round number within the season.',
             'Allows season progression to modify other competitiveness signals.',
             'Required calendar metadata; present in every modeling row.')
}
assert set(feature_definitions) == set(FEATURES)
feature_catalog = feature_stats.merge(pd.DataFrame([
    {'feature': f, 'group': g, 'definition': feature_definitions[f][0],
     'motivation': feature_definitions[f][1], 'missingness_policy': feature_definitions[f][2],
     'availability': availability[g],
     'evidence': 'Training missingness/correlation: 11_feature_evidence; grid/form rates: '
                 '12_feature_effects; redundancy: 13_feature_correlation; group ablation: feature_group_ablation.'}
    for g, fs in FEATURE_GROUPS.items() for f in fs]),
    on='feature', validate='one_to_one')
feature_catalog.to_csv(OUT / 'tables' / 'feature_catalog.csv', index=False)
display(feature_catalog)
strongest = feature_stats.set_index('feature').spearman_with_podium.abs().nlargest(3).index.tolist()
data_summary.update({'model_rows': len(model_table), 'model_features': len(FEATURES),
                     'feature_evidence': f'The largest training univariate associations are {", ".join(strongest)}.',
                     'cutoff': 'After qualifying and final grid, immediately before the Grand Prix starts.'})
(OUT / 'data_summary.json').write_text(json.dumps(data_summary, indent=2), encoding='utf-8')
display(Markdown(f"**Feature conclusion:** {len(FEATURES)} pre-race predictors for {len(model_table):,} rows. "
                 f"{data_summary['feature_evidence']} Every history timestamp is strictly before the target race. "
                 "Missing inputs are retained for train-fitted imputation, rather than removing hard cases."))
