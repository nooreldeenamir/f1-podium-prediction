# %% [markdown]
# ## 2. Relational analyses: three questions, three explicit denominators
#
# These analyses are descriptive. A join can create plausible but wrong answers if
# its grain changes, so every enrichment below validates its cardinality. The
# cleaned result grain is one driver at one race. Race outcomes are used here to
# answer historical questions; the predictive section separately restricts every
# feature to information available before the race.

# %%
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a_table_dir = OUT / "tables"
a_table_dir.mkdir(parents=True, exist_ok=True)
analyses_summary = {}
analysis_figures = []

def a_wilson(successes, totals, z=1.96):
    """95% Wilson intervals; descriptive binomial uncertainty, not causal CIs."""
    a_s = np.asarray(successes, dtype=float)
    a_n = np.asarray(totals, dtype=float)
    a_p = np.divide(a_s, a_n, out=np.full_like(a_s, np.nan), where=a_n > 0)
    a_d = 1 + z ** 2 / a_n
    a_c = (a_p + z ** 2 / (2 * a_n)) / a_d
    a_h = z * np.sqrt(a_p * (1 - a_p) / a_n + z ** 2 / (4 * a_n ** 2)) / a_d
    return a_c - a_h, a_c + a_h

def a_rates(a_frame, a_groups, a_outcome="podium"):
    a_result = a_frame.groupby(a_groups, observed=True, dropna=False)[a_outcome].agg(
        podiums="sum", starts="size"
    ).reset_index()
    a_result["rate"] = a_result["podiums"] / a_result["starts"]
    a_result["ci_low"], a_result["ci_high"] = a_wilson(
        a_result["podiums"], a_result["starts"]
    )
    return a_result

# Select and rename dimension columns BEFORE joining to prevent accidental use of
# similarly named race/circuit/constructor fields or automatic suffixes.
a_races = race_lookup[["raceId", "year", "round", "circuitId", "date"]].copy()
a_circuits = tables["circuits"][["circuitId", "name", "country"]].rename(
    columns={"name": "circuit_name", "country": "circuit_country"}
)
a_drivers = tables["drivers"][["driverId", "nationality"]].copy()
a_constructors = tables["constructors"][["constructorId", "name"]].rename(
    columns={"name": "constructor_name"}
)
a_statuses = tables["status"][["statusId", "status"]].rename(
    columns={"status": "race_status"}
)
a_results = results_unique[[
    "raceId", "driverId", "constructorId", "grid", "positionOrder",
    "positionText", "statusId", "laps"
]].copy()
a_results["podium"] = pd.to_numeric(a_results["positionOrder"], errors="coerce").le(3).astype(int)
assert not a_results.duplicated(["raceId", "driverId"]).any()
assert a_results["positionOrder"].notna().all(), "Unknown outcome must not be counted as a non-podium."
a_joined = a_results.merge(a_races, on="raceId", how="left", validate="many_to_one")
a_joined = a_joined.merge(a_circuits, on="circuitId", how="left", validate="many_to_one")
a_joined = a_joined.merge(a_drivers, on="driverId", how="left", validate="many_to_one")
a_joined = a_joined.merge(a_constructors, on="constructorId", how="left", validate="many_to_one")
a_joined = a_joined.merge(a_statuses, on="statusId", how="left", validate="many_to_one")
assert len(a_joined) == len(a_results)
assert a_joined[["year", "circuit_name", "constructor_name", "race_status"]].notna().all().all()

# %% [markdown]
# ### Q1. Which circuits convert front-row positions into podiums, and did ranks change after 2014?
#
# Qualifying position 1–2 and final starting grid position 1–2 are different
# exposures: penalties and sprint formats can move drivers between them. We
# compare both definitions on the **same races**, retaining races with two observed
# qualifying front-row starters, two observed grid front-row starters, and result
# records for all of them. W/F and explicit withdrawal/nonqualifying labels exclude
# an entry only when it has no positive recorded laps. Positive laps establish a
# start even if its archive label says Withdrew; conflicts are exported in section 1.
# `positionText=F` means failed to qualify; `N` means unclassified and remains a
# start. Classification status alone must not turn an actual starter into a DNS.
# This avoids counting a front-row DNS as a race start. The window begins in 1994 because earlier qualifying
# rows are unavailable. Before 2014 means 1994–2013; after the hybrid-era change
# means 2014–2024. Circuit identity follows `circuitId`, not a changing race name.
#
# Rankings require at least 20 driver starts (10 races) in a circuit/era/definition
# cell. Smaller cells remain in the exported table but do not determine ranks.
# Wilson bars show sampling uncertainty. Driver starts within a race are
# dependent, so these bars are descriptive and do not establish significant
# differences. Track calendars, weather, cars and field composition also change.

# %%
a_quali = tables["qualifying"][["raceId", "driverId", "position", "qualifyId"]].copy()
a_quali["qual_position"] = pd.to_numeric(a_quali["position"], errors="coerce")
# The supplied qualifying table is unique at this grain. A future duplicate must
# be investigated rather than resolved using its qualifying outcome.
a_q_duplicates = int(a_quali.duplicated(["raceId", "driverId"]).sum())
assert a_q_duplicates == 0, "Investigate duplicate qualifying driver-race rows before joining."
a_q_join = a_joined.merge(
    a_quali[["raceId", "driverId", "qual_position"]],
    on=["raceId", "driverId"], how="left", validate="one_to_one"
)
a_q_join = a_q_join.loc[a_q_join["year"].between(1994, 2024)].copy()
a_q_nonstart = ~classify_race_starts(a_q_join, "race_status")
a_q_starters = a_q_join[~a_q_nonstart].copy()
a_q_counts = a_q_starters.groupby("raceId", observed=True).agg(
    q_front=("qual_position", lambda a_v: int(a_v.isin([1, 2]).sum())),
    grid_front=("grid", lambda a_v: int(a_v.isin([1, 2]).sum()))
)
a_q_eligible = a_q_counts.index[(a_q_counts.q_front == 2) & (a_q_counts.grid_front == 2)]
a_q_common = a_q_starters[a_q_starters.raceId.isin(a_q_eligible)].copy()
a_q_common["era"] = np.where(a_q_common.year < 2014, "1994–2013", "2014–2024")
a_q_exposures = pd.concat([
    a_q_common[a_q_common.qual_position.isin([1, 2])].assign(definition="Qualifying 1–2"),
    a_q_common[a_q_common.grid.isin([1, 2])].assign(definition="Grid 1–2")
], ignore_index=True)
a_q1_rates = a_rates(a_q_exposures, ["circuitId", "circuit_name", "era", "definition"])
a_q1_rates["supported"] = a_q1_rates.starts.ge(20)
a_q1_rates["rank"] = np.nan
a_q1_rates.loc[a_q1_rates.supported, "rank"] = a_q1_rates[a_q1_rates.supported].groupby(
    ["era", "definition"], observed=True
)["rate"].rank(method="min", ascending=False)
a_q1_rates.to_csv(a_table_dir / "q1_frontrow_rates.csv", index=False)

# Drivers can move into/out of the front row even when the total is still two.
a_q_disagree = a_q_common.assign(
    qualifying_front=a_q_common.qual_position.isin([1, 2]),
    grid_front=a_q_common.grid.isin([1, 2])
)
a_q_disagree["front_row_membership_differs"] = a_q_disagree.qualifying_front.ne(a_q_disagree.grid_front)
a_q_disagreement = a_q_disagree.groupby("era", observed=True).agg(
    races=("raceId", "nunique"),
    driver_rows=("driverId", "size"),
    differing_memberships=("front_row_membership_differs", "sum")
).reset_index()
a_q_disagreement.to_csv(a_table_dir / "q1_qualifying_grid_disagreement.csv", index=False)

a_q1_changes = a_q1_rates.pivot(
    index=["circuitId", "circuit_name", "definition"], columns="era",
    values=["rate", "starts", "rank"]
)
a_q1_changes.columns = [f"{a_measure}_{a_era}" for a_measure, a_era in a_q1_changes.columns]
a_q1_changes = a_q1_changes.reset_index()
a_q1_changes["supported_both"] = (
    a_q1_changes["starts_1994–2013"].ge(20) & a_q1_changes["starts_2014–2024"].ge(20)
)
a_q1_changes["rate_change_pp"] = 100 * (
    a_q1_changes["rate_2014–2024"] - a_q1_changes["rate_1994–2013"]
)
# Ranks here compare the SAME set of circuits in both eras, avoiding calendar entry/exit effects.
a_q1_changes["common_rank_before"] = np.nan
a_q1_changes["common_rank_after"] = np.nan
a_q1_changes.loc[a_q1_changes.supported_both, "common_rank_before"] = a_q1_changes[
    a_q1_changes.supported_both
].groupby("definition")["rate_1994–2013"].rank(method="min", ascending=False)
a_q1_changes.loc[a_q1_changes.supported_both, "common_rank_after"] = a_q1_changes[
    a_q1_changes.supported_both
].groupby("definition")["rate_2014–2024"].rank(method="min", ascending=False)
a_q1_changes["common_rank_change"] = a_q1_changes.common_rank_after - a_q1_changes.common_rank_before
a_q1_changes.to_csv(a_table_dir / "q1_era_changes.csv", index=False)

a_q1_fig, a_q1_axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
for a_ax, a_era in zip(a_q1_axes, ["1994–2013", "2014–2024"]):
    a_grid_cells = a_q1_rates[(a_q1_rates.era == a_era) & (a_q1_rates.definition == "Grid 1–2") & a_q1_rates.supported]
    a_display_cells = a_grid_cells.nlargest(12, "starts").sort_values("rate")
    a_names = a_display_cells.circuit_name.tolist()
    for a_shift, a_def, a_color in [(-0.13, "Grid 1–2", "#236b8e"), (0.13, "Qualifying 1–2", "#d76736")]:
        a_plot = a_q1_rates[(a_q1_rates.era == a_era) & (a_q1_rates.definition == a_def)].set_index("circuit_name").reindex(a_names)
        a_y = np.arange(len(a_names)) + a_shift
        a_ax.errorbar(100 * a_plot.rate, a_y,
                      xerr=100 * np.vstack([a_plot.rate - a_plot.ci_low, a_plot.ci_high - a_plot.rate]),
                      fmt="o", color=a_color, capsize=3, label=a_def, markersize=5)
    a_ax.set_yticks(np.arange(len(a_names)), [f"{a_n} (n={int(a_v)})" for a_n, a_v in zip(a_names, a_display_cells.starts)])
    a_ax.set_title(a_era + f" · {len(a_names)} best-supported circuits")
    a_ax.set_xlabel("Front-row podium conversion (%) · 95% Wilson interval")
    a_ax.set_xlim(0, 105)
    a_ax.grid(axis="x", alpha=0.2)
a_q1_axes[0].legend(loc="lower right")
a_q1_fig.suptitle("Qualifying front row and actual starting front row answer different questions", fontsize=13)
plt.tight_layout()
savefig("q1_frontrow_circuits.png", "Front-row podium conversion by circuit and era on the same eligible races, excluding recorded nonstarters; qualifying and grid definitions use two starters per race. Bars are descriptive 95% Wilson intervals and plotted circuits have at least 20 starts per cell.")
analysis_figures.append("q1_frontrow_circuits.png")

a_q1_delta = a_q1_changes[a_q1_changes.supported_both & a_q1_changes.definition.eq("Grid 1–2")].sort_values("rate_change_pp")
plt.figure(figsize=(10, max(4, 0.35 * len(a_q1_delta))))
plt.barh(a_q1_delta.circuit_name, a_q1_delta.rate_change_pp,
         color=np.where(a_q1_delta.rate_change_pp >= 0, "#236b8e", "#d76736"))
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Grid-front-row conversion change: 2014–2024 minus 1994–2013 (percentage points)")
plt.title("Same circuits in both eras · minimum 20 front-row driver starts per era")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()
savefig("q1_era_changes.png", "Change in actual-grid front-row podium conversion from 1994–2013 to 2014–2024 for the same circuit identities, with at least 20 observed starts in each era. Calendar changes, cars and field composition prevent causal attribution to hybrid rules.")
analysis_figures.append("q1_era_changes.png")
display(a_q1_rates[a_q1_rates.supported].sort_values(["era", "definition", "rank", "starts"], ascending=[True, True, True, False]).groupby(["era", "definition"], observed=True).head(5))
display(a_q1_changes[a_q1_changes.supported_both].sort_values("rate_change_pp", ascending=False))

a_q1_leaders = []
for (a_era, a_def), a_block in a_q1_rates[a_q1_rates.supported].groupby(["era", "definition"], observed=True):
    a_best = a_block.sort_values(["rate", "starts"], ascending=False).iloc[0]
    a_q1_leaders.append(f"{a_era} {a_def}: {a_best.circuit_name}, {a_best.rate:.1%} ({int(a_best.podiums)}/{int(a_best.starts)})")
a_q1_largest = a_q1_delta.loc[a_q1_delta.rate_change_pp.abs().idxmax()] if len(a_q1_delta) else None
a_q1_change_text = (
    f"The largest supported absolute grid-conversion change is {a_q1_largest.circuit_name}: "
    f"{a_q1_largest.rate_change_pp:+.1f} percentage points; its common-circuit rank changed "
    f"from {a_q1_largest.common_rank_before:.0f} to {a_q1_largest.common_rank_after:.0f}. "
    if a_q1_largest is not None else "No circuit has enough support in both eras to rank its change. "
)
analyses_summary["q1"] = (
    f"Compared {len(a_q_eligible):,} races from 1994–2024 with both front-row definitions observed; "
    f"excluded {a_q_join.raceId.nunique() - len(a_q_eligible):,} races with incomplete front-row support. "
    f"Recorded nonstarters are removed before eligibility is checked ({int(a_q_nonstart.sum())} driver entries in this window). "
    + "; ".join(a_q1_leaders) + ". " + a_q1_change_text +
    f"Front-row membership differs for {int(a_q_disagreement.differing_memberships.sum()):,} driver-race "
    "records, so qualifying and grid must not be conflated. Ranks are descriptive, with ties and wide "
    "intervals possible; changes cannot be attributed causally to hybrid regulations."
)
display(Markdown("**Answer.** " + analyses_summary["q1"]))

# %% [markdown]
# ### Q2. Is home-country podium probability higher after controlling for starting position?
#
# A driver's recorded nationality is mapped explicitly to one or more countries;
# this is a nationality proxy, not residence, birthplace or self-reported home.
# `British → UK`, `American → USA`, and both `USA` and `United States` circuit
# labels are normalized. Dual nationalities match either country. East German is
# mapped geographically to Germany; Rhodesian to Zimbabwe, where this dataset has
# no circuit. Missing/unmapped values are excluded and counted rather than being
# treated as away. The complete mapping is exported for review.
#
# Primary scope is 1994–2024, matching Q1's modern coverage. We directly
# standardize home and away rates to the **same observed distribution of exact
# starting grid position × era**. Grid zero is a separate pit-lane/unrecorded
# category; it is not treated as first place. Each retained stratum needs at least
# five home and five away rows. This common-support restriction and its retained
# fraction are reported. This question's unit is a pre-race driver entry, including
# recorded nonstarters as non-podiums; it is not a mechanical-start denominator.
# 1,000 bootstrap resamples draw entire races, preserving
# dependence between competitors; weights and the selected common-support target
# remain fixed. This controls observed grid and broad era only. Driver quality,
# constructor, circuit and detailed season differences remain confounders, so the
# result is an association, not a causal home advantage.

# %%
a_nationality_map = {
    "American": ("USA",), "American-Italian": ("USA", "Italy"),
    "Argentine": ("Argentina",), "Argentinian": ("Argentina",),
    "Argentine-Italian": ("Argentina", "Italy"), "Australian": ("Australia",),
    "Austrian": ("Austria",), "Belgian": ("Belgium",), "Brazilian": ("Brazil",),
    "British": ("UK",), "Canadian": ("Canada",), "Chilean": ("Chile",),
    "Chinese": ("China",), "Colombian": ("Colombia",), "Czech": ("Czech Republic",),
    "Danish": ("Denmark",), "Dutch": ("Netherlands",), "East German": ("Germany",),
    "Finnish": ("Finland",), "French": ("France",), "German": ("Germany",),
    "Hungarian": ("Hungary",), "Indian": ("India",), "Indonesian": ("Indonesia",),
    "Irish": ("Ireland",), "Italian": ("Italy",), "Japanese": ("Japan",),
    "Liechtensteiner": ("Liechtenstein",), "Malaysian": ("Malaysia",),
    "Mexican": ("Mexico",), "Monegasque": ("Monaco",), "New Zealander": ("New Zealand",),
    "Polish": ("Poland",), "Portuguese": ("Portugal",), "Rhodesian": ("Zimbabwe",),
    "Russian": ("Russia",), "South African": ("South Africa",), "Spanish": ("Spain",),
    "Swedish": ("Sweden",), "Swiss": ("Switzerland",), "Thai": ("Thailand",),
    "Uruguayan": ("Uruguay",), "Venezuelan": ("Venezuela",)
}
pd.DataFrame([
    {"nationality": a_nat, "mapped_countries": " | ".join(a_countries)}
    for a_nat, a_countries in sorted(a_nationality_map.items())
]).to_csv(a_table_dir / "q2_nationality_country_mapping.csv", index=False)
a_home = a_joined[a_joined.year.between(1994, 2024)].copy()
a_home["country_normalized"] = a_home.circuit_country.astype("string").str.strip().replace({"United States": "USA"})
a_home["nationality_normalized"] = a_home.nationality.astype("string").str.strip()
a_home["home_countries"] = a_home.nationality_normalized.map(a_nationality_map)
a_home_missing = int((a_home.home_countries.isna() | a_home.country_normalized.isna() | a_home.grid.isna()).sum())
a_home = a_home[a_home.home_countries.notna() & a_home.country_normalized.notna() & a_home.grid.notna()].copy()
a_home["home"] = [a_country in a_nations for a_country, a_nations in zip(a_home.country_normalized, a_home.home_countries)]
a_home["era"] = np.where(a_home.year < 2014, "1994–2013", "2014–2024")
a_home["stratum"] = a_home.grid.astype(int).astype(str) + " | " + a_home.era
a_home_strata = a_rates(a_home, ["stratum", "grid", "era", "home"])
a_home_support = a_home_strata.pivot(index="stratum", columns="home", values="starts").reindex(columns=[False, True]).fillna(0)
a_home_shared = a_home_support.index[a_home_support.min(axis=1).ge(5)].tolist()
assert a_home_shared, "Home analysis has no common grid/era support."
a_home_kept = a_home[a_home.stratum.isin(a_home_shared)].copy()
a_home_weights = a_home_kept.stratum.value_counts().reindex(a_home_shared) / len(a_home_kept)
a_home_strata["shared_support"] = a_home_strata.stratum.isin(a_home_shared)
a_home_strata["target_weight"] = a_home_strata.stratum.map(a_home_weights)
a_home_strata.rename(columns={"starts": "driver_entries"}).to_csv(a_table_dir / "q2_grid_strata.csv", index=False)

# Lab 1-style pivot table: one row per grid/era stratum and a column for each
# home flag. Multiply each stratum's observed rate by the SAME target weight.
# Direct standardization goes beyond the lab: it balances these observed grid/era
# distributions without asserting that home country causes a different outcome.
a_home_rate_pivot = a_home_kept.pivot_table(
    index="stratum", columns="home", values="podium", aggfunc="mean", observed=True
).reindex(index=a_home_shared, columns=[False, True])
a_home_weighted_rates = a_home_rate_pivot.mul(a_home_weights, axis="index")
a_home_std = a_home_weighted_rates.sum(axis=0).to_numpy()
display(a_home_rate_pivot.rename(columns={False: "Away rate", True: "Home rate"}))
a_home_rate_pivot.rename(columns={False: "away_rate", True: "home_rate"}).assign(
    target_weight=a_home_weights
).to_csv(a_table_dir / "q2_standardization_pivot.csv")

def a_whole_race_bootstrap(frame, target_weights, rng, draws=1000):
    """Resample complete races; recompute stratum rates with fixed target weights.

    Each sampled race can appear multiple times. Mapping its multiplicity onto
    race-level counts has the same result as copying every driver row that many
    times, while keeping the intermediate Pandas table small and readable.
    """
    race_counts = frame.groupby(["raceId", "stratum", "home"], observed=True).agg(
        podiums=("podium", "sum"), entries=("podium", "size")
    ).reset_index()
    race_ids = frame.raceId.unique()
    estimates, valid_draws = [], []
    for draw in range(draws):
        sampled_races = rng.choice(race_ids, size=len(race_ids), replace=True)
        multiplicities = pd.Series(sampled_races).value_counts()
        draw_counts = race_counts.copy()
        draw_counts["multiplicity"] = draw_counts.raceId.map(multiplicities).fillna(0)
        draw_counts["podiums"] = draw_counts.podiums * draw_counts.multiplicity
        draw_counts["entries"] = draw_counts.entries * draw_counts.multiplicity
        stratum_counts = draw_counts.groupby(["stratum", "home"], observed=True)[
            ["podiums", "entries"]
        ].sum()
        stratum_counts["rate"] = stratum_counts.podiums / stratum_counts.entries
        draw_pivot = stratum_counts.reset_index().pivot(
            index="stratum", columns="home", values="rate"
        ).reindex(index=target_weights.index, columns=[False, True])
        valid = bool(draw_pivot.notna().all().all())
        valid_draws.append(valid)
        if valid:
            estimates.append(draw_pivot.mul(target_weights, axis="index").sum(axis=0).to_numpy())
    return np.asarray(estimates), np.asarray(valid_draws)

a_home_rng = np.random.default_rng(SEED)
a_home_boot_std, a_home_boot_valid = a_whole_race_bootstrap(
    a_home_kept, a_home_weights, a_home_rng
)
assert len(a_home_boot_std) >= 500, "Insufficient stable bootstrap draws; increase stratum support."
a_home_std_ci = np.quantile(a_home_boot_std, [0.025, 0.975], axis=0)
a_home_difference = a_home_std[1] - a_home_std[0]
a_home_diff_ci = np.quantile(a_home_boot_std[:, 1] - a_home_boot_std[:, 0], [0.025, 0.975])

# Crude rates use the full mapped window. Race clustering is also used for their CIs.
a_home_raw_rates = a_home.groupby("home").podium.agg(["sum", "size"]).reindex([False, True])
a_home_raw = (a_home_raw_rates["sum"] / a_home_raw_rates["size"]).to_numpy()
a_home_raw_boot, a_home_raw_valid = a_whole_race_bootstrap(
    a_home.assign(stratum="All mapped entries"),
    pd.Series({"All mapped entries": 1.0}), a_home_rng
)
assert a_home_raw_valid.all(), "A crude resample unexpectedly lost a home/away group."
a_home_raw_ci = np.quantile(a_home_raw_boot, [0.025, 0.975], axis=0)
pd.DataFrame({
    "away_rate": a_home_boot_std[:, 0], "home_rate": a_home_boot_std[:, 1],
    "home_minus_away_pp": 100 * (a_home_boot_std[:, 1] - a_home_boot_std[:, 0])
}).to_csv(a_table_dir / "q2_race_bootstrap_draws.csv", index=False)
a_home_report = pd.DataFrame([
    {"estimate": a_label, "group": a_group, "rate": a_values[a_i],
     "ci_low": a_ci[0, a_i], "ci_high": a_ci[1, a_i],
     "rows": int(a_rows[a_i])}
    for a_label, a_values, a_ci, a_rows in [
        ("Crude: all mapped rows", a_home_raw, a_home_raw_ci, a_home_raw_rates["size"].to_numpy()),
        ("Standardized: shared grid/era support", a_home_std, a_home_std_ci, a_home_kept.groupby("home").size().reindex([False, True]).to_numpy())
    ] for a_i, a_group in enumerate(["Away", "Home"])
])
a_home_report["adjusted_home_minus_away_pp"] = 100 * a_home_difference
a_home_report["adjusted_difference_ci_low_pp"] = 100 * a_home_diff_ci[0]
a_home_report["adjusted_difference_ci_high_pp"] = 100 * a_home_diff_ci[1]
a_home_report.to_csv(a_table_dir / "q2_home_advantage.csv", index=False)
display(a_home_report)

plt.figure(figsize=(9, 5))
for a_i, (a_label, a_values, a_ci, a_color) in enumerate([
    ("Crude", a_home_raw, a_home_raw_ci, "#8a949e"),
    ("Same grid × era distribution", a_home_std, a_home_std_ci, "#236b8e")
]):
    a_positions = np.arange(2) + (-0.10 if a_i == 0 else 0.10)
    plt.errorbar(a_positions, 100 * a_values,
                 yerr=100 * np.vstack([a_values - a_ci[0], a_ci[1] - a_values]),
                 fmt="o", color=a_color, label=a_label, capsize=5, markersize=8)
plt.xticks([0, 1], ["Away-country races", "Home-country races"])
plt.ylabel("Podium probability (%) · 95% race-bootstrap interval")
plt.title(f"Home association after exact-grid and era standardization\nAdjusted home − away: {100*a_home_difference:+.2f} pp [{100*a_home_diff_ci[0]:+.2f}, {100*a_home_diff_ci[1]:+.2f}]")
plt.legend()
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
savefig("q2_home_advantage.png", "Home-country and away-country driver-entry podium probabilities in 1994–2024, before and after standardization to the same exact grid-position × era distribution. Intervals use 1,000 whole-race bootstrap resamples; adjusted estimates apply only to shared-support strata, and are associational.")
analysis_figures.append("q2_home_advantage.png")
a_home_interpretation = (
    "The adjusted interval excludes zero, supporting an observed association in this target population."
    if a_home_diff_ci[0] > 0 or a_home_diff_ci[1] < 0
    else "The adjusted interval includes zero; these data do not establish a clear home-country association after this adjustment."
)
analyses_summary["q2"] = (
    f"In 1994–2024, crude home podium rate is {a_home_raw[1]:.1%} versus {a_home_raw[0]:.1%} away. "
    f"After standardizing exact grid × era, home is {a_home_std[1]:.1%} versus {a_home_std[0]:.1%} away: "
    f"difference {100*a_home_difference:+.2f} percentage points (95% race-bootstrap interval "
    f"{100*a_home_diff_ci[0]:+.2f} to {100*a_home_diff_ci[1]:+.2f}). "
    f"Shared support retains {len(a_home_kept):,}/{len(a_home):,} rows ({len(a_home_kept)/len(a_home):.1%}) "
    f"across {len(a_home_shared)} strata; {a_home_missing} rows are excluded for unavailable mapping/country/grid. "
    "The unit is a driver entry; recorded nonstarters remain non-podiums for this pre-race association. "
    f"{int(a_home_boot_valid.sum())}/1,000 adjusted bootstrap draws retained all strata. "
    + a_home_interpretation + " Nationality is a proxy for home, and unmeasured driver/team/circuit differences prevent causal interpretation."
)
display(Markdown("**Answer.** " + analyses_summary["q2"]))

# %% [markdown]
# ### Q3. Which constructors have the highest mechanical retirement rates in 2014–2021 versus 2022–2024?
#
# The numerator is a recorded mechanical retirement cause; the denominator is
# driver-race **starts**, including finishers and other retirement causes. Excluding
# collisions only from the numerator is necessary: dropping them from the
# denominator would inflate a constructor's mechanical rate.
# A cause label alone is insufficient: both strict and broad definitions also
# require recorded laps to be fewer than the race winner's laps, as an explicit
# early-end proxy. A driver classified in the results can still have retired early
# and satisfy that rule. A mechanically labelled full-distance result is flagged
# and kept in the start denominator, but excluded from either retirement numerator.
# Winner distance is a retrospective analysis field, never a predictive feature;
# it cannot perfectly distinguish every lapped finish from a retirement, so the
# component-cause label and distance condition must both be present.
#
# The strict map includes explicit powertrain, hydraulic/electrical, cooling,
# brake, suspension, drivetrain and generic mechanical/technical failures. A broad
# sensitivity adds ambiguous tyre/wheel/bodywork/fuel/seat and unspecified
# `Retired` labels, whose causes may overlap impact damage or operational errors.
# Explicit accidents, collision/damage/debris, driver health, and regulatory
# disqualification causes are never counted as mechanical. Exact maps and observed
# frequencies are exported; no status is selected by a vague substring search.
#
# `positionText=W` (withdrawal), `F` (failed to qualify), and explicit
# withdrawal/nonqualifying labels exclude an entry only without positive recorded
# laps. Positive laps override those labels: a mid-race withdrawal is still a start.
# The shared rule and contradictory records are audited in section 1.
# Grid zero alone is not DNS, and zero laps alone can be a lap-one
# collision. A second sensitivity drops remaining zero-lap mechanical/ambiguous
# rows, because the data do not always distinguish a formation-lap DNS from an
# immediate retirement. Constructor IDs are kept separate across renames rather
# than guessing lineage. Ranking requires at least 40 starts (about one full
# two-car season), and ranks remain descriptive rather than statistically proven.
# `positionText=N` is an unclassified result; these drivers did start and remain
# in the denominator. Whether a result was classified is separate from its cause.

# %%
a_mechanical_strict = {
    "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical",
    "Radiator", "Suspension", "Brakes", "Differential", "Overheating", "Mechanical",
    "Driveshaft", "Fuel pressure", "Water pressure", "Throttle", "Steering",
    "Technical", "Electronics", "Heat shield fire", "Exhaust", "Oil leak",
    "Water leak", "Fuel pump", "Track rod", "Oil pressure", "Engine fire",
    "Engine misfire", "Pneumatics", "Wheel bearing", "Fuel system", "Oil line",
    "Launch control", "Power loss", "Drivetrain", "Ignition", "Battery",
    "Halfshaft", "Crankshaft", "Alternator", "Oil pump", "Fuel leak", "Injection",
    "Distributor", "Turbo", "CV joint", "Water pump", "Spark plugs", "Fuel pipe",
    "Oil pipe", "Axle", "Water pipe", "Magneto", "Supercharger", "Power Unit",
    "ERS", "Cooling system"
}
a_mechanical_ambiguous = {
    "Tyre", "Puncture", "Tyre puncture", "Wheel", "Wheel rim", "Wheel nut",
    "Front wing", "Broken wing", "Rear wing", "Undertray", "Chassis", "Brake duct",
    "Refuelling", "Fuel rig", "Fuel", "Out of fuel", "Fire", "Vibrations", "Handling",
    "Driver Seat", "Seat", "Safety belt", "Retired"
}
a_nonstart_statuses = START_EXCLUSION_STATUSES
a_crash_statuses = {"Accident", "Collision", "Spun off", "Fatal accident", "Collision damage", "Damage", "Debris"}
a_driver_statuses = {"Physical", "Injured", "Injury", "Driver unwell", "Eye injury", "Illness"}
a_regulatory_statuses = {"Disqualified", "Underweight", "Excluded"}

def a_status_category(a_status):
    if a_status in a_mechanical_strict:
        return "Mechanical: strict"
    if a_status in a_mechanical_ambiguous:
        return "Ambiguous: broad sensitivity only"
    if a_status == "Withdrew":
        return "Withdrawal: inspect laps for start evidence"
    if a_status in a_nonstart_statuses:
        return "Nonstart / nonqualifying"
    if a_status in a_crash_statuses:
        return "Crash / damage: never mechanical"
    if a_status in a_driver_statuses:
        return "Driver health: never mechanical"
    if a_status in a_regulatory_statuses:
        return "Regulatory: never mechanical"
    if a_status == "Finished" or str(a_status).startswith("+"):
        return "Finished / lapped"
    return "Other / unresolved: excluded from numerator"

a_reliability_all = a_joined[a_joined.year.between(2014, 2024)].copy()
a_reliability_all["era"] = np.where(a_reliability_all.year <= 2021, "2014–2021", "2022–2024")
a_reliability_all["cause_category"] = a_reliability_all.race_status.map(a_status_category)
a_reliability_winners = a_reliability_all[a_reliability_all.positionOrder.eq(1)][["raceId", "laps"]].rename(
    columns={"laps": "winner_laps"}
)
assert not a_reliability_winners.raceId.duplicated().any(), "Each analyzed race needs one winner distance."
a_reliability_all = a_reliability_all.merge(a_reliability_winners, on="raceId", how="left", validate="many_to_one")
assert a_reliability_all.winner_laps.notna().all() and a_reliability_all.winner_laps.gt(0).all()
assert a_reliability_all.laps.notna().all(), "Retirement distance must be known."
a_reliability_all["ended_early_proxy"] = a_reliability_all.laps.lt(a_reliability_all.winner_laps)
a_reliability_all["full_distance_component_status"] = (
    a_reliability_all.race_status.isin(a_mechanical_strict | a_mechanical_ambiguous)
    & ~a_reliability_all.ended_early_proxy
)
a_reliability_all["nonstarter"] = ~classify_race_starts(a_reliability_all, "race_status")
assert not a_reliability_all.loc[a_reliability_all.nonstarter, "laps"].gt(0).any()
a_status_mapping = a_statuses.copy()
a_status_mapping["cause_category"] = a_status_mapping.race_status.map(a_status_category)
a_status_mapping["rows_2014_2024"] = a_status_mapping.race_status.map(a_reliability_all.race_status.value_counts()).fillna(0).astype(int)
a_status_mapping["nonstarters_2014_2024"] = a_status_mapping.race_status.map(
    a_reliability_all[a_reliability_all.nonstarter].race_status.value_counts()
).fillna(0).astype(int)
a_status_mapping["full_distance_component_status_rows"] = a_status_mapping.race_status.map(
    a_reliability_all[a_reliability_all.full_distance_component_status].race_status.value_counts()
).fillna(0).astype(int)
a_status_mapping.to_csv(a_table_dir / "q3_status_mapping.csv", index=False)
a_reliability_all[a_reliability_all.full_distance_component_status][[
    "raceId", "year", "driverId", "constructorId", "constructor_name", "laps",
    "winner_laps", "positionOrder", "positionText", "race_status", "cause_category"
]].to_csv(a_table_dir / "q3_full_distance_component_status.csv", index=False)
a_reliability_all[a_reliability_all.nonstarter][[
    "raceId", "year", "driverId", "constructorId", "constructor_name", "grid",
    "laps", "positionText", "race_status"
]].to_csv(a_table_dir / "q3_excluded_nonstarters.csv", index=False)
a_reliability = a_reliability_all[~a_reliability_all.nonstarter].copy()
a_reliability["mechanical"] = (
    a_reliability.race_status.isin(a_mechanical_strict) & a_reliability.ended_early_proxy
).astype(int)
a_reliability["mechanical_broad"] = (
    a_reliability.race_status.isin(a_mechanical_strict | a_mechanical_ambiguous) & a_reliability.ended_early_proxy
).astype(int)
assert a_reliability.loc[a_reliability.mechanical.eq(1), "laps"].lt(
    a_reliability.loc[a_reliability.mechanical.eq(1), "winner_laps"]
).all()
a_reliability["uncertain_zero_lap_start"] = (
    a_reliability.laps.eq(0) & a_reliability.mechanical_broad.eq(1)
)
a_q3_rates = a_reliability.groupby(["constructorId", "constructor_name", "era"], observed=True).agg(
    starts=("driverId", "size"), mechanical=("mechanical", "sum"),
    mechanical_broad=("mechanical_broad", "sum"),
    uncertain_zero_lap_starts=("uncertain_zero_lap_start", "sum"),
    full_distance_component_status_rows=("full_distance_component_status", "sum")
).reset_index()
a_q3_rates["rate"] = a_q3_rates.mechanical / a_q3_rates.starts
a_q3_rates["broad_rate"] = a_q3_rates.mechanical_broad / a_q3_rates.starts
a_q3_rates["ci_low"], a_q3_rates["ci_high"] = a_wilson(a_q3_rates.mechanical, a_q3_rates.starts)
a_q3_rates["supported"] = a_q3_rates.starts.ge(40)
a_q3_rates["rank"] = np.nan
a_q3_rates["broad_rank"] = np.nan
a_q3_rates.loc[a_q3_rates.supported, "rank"] = a_q3_rates[a_q3_rates.supported].groupby("era")["rate"].rank(method="min", ascending=False)
a_q3_rates.loc[a_q3_rates.supported, "broad_rank"] = a_q3_rates[a_q3_rates.supported].groupby("era")["broad_rate"].rank(method="min", ascending=False)
a_q3_zero = a_reliability[~a_reliability.uncertain_zero_lap_start].groupby(["constructorId", "era"], observed=True).mechanical.agg(["sum", "size"])
a_q3_zero["zero_lap_excluded_rate"] = a_q3_zero["sum"] / a_q3_zero["size"]
a_q3_rates = a_q3_rates.merge(a_q3_zero[["zero_lap_excluded_rate"]], on=["constructorId", "era"], how="left", validate="one_to_one")
a_q3_rates.to_csv(a_table_dir / "q3_constructor_reliability.csv", index=False)
display(a_status_mapping[a_status_mapping.rows_2014_2024.gt(0)].sort_values(["cause_category", "rows_2014_2024"], ascending=[True, False]))
display(a_q3_rates.sort_values(["era", "rate"], ascending=[True, False]))

a_q3_fig, a_q3_axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
for a_ax, a_era in zip(a_q3_axes, ["2014–2021", "2022–2024"]):
    a_plot = a_q3_rates[(a_q3_rates.era == a_era) & a_q3_rates.supported].sort_values("rate")
    a_y = np.arange(len(a_plot))
    a_ax.errorbar(100*a_plot.rate, a_y,
                  xerr=100*np.vstack([a_plot.rate-a_plot.ci_low, a_plot.ci_high-a_plot.rate]),
                  fmt="o", color="#236b8e", capsize=3, label="Strict mechanical", markersize=6)
    a_ax.scatter(100*a_plot.broad_rate, a_y + 0.12, marker="x", color="#d76736", label="Broad status sensitivity", s=35)
    a_ax.set_yticks(a_y, [f"{a_name} (n={a_n})" for a_name, a_n in zip(a_plot.constructor_name, a_plot.starts)])
    a_ax.set_title(a_era)
    a_ax.set_xlabel("Mechanical retirements per driver start (%)")
    a_ax.grid(axis="x", alpha=0.2)
    a_ax.set_xlim(left=0)
a_q3_axes[0].legend(loc="lower right")
a_q3_fig.suptitle("Constructor mechanical retirement rates · strict Wilson intervals and broad sensitivity", fontsize=13)
plt.tight_layout()
savefig("q3_constructor_reliability.png", "Recorded mechanical causes with laps below winner distance, divided by driver starts by constructor ID and era. Positive-lap withdrawals remain starts; explicit withdrawal/nonqualifying labels exclude entries only without positive laps. Bars show descriptive 95% Wilson intervals; orange crosses add ambiguous causes under the same early-end gate. Ranking requires at least 40 starts and does not merge renamed teams.")
analysis_figures.append("q3_constructor_reliability.png")
a_q3_leaders = []
for a_era, a_block in a_q3_rates[a_q3_rates.supported].groupby("era", observed=True):
    a_strict_leader = a_block.sort_values(["rate", "starts"], ascending=False).iloc[0]
    a_broad_leader = a_block.sort_values(["broad_rate", "starts"], ascending=False).iloc[0]
    a_q3_leaders.append(
        f"{a_era}: {a_strict_leader.constructor_name} has the highest supported strict rate, "
        f"{a_strict_leader.rate:.1%} ({int(a_strict_leader.mechanical)}/{int(a_strict_leader.starts)}), "
        f"95% Wilson interval {a_strict_leader.ci_low:.1%}–{a_strict_leader.ci_high:.1%}; "
        f"the broad-map leader is {a_broad_leader.constructor_name} ({a_broad_leader.broad_rate:.1%})"
    )
analyses_summary["q3"] = (
    "; ".join(a_q3_leaders) + ". "
    f"Excluded {int(a_reliability_all.nonstarter.sum())} recorded nonstarters from both numerator and denominator. "
    "Withdrawal/nonqualifying labels exclude only entries without positive laps; positive-lap withdrawals remain starts. "
    f"Both cause maps require laps below winner distance as an early-end proxy; "
    f"{int(a_reliability.full_distance_component_status.sum())} full-distance component-status row(s) are "
    "flagged and excluded from both retirement numerators, while remaining in the start denominator. "
    f"{int(a_reliability.uncertain_zero_lap_start.sum())} remaining zero-lap mechanical/ambiguous rows have "
    "uncertain start status; the exported table reports a sensitivity excluding these. "
    "The numerator excludes explicit crash/damage, driver-health and regulatory outcomes while these started "
    "entries remain in the denominator. Rankings use at least 40 starts and recorded constructor IDs, "
    "so a rename does not automatically inherit its predecessor's rate. Wilson intervals and unequal exposure "
    "mean that the largest observed rate is not necessarily evidence of the least reliable constructor."
)
display(Markdown("**Answer.** " + analyses_summary["q3"]))
(OUT / 'analyses_summary.json').write_text(json.dumps(analyses_summary, indent=2), encoding='utf-8')

# %% [markdown]
# **What these joins establish.** Each output is computed at a declared grain and
# uses a denominator tied to the question. Matching qualifying coverage prevents
# Q1 from comparing different historical samples; standardization stops Q2 from
# simply reflecting home drivers' grid distribution; explicit status and withdrawal
# handling prevents Q3 from treating crashes or mechanically caused DNS as race
# retirements. Exported tables retain small cells and ambiguous cases so the
# conclusions can be audited instead of hiding them in a chart.
