"""Data quality and signal audit.

CMS states in the DE-SynPUF user guide that the relationships between variables were
deliberately altered to limit re-identification risk, and that multivariate results
should be treated with caution. That is a claim about the data that can be measured
rather than repeated, so this module quantifies how much year over year signal
actually survived, and records the level shift in the third year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def check_row_counts(config, counts: dict[str, int]) -> dict:
    """Compares observed record counts with the counts CMS publishes for this subsample."""
    expected = config["data"]["expected_counts"]
    rows = []
    for key, published in expected.items():
        observed = counts.get(key)
        rows.append(
            {
                "file": key,
                "published": int(published),
                "observed": None if observed is None else int(observed),
                "matches": None if observed is None else bool(int(observed) == int(published)),
            }
        )
    return {"checks": rows, "all_match": all(row["matches"] for row in rows if row["matches"] is not None)}


def check_chronic_prevalence(config, panel: pd.DataFrame) -> pd.DataFrame:
    """Compares 2008 condition prevalence with the figures published in the user guide."""
    published = config["data"]["published_prevalence"]
    baseline = panel[panel["year"] == 2008]
    rows = []
    for column, expected_pct in published.items():
        observed_pct = float(baseline[column].mean() * 100)
        rows.append(
            {
                "condition": column,
                "published_pct": float(expected_pct),
                "observed_pct": round(observed_pct, 2),
                "abs_difference": round(abs(observed_pct - float(expected_pct)), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def reconcile_costs(member_year: pd.DataFrame) -> pd.DataFrame:
    """Checks claim derived cost against the annual totals carried on the summary file.

    The summary file totals cover the institutional and professional sources only, so
    the comparison is made against those three and not against the drug spend.
    """
    frame = member_year.copy()
    frame["claims_derived"] = frame[["ip_cost", "op_cost", "carrier_cost"]].sum(axis=1)
    rows = []
    for year, group in frame.groupby("year"):
        summary_total = float(group["summary_reimbursement"].sum())
        derived_total = float(group["claims_derived"].sum())
        difference = derived_total - summary_total
        rows.append(
            {
                "year": int(year),
                "summary_file_total": round(summary_total, 2),
                "claims_derived_total": round(derived_total, 2),
                "difference": round(difference, 2),
                "ratio": round(derived_total / summary_total, 4) if summary_total else np.nan,
                "correlation": round(float(group["claims_derived"].corr(group["summary_reimbursement"])), 4),
            }
        )
    return pd.DataFrame(rows)


def cost_distribution_profile(member_year: pd.DataFrame) -> pd.DataFrame:
    """Zero mass, skew and concentration of spend, by year."""
    rows = []
    for year, group in member_year.groupby("year"):
        cost = group["total_cost"]
        positive = cost[cost > 0]
        ordered = cost.sort_values(ascending=False)
        total = cost.sum()
        top_decile_cut = max(int(len(cost) * 0.10), 1)
        top_percent_cut = max(int(len(cost) * 0.01), 1)
        rows.append(
            {
                "year": int(year),
                "members": int(len(cost)),
                "pct_zero_cost": round(float((cost == 0).mean() * 100), 2),
                "mean_cost": round(float(cost.mean()), 2),
                "median_cost": round(float(cost.median()), 2),
                "mean_cost_given_positive": round(float(positive.mean()), 2),
                "p95_cost": round(float(cost.quantile(0.95)), 2),
                "p99_cost": round(float(cost.quantile(0.99)), 2),
                "max_cost": round(float(cost.max()), 2),
                "skew_of_positive": round(float(stats.skew(positive)), 3),
                "pct_spend_in_top_decile": round(float(ordered.head(top_decile_cut).sum() / total * 100), 2),
                "pct_spend_in_top_percent": round(float(ordered.head(top_percent_cut).sum() / total * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def longitudinal_signal(member_year: pd.DataFrame, high_cost_quantile: float = 0.90) -> pd.DataFrame:
    """Measures how much of the year over year cost relationship survived synthesis.

    Reports rank correlation and the persistence of the high cost group, both against
    the random targeting baseline implied by the quantile.
    """
    wide = member_year.pivot_table(index="DESYNPUF_ID", columns="year", values="total_cost")
    years = sorted(int(year) for year in wide.columns)
    rows = []
    for first, second in zip(years, years[1:]):
        paired = wide[[first, second]].dropna()
        base = paired[first]
        follow = paired[second]
        in_top_first = base >= base.quantile(high_cost_quantile)
        in_top_second = follow >= follow.quantile(high_cost_quantile)
        persistence = float(in_top_second[in_top_first].mean())
        baseline_rate = float(in_top_second.mean())
        rows.append(
            {
                "from_year": first,
                "to_year": second,
                "members": int(len(paired)),
                "pearson": round(float(base.corr(follow)), 4),
                "spearman": round(float(base.corr(follow, method="spearman")), 4),
                "high_cost_persistence": round(persistence * 100, 2),
                "random_baseline": round(baseline_rate * 100, 2),
                "persistence_lift": round(persistence / baseline_rate, 2) if baseline_rate else np.nan,
            }
        )
    return pd.DataFrame(rows)


def year_level_shift(member_year: pd.DataFrame) -> pd.DataFrame:
    """Records the change in mean spend between consecutive years."""
    means = member_year.groupby("year")["total_cost"].mean()
    rows = []
    previous_year = None
    for year, value in means.items():
        change = np.nan if previous_year is None else float(value / means[previous_year] - 1) * 100
        rows.append(
            {
                "year": int(year),
                "mean_cost": round(float(value), 2),
                "change_vs_prior_pct": None if previous_year is None else round(change, 2),
            }
        )
        previous_year = year
    return pd.DataFrame(rows)


def emergency_code_coverage(member_year: pd.DataFrame) -> dict:
    """Confirms the emergency visit proxy actually fires in the outpatient file."""
    total = float(member_year["op_emergency_visits"].sum())
    members = int((member_year["op_emergency_visits"] > 0).sum())
    return {
        "total_emergency_visits": total,
        "members_with_any": members,
        "pct_members_with_any": round(members / len(member_year) * 100, 2),
        "proxy_is_populated": bool(total > 0),
    }
