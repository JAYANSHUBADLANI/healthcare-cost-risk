"""Risk tiering and the care management business case.

The cost model produces a continuous expected cost. A care management team cannot act
on a continuous number, so it is cut into operational tiers and each tier is costed.

Every economic input here is an assumption, not something the claims data knows. The
outreach cost per member, the share of spend that proactive management can avoid, and
the share of targeted members who actually engage are all stated in the configuration
and swept in a sensitivity grid, because the honest answer to what this programme
returns is a function of those three numbers rather than a single figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_NAMES = ("low", "medium", "high")


def assign_tiers(predicted: pd.Series, cutoffs: list[float]) -> pd.Series:
    """Cuts predicted risk into tiers at the given population percentiles."""
    lower, upper = float(cutoffs[0]), float(cutoffs[1])
    ranks = predicted.rank(method="first", pct=True)
    tiers = pd.Series(TIER_NAMES[0], index=predicted.index, dtype=object)
    tiers[ranks > lower] = TIER_NAMES[1]
    tiers[ranks > upper] = TIER_NAMES[2]
    return pd.Series(
        pd.Categorical(tiers, categories=list(TIER_NAMES), ordered=True), index=predicted.index
    )


def tier_profile(frame: pd.DataFrame, tier_column: str, actual_column: str, predicted_column: str) -> pd.DataFrame:
    """Realised cost, membership and spend share for each tier."""
    total_spend = frame[actual_column].sum()
    grouped = frame.groupby(tier_column, observed=True).agg(
        members=(actual_column, "size"),
        mean_predicted_cost=(predicted_column, "mean"),
        mean_actual_cost=(actual_column, "mean"),
        median_actual_cost=(actual_column, "median"),
        total_actual_cost=(actual_column, "sum"),
    )
    grouped["pct_of_members"] = grouped["members"] / len(frame) * 100
    grouped["pct_of_total_spend"] = grouped["total_actual_cost"] / total_spend * 100
    grouped["cost_concentration_index"] = grouped["pct_of_total_spend"] / grouped["pct_of_members"]
    return grouped.round(4).reset_index()


def business_case(
    frame: pd.DataFrame,
    tier_column: str,
    actual_column: str,
    target_tier: str,
    outreach_cost_per_member: float,
    avoidable_cost_fraction: float,
    engagement_rate: float,
) -> pd.DataFrame:
    """Compares targeting the high tier against a random sample of equal size.

    Savings accrue only on the members who engage, and only on the avoidable share of
    their realised spend. The random comparison uses the population mean spend, which
    is the exact expectation for a randomly chosen group of the same size, so the
    comparison does not depend on the luck of one particular random draw.
    """
    targeted = frame[frame[tier_column] == target_tier]
    size = int(len(targeted))
    population_mean = float(frame[actual_column].mean())

    scenarios = {
        "targeted_high_tier": float(targeted[actual_column].sum()),
        "random_same_size": population_mean * size,
    }

    rows = []
    for name, spend_reached in scenarios.items():
        outreach_spend = size * outreach_cost_per_member
        avoided = spend_reached * engagement_rate * avoidable_cost_fraction
        net = avoided - outreach_spend
        rows.append(
            {
                "scenario": name,
                "members_targeted": size,
                "spend_reached": round(spend_reached, 2),
                "mean_spend_per_targeted_member": round(spend_reached / size, 2) if size else np.nan,
                "outreach_spend": round(outreach_spend, 2),
                "avoided_cost": round(avoided, 2),
                "net_benefit": round(net, 2),
                "return_per_dollar_spent": round(avoided / outreach_spend, 4) if outreach_spend else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    targeted_row = table[table["scenario"] == "targeted_high_tier"].iloc[0]
    random_row = table[table["scenario"] == "random_same_size"].iloc[0]
    table["advantage_over_random"] = round(
        float(targeted_row["net_benefit"] - random_row["net_benefit"]), 2
    )
    return table


def break_even_avoidable_fraction(
    frame: pd.DataFrame,
    tier_column: str,
    actual_column: str,
    target_tier: str,
    outreach_cost_per_member: float,
    engagement_rate: float,
) -> float:
    """The avoidable share of spend at which the programme exactly pays for itself."""
    targeted = frame[frame[tier_column] == target_tier]
    size = len(targeted)
    if size == 0:
        return float("nan")
    spend_reached = float(targeted[actual_column].sum())
    denominator = spend_reached * engagement_rate
    if denominator <= 0:
        return float("nan")
    return float(size * outreach_cost_per_member / denominator)


def sensitivity_grid(
    frame: pd.DataFrame,
    tier_column: str,
    actual_column: str,
    target_tier: str,
    outreach_costs: list[float],
    avoidable_fractions: list[float],
    engagement_rate: float,
) -> pd.DataFrame:
    """Net benefit of targeting the high tier across the assumed economic inputs."""
    rows = []
    for cost in outreach_costs:
        for fraction in avoidable_fractions:
            case = business_case(
                frame,
                tier_column,
                actual_column,
                target_tier,
                cost,
                fraction,
                engagement_rate,
            )
            targeted = case[case["scenario"] == "targeted_high_tier"].iloc[0]
            rows.append(
                {
                    "outreach_cost_per_member": cost,
                    "avoidable_cost_fraction": fraction,
                    "net_benefit": float(targeted["net_benefit"]),
                    "return_per_dollar_spent": float(targeted["return_per_dollar_spent"]),
                }
            )
    return pd.DataFrame(rows)
