"""Risk tier assignment and the economics layered on top of it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tiering import (
    assign_tiers,
    break_even_avoidable_fraction,
    business_case,
    sensitivity_grid,
    tier_profile,
)


@pytest.fixture
def tiered(scored_holdout) -> pd.DataFrame:
    frame = scored_holdout.copy()
    frame["risk_tier"] = assign_tiers(frame["expected_cost"], [0.80, 0.95])
    return frame


def test_tier_sizes_follow_the_configured_cutoffs(tiered):
    shares = tiered["risk_tier"].value_counts(normalize=True)
    assert shares["low"] == pytest.approx(0.80, abs=0.01)
    assert shares["medium"] == pytest.approx(0.15, abs=0.01)
    assert shares["high"] == pytest.approx(0.05, abs=0.01)


def test_higher_tiers_carry_higher_realised_cost(tiered):
    profile = tier_profile(tiered, "risk_tier", "target_cost", "expected_cost").set_index("risk_tier")
    assert profile.loc["high", "mean_actual_cost"] > profile.loc["medium", "mean_actual_cost"]
    assert profile.loc["medium", "mean_actual_cost"] > profile.loc["low", "mean_actual_cost"]


def test_tier_shares_of_members_and_spend_sum_to_one_hundred(tiered):
    profile = tier_profile(tiered, "risk_tier", "target_cost", "expected_cost")
    assert profile["pct_of_members"].sum() == pytest.approx(100.0, abs=0.01)
    assert profile["pct_of_total_spend"].sum() == pytest.approx(100.0, abs=0.01)


def test_business_case_arithmetic_is_internally_consistent(tiered):
    case = business_case(tiered, "risk_tier", "target_cost", "high", 500.0, 0.10, 0.35)
    row = case[case["scenario"] == "targeted_high_tier"].iloc[0]
    assert row["outreach_spend"] == pytest.approx(row["members_targeted"] * 500.0)
    assert row["avoided_cost"] == pytest.approx(row["spend_reached"] * 0.35 * 0.10, rel=1e-6)
    assert row["net_benefit"] == pytest.approx(row["avoided_cost"] - row["outreach_spend"], rel=1e-6)


def test_targeting_the_high_tier_beats_a_random_group_of_the_same_size(tiered):
    case = business_case(tiered, "risk_tier", "target_cost", "high", 500.0, 0.10, 0.35)
    targeted = case[case["scenario"] == "targeted_high_tier"].iloc[0]
    random_group = case[case["scenario"] == "random_same_size"].iloc[0]
    assert targeted["members_targeted"] == random_group["members_targeted"]
    assert targeted["spend_reached"] > random_group["spend_reached"]
    assert targeted["net_benefit"] > random_group["net_benefit"]


def test_break_even_fraction_produces_zero_net_benefit(tiered):
    fraction = break_even_avoidable_fraction(tiered, "risk_tier", "target_cost", "high", 500.0, 0.35)
    case = business_case(tiered, "risk_tier", "target_cost", "high", 500.0, fraction, 0.35)
    row = case[case["scenario"] == "targeted_high_tier"].iloc[0]
    assert row["net_benefit"] == pytest.approx(0.0, abs=1.0)


def test_net_benefit_falls_as_outreach_gets_more_expensive(tiered):
    grid = sensitivity_grid(tiered, "risk_tier", "target_cost", "high", [200.0, 500.0, 1000.0], [0.10], 0.35)
    ordered = grid.sort_values("outreach_cost_per_member")
    assert ordered["net_benefit"].is_monotonic_decreasing


def test_net_benefit_rises_as_more_cost_is_avoidable(tiered):
    grid = sensitivity_grid(tiered, "risk_tier", "target_cost", "high", [500.0], [0.02, 0.10, 0.20], 0.35)
    ordered = grid.sort_values("avoidable_cost_fraction")
    assert ordered["net_benefit"].is_monotonic_increasing
