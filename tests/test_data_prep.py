"""Claim date handling, cost definitions and the audit detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit import cost_distribution_profile, longitudinal_signal, year_level_shift
from src.data_io import numbered_columns, year_from_date


def test_year_is_taken_from_the_claim_through_date():
    dates = pd.Series([20080115, 20091231, 20100701])
    assert year_from_date(dates).tolist() == [2008, 2009, 2010]


def test_missing_claim_dates_do_not_become_a_year():
    dates = pd.Series([20080115, np.nan])
    result = year_from_date(dates)
    assert result.tolist()[0] == 2008
    assert pd.isna(result.tolist()[1])


def test_numbered_columns_builds_the_line_slot_names():
    assert numbered_columns("LINE_NCH_PMT_AMT_", 3) == [
        "LINE_NCH_PMT_AMT_1",
        "LINE_NCH_PMT_AMT_2",
        "LINE_NCH_PMT_AMT_3",
    ]


def test_zero_cost_members_are_counted_in_the_distribution(member_year):
    frame = member_year.copy()
    frame.loc[frame.index[:50], "total_cost"] = 0.0
    profile = cost_distribution_profile(frame)
    assert (profile["pct_zero_cost"] > 0).any()
    assert (profile["members"] == frame.groupby("year").size().to_numpy()).all()


def test_top_decile_share_exceeds_a_tenth_for_skewed_spend(member_year):
    profile = cost_distribution_profile(member_year)
    assert (profile["pct_spend_in_top_decile"] > 10.0).all()


def test_signal_detector_finds_a_planted_year_over_year_relationship(member_year):
    signal = longitudinal_signal(member_year, 0.90)
    row = signal.iloc[0]
    assert row["spearman"] > 0.5
    assert row["persistence_lift"] > 1.5


def test_signal_detector_reports_no_lift_when_years_are_independent():
    rng = np.random.default_rng(21)
    members = [f"M{i:04d}" for i in range(2000)]
    rows = []
    for year in (2008, 2009):
        for member in members:
            rows.append({"DESYNPUF_ID": member, "year": year, "total_cost": float(rng.gamma(1.3, 2000))})
    signal = longitudinal_signal(pd.DataFrame(rows), 0.90)
    assert signal.iloc[0]["persistence_lift"] < 1.4


def test_level_shift_reports_the_change_between_years(member_year):
    shift = year_level_shift(member_year)
    assert pd.isna(shift.iloc[0]["change_vs_prior_pct"]) or shift.iloc[0]["change_vs_prior_pct"] is None
    assert shift.iloc[1]["change_vs_prior_pct"] < 0
