"""Evaluation metrics, especially the curves the targeting decision rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calibration_by_decile,
    combined_metrics,
    concentration_curve,
    high_cost_auc,
)

GRID = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]


def test_targeting_everyone_captures_all_spend(scored_holdout):
    curve = concentration_curve(
        scored_holdout["target_cost"], scored_holdout["expected_cost"].to_numpy(), GRID, 0.90
    )
    full = curve[curve["top_k_pct"] == 100.0].iloc[0]
    assert full["pct_of_spend_captured"] == pytest.approx(100.0, abs=0.01)
    assert full["pct_of_high_cost_members_captured"] == pytest.approx(100.0, abs=0.01)


def test_model_beats_random_and_never_beats_perfect_foresight(scored_holdout):
    curve = concentration_curve(
        scored_holdout["target_cost"], scored_holdout["expected_cost"].to_numpy(), GRID, 0.90
    )
    assert (curve["pct_of_spend_captured"] >= curve["pct_of_spend_random"] - 1e-9).all()
    assert (curve["pct_of_spend_captured"] <= curve["pct_of_spend_oracle"] + 1e-9).all()


def test_random_ranking_gives_no_lift():
    rng = np.random.default_rng(3)
    actual = pd.Series(rng.gamma(1.2, 2500, size=6000))
    noise = rng.normal(size=6000)
    curve = concentration_curve(actual, noise, [0.10, 0.20], 0.90)
    assert curve["lift_vs_random"].max() < 1.35


def test_perfect_ranking_reaches_the_oracle_line(scored_holdout):
    actual = scored_holdout["target_cost"]
    curve = concentration_curve(actual, actual.to_numpy(), GRID, 0.90)
    assert curve["pct_of_spend_captured"].to_numpy() == pytest.approx(
        curve["pct_of_spend_oracle"].to_numpy(), abs=0.01
    )


def test_capture_is_monotone_in_targeting_depth(scored_holdout):
    curve = concentration_curve(
        scored_holdout["target_cost"], scored_holdout["expected_cost"].to_numpy(), GRID, 0.90
    ).sort_values("top_k_pct")
    assert curve["pct_of_spend_captured"].is_monotonic_increasing


def test_rescaling_predictions_changes_calibration_but_not_ranking(scored_holdout):
    actual = scored_holdout["target_cost"]
    predicted = scored_holdout["expected_cost"].to_numpy()
    base = combined_metrics(actual, predicted)
    scaled = combined_metrics(actual, predicted * 2.0)
    assert scaled["spearman"] == pytest.approx(base["spearman"])
    assert scaled["predictive_ratio"] == pytest.approx(base["predictive_ratio"] * 2.0, rel=1e-3)


def test_high_cost_flag_size_matches_the_quantile(scored_holdout):
    result = high_cost_auc(scored_holdout["target_cost"], scored_holdout["expected_cost"].to_numpy(), 0.90)
    assert result["members_flagged"] == pytest.approx(len(scored_holdout) * 0.10, rel=0.05)
    assert 0.5 < result["auc"] <= 1.0


def test_calibration_table_covers_every_member(scored_holdout):
    table = calibration_by_decile(scored_holdout["target_cost"], scored_holdout["expected_cost"].to_numpy())
    assert len(table) == 10
    assert table["members"].sum() == len(scored_holdout)
