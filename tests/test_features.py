"""Feature construction, leakage and collinearity handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    build_modeling_frame,
    design_matrix,
    feature_columns,
    near_collinear_pairs,
    select_modeling_columns,
)


def test_target_is_taken_from_the_following_year(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    expected = member_year[member_year["year"] == 2009].set_index("DESYNPUF_ID")["total_cost"]
    actual = frame.set_index("DESYNPUF_ID")["target_cost"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_feature_year_rows_only(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    assert set(frame["year"].unique()) == {2008}


def test_no_target_column_leaks_into_the_design_matrix(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    columns = feature_columns(chronic_columns)
    assert "target_cost" not in columns
    assert "target_has_cost" not in columns
    assert not any(column.startswith("target") for column in columns)


def test_counted_conditions_excluded_because_it_is_the_sum_of_the_flags(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    assert (frame["chronic_condition_count"] == frame[chronic_columns].sum(axis=1)).all()
    assert "chronic_condition_count" not in feature_columns(chronic_columns)


def test_design_matrix_is_full_rank_after_selection(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    columns, _ = select_modeling_columns(frame, chronic_columns, 0.99)
    matrix = design_matrix(frame, chronic_columns, columns)
    assert np.linalg.matrix_rank(matrix.to_numpy()) == matrix.shape[1]


def test_near_collinear_pairs_flags_a_duplicated_column():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0], "c": [4.0, 1.0, 3.0, 2.0]})
    pairs = near_collinear_pairs(frame, 0.99)
    assert ("a", "b", 1.0) in [(first, second, round(value, 4)) for first, second, value in pairs]


def test_selection_keeps_the_earlier_column_of_a_duplicated_pair(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    columns, ledger = select_modeling_columns(frame, chronic_columns, 0.99)
    assert "log_rx_fills" in columns
    assert "log_rx_distinct_products" not in columns
    assert "log_rx_distinct_products" in set(ledger["dropped_feature"])


def test_log_features_are_finite_for_zero_cost_members(member_year, chronic_columns):
    frame = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    matrix = design_matrix(frame, chronic_columns)
    assert np.isfinite(matrix.to_numpy()).all()
