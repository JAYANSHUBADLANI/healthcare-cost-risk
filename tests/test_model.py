"""The two part structure and the guards that stop it failing silently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_modeling_frame, design_matrix, feature_columns, select_modeling_columns
from src.model import (
    ModelConvergenceError,
    RankDeficientDesignError,
    assert_full_rank,
    coefficient_table,
    fit_two_part,
)


@pytest.fixture
def fitted(member_year, chronic_columns):
    train = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    columns, _ = select_modeling_columns(train, chronic_columns, 0.99)
    return fit_two_part(train, chronic_columns, columns), train, columns


def test_rank_guard_rejects_a_collinear_design():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    with pytest.raises(RankDeficientDesignError):
        assert_full_rank(frame)


def test_rank_guard_accepts_an_independent_design():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 1.0, 3.0, 2.0]})
    assert_full_rank(frame)


def test_including_the_condition_count_would_be_rank_deficient(member_year, chronic_columns):
    train = build_modeling_frame(member_year, 2008, 2009, chronic_columns)
    columns = feature_columns(chronic_columns) + ["chronic_condition_count"]
    matrix = design_matrix(train, chronic_columns, columns)
    with pytest.raises(RankDeficientDesignError):
        assert_full_rank(matrix)


def test_expected_cost_is_the_product_of_the_two_parts(fitted, chronic_columns):
    model, train, _ = fitted
    parts = model.predict_parts(train, chronic_columns)
    product = parts["probability_any_cost"] * parts["conditional_cost"]
    np.testing.assert_allclose(parts["expected_cost"].to_numpy(), product.to_numpy(), rtol=1e-10)


def test_probabilities_stay_in_range_and_conditional_cost_is_positive(fitted, chronic_columns):
    model, train, _ = fitted
    parts = model.predict_parts(train, chronic_columns)
    assert parts["probability_any_cost"].between(0.0, 1.0).all()
    assert (parts["conditional_cost"] > 0).all()


def test_severity_is_fitted_only_on_members_with_cost(fitted):
    model, train, _ = fitted
    assert model.severity.nobs == int((train["target_cost"] > 0).sum())
    assert model.participation.nobs == len(train)


def test_coefficients_are_finite_and_bounded(fitted):
    model, _, _ = fitted
    table = coefficient_table(model)
    assert np.isfinite(table["participation_coef"]).all()
    assert np.isfinite(table["severity_coef"]).all()
    assert table["participation_coef"].abs().max() < 50
    assert table["severity_coef"].abs().max() < 50


def test_prediction_is_stable_when_rows_are_reordered(fitted, chronic_columns):
    model, train, _ = fitted
    shuffled = train.sample(frac=1.0, random_state=7)
    original = model.predict_parts(train, chronic_columns)["expected_cost"]
    reordered = model.predict_parts(shuffled, chronic_columns)["expected_cost"]
    np.testing.assert_allclose(
        original.sort_index().to_numpy(), reordered.sort_index().to_numpy(), rtol=1e-10
    )
