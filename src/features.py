"""Feature construction for the prospective year pair design.

Every feature is taken from the feature year only. The target is total cost in the
following year, so nothing that would be unknown at scoring time can leak in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMOGRAPHIC_FEATURES = ["age", "is_female", "has_esrd"]

COVERAGE_FEATURES = ["part_a_months", "part_b_months", "hmo_months", "part_d_months"]

UTILISATION_RAW = [
    "ip_admissions",
    "ip_days",
    "op_visits",
    "op_emergency_visits",
    "carrier_claims",
    "carrier_distinct_physicians",
    "rx_fills",
    "rx_distinct_products",
    "rx_days_supply",
]

PRIOR_COST_RAW = ["total_cost", "ip_cost", "op_cost", "carrier_cost", "rx_cost"]


def _log1p_clipped(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0))


def build_modeling_frame(
    member_year: pd.DataFrame,
    feature_year: int,
    target_year: int,
    chronic_columns: list[str],
) -> pd.DataFrame:
    """Joins feature year attributes to the following year's realised cost.

    The join is an inner join on member id, which keeps the members enrolled in both
    years. Members who die during the target year are retained because end of life
    spending is a real and large part of what a care management team has to plan for,
    and dropping them would flatter the model.
    """
    features = member_year[member_year["year"] == feature_year].copy()
    target = (
        member_year.loc[member_year["year"] == target_year, ["DESYNPUF_ID", "total_cost", "died_in_year"]]
        .rename(columns={"total_cost": "target_cost", "died_in_year": "target_year_death"})
    )
    frame = features.merge(target, on="DESYNPUF_ID", how="inner")

    for column in PRIOR_COST_RAW:
        frame[f"log_prior_{column}"] = _log1p_clipped(frame[column])
    for column in UTILISATION_RAW:
        frame[f"log_{column}"] = _log1p_clipped(frame[column])

    frame["prior_had_cost"] = (frame["total_cost"] > 0).astype(int)
    frame["prior_had_admission"] = (frame["ip_admissions"] > 0).astype(int)
    frame["prior_had_emergency"] = (frame["op_emergency_visits"] > 0).astype(int)
    frame["chronic_condition_count"] = frame[chronic_columns].sum(axis=1)

    frame["target_has_cost"] = (frame["target_cost"] > 0).astype(int)
    frame["feature_year"] = feature_year
    frame["target_year"] = target_year
    return frame


def feature_columns(chronic_columns: list[str]) -> list[str]:
    """The design matrix column order shared by every model in the project.

    The counted number of chronic conditions is deliberately absent. It is the exact
    sum of the individual condition flags, so including both makes the design matrix
    singular and the fitted coefficients meaningless. The count is still built on the
    frame because it is used as a standalone reference ranking.
    """
    return (
        DEMOGRAPHIC_FEATURES
        + COVERAGE_FEATURES
        + list(chronic_columns)
        + [f"log_{column}" for column in UTILISATION_RAW]
        + [f"log_prior_{column}" for column in PRIOR_COST_RAW]
        + ["prior_had_cost", "prior_had_admission", "prior_had_emergency"]
    )


def design_matrix(
    frame: pd.DataFrame, chronic_columns: list[str], columns: list[str] | None = None
) -> pd.DataFrame:
    selected = columns if columns is not None else feature_columns(chronic_columns)
    matrix = frame[selected].astype(float)
    return matrix.fillna(matrix.median(numeric_only=True))


def near_collinear_pairs(matrix: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Finds feature pairs whose absolute correlation is at or above the threshold."""
    correlation = matrix.corr().abs()
    order = list(matrix.columns)
    pairs = []
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            value = float(correlation.loc[first, second])
            if value >= threshold:
                pairs.append((first, second, round(value, 6)))
    return pairs


def select_modeling_columns(
    frame: pd.DataFrame, chronic_columns: list[str], threshold: float
) -> tuple[list[str], pd.DataFrame]:
    """Drops the later member of any near duplicate feature pair.

    In this extract the distinct drug product count and the fill count are the same
    variable for 99.6 percent of members, and the distinct performing physician count
    tracks the carrier claim count almost as closely. Provider and product identifiers
    were randomised when the file was synthesised, so a distinct count of them mostly
    re-measures claim volume rather than breadth of care. Keeping both members of such
    a pair leaves the coefficients unidentifiable, so the first in the canonical order
    is kept and the second is recorded here as dropped.
    """
    candidates = feature_columns(chronic_columns)
    matrix = design_matrix(frame, chronic_columns, candidates)
    pairs = near_collinear_pairs(matrix, threshold)

    dropped: dict[str, tuple[str, float]] = {}
    for first, second, value in pairs:
        if first in dropped or second in dropped:
            continue
        dropped[second] = (first, value)

    kept = [column for column in candidates if column not in dropped]
    ledger = pd.DataFrame(
        [
            {"dropped_feature": name, "kept_feature": partner, "abs_correlation": value}
            for name, (partner, value) in dropped.items()
        ]
    )
    return kept, ledger
