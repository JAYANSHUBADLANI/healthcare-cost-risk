"""Shared fixtures built in memory so the suite runs without the CMS extract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CHRONIC_COLUMNS = ["SP_CHF", "SP_DIABETES", "SP_COPD"]


@pytest.fixture
def chronic_columns() -> list[str]:
    return list(CHRONIC_COLUMNS)


@pytest.fixture
def member_year() -> pd.DataFrame:
    """A small two year panel with a deliberate cost relationship between years.

    Utilisation is drawn with its own randomness rather than being a fixed multiple of
    cost, so the design matrix is not artificially rank deficient. The two exceptions
    are deliberate: the distinct physician and distinct product counts are made equal
    to the claim and fill counts, which is what the real extract does and what the
    collinearity pruning has to catch.
    """
    rng = np.random.default_rng(11)
    size = 600
    members = [f"M{index:04d}" for index in range(size)]
    latent = rng.gamma(shape=1.4, scale=1800, size=size)
    age = rng.integers(65, 95, size=size)
    female = rng.integers(0, 2, size=size)
    esrd = rng.binomial(1, 0.03, size=size)
    conditions = {name: rng.binomial(1, rate, size=size) for name, rate in zip(CHRONIC_COLUMNS, (0.28, 0.35, 0.14))}

    rows = []
    for year in (2008, 2009):
        for index, member in enumerate(members):
            base = latent[index] * (1.0 if year == 2008 else 0.9)
            cost = float(max(base + rng.gamma(shape=1.2, scale=700) - 900, 0.0))
            shares = rng.dirichlet([3.0, 2.0, 3.0, 2.0])
            admissions = int(rng.poisson(cost / 9000))
            carrier_claims = int(rng.poisson(max(cost / 900, 0.2)))
            fills = int(rng.poisson(max(cost / 700, 0.2)))
            rows.append(
                {
                    "DESYNPUF_ID": member,
                    "year": year,
                    "age": int(age[index]) + (year - 2008),
                    "is_female": int(female[index]),
                    "has_esrd": int(esrd[index]),
                    "died_in_year": 0,
                    "SP_CHF": int(conditions["SP_CHF"][index]),
                    "SP_DIABETES": int(conditions["SP_DIABETES"][index]),
                    "SP_COPD": int(conditions["SP_COPD"][index]),
                    "part_a_months": int(rng.integers(6, 13)),
                    "part_b_months": int(rng.integers(6, 13)),
                    "hmo_months": int(rng.integers(0, 4)),
                    "part_d_months": int(rng.integers(6, 13)),
                    "ip_cost": cost * shares[0],
                    "op_cost": cost * shares[1],
                    "carrier_cost": cost * shares[2],
                    "rx_cost": cost * shares[3],
                    "total_cost": cost,
                    "ip_admissions": admissions,
                    "ip_days": int(admissions * rng.integers(1, 9)) if admissions else 0,
                    "op_visits": int(rng.poisson(max(cost / 1500, 0.2))),
                    "op_emergency_visits": int(rng.poisson(max(cost / 12000, 0.05))),
                    "carrier_claims": carrier_claims,
                    "carrier_distinct_physicians": carrier_claims,
                    "rx_fills": fills,
                    "rx_distinct_products": fills,
                    "rx_days_supply": int(fills * rng.integers(20, 40)) if fills else 0,
                    "has_any_cost": int(cost > 0),
                }
            )
    frame = pd.DataFrame(rows)
    frame["chronic_condition_count"] = frame[CHRONIC_COLUMNS].sum(axis=1)
    return frame


@pytest.fixture
def scored_holdout() -> pd.DataFrame:
    """Predictions with a known imperfect but positive relationship to realised cost."""
    rng = np.random.default_rng(5)
    size = 4000
    actual = rng.gamma(shape=1.1, scale=3000, size=size)
    predicted = actual * rng.uniform(0.55, 1.45, size=size) + rng.normal(0, 400, size=size)
    return pd.DataFrame({"target_cost": actual, "expected_cost": np.clip(predicted, 1.0, None)})
