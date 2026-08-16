"""The two part cost model.

Healthcare spend is a mixture: a point mass of members who use nothing, and a heavy
right tail among the rest. One regression has to satisfy both at once and ends up
fitting neither, so the cost is split into the probability of any spend and the
conditional level of spend given that it happens.

Part two uses a Gamma family with a log link rather than ordinary least squares on
logged cost. A log OLS fit predicts the mean of the log, and converting that back to
the original scale needs a smearing correction that is only valid when the error
variance is constant. The Gamma GLM models the mean on the original scale directly
and lets the variance grow with the mean, which is what claims data does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from .features import design_matrix


@dataclass
class TwoPartModel:
    """Holds both fitted parts plus the scaler shared between them."""

    scaler: StandardScaler
    participation: sm.GLM
    severity: sm.GLM
    columns: list[str]

    def _scaled(self, frame: pd.DataFrame, chronic_columns: list[str]) -> pd.DataFrame:
        matrix = design_matrix(frame, chronic_columns, self.columns)
        scaled = pd.DataFrame(
            self.scaler.transform(matrix), columns=self.columns, index=matrix.index
        )
        return sm.add_constant(scaled, has_constant="add")

    def predict_parts(self, frame: pd.DataFrame, chronic_columns: list[str]) -> pd.DataFrame:
        exog = self._scaled(frame, chronic_columns)
        probability = np.asarray(self.participation.predict(exog))
        conditional = np.asarray(self.severity.predict(exog))
        expected = probability * conditional
        return pd.DataFrame(
            {
                "probability_any_cost": probability,
                "conditional_cost": conditional,
                "expected_cost": expected,
            },
            index=frame.index,
        )


class RankDeficientDesignError(ValueError):
    """Raised when the design matrix is singular, which makes coefficients meaningless."""


class ModelConvergenceError(RuntimeError):
    """Raised when an iteratively reweighted least squares fit did not converge."""


def assert_full_rank(matrix: pd.DataFrame) -> None:
    """Guards against collinear features silently producing unbounded coefficients."""
    rank = int(np.linalg.matrix_rank(matrix.to_numpy(dtype=float)))
    expected = matrix.shape[1]
    if rank < expected:
        raise RankDeficientDesignError(
            f"design matrix has rank {rank} but {expected} columns, so at least one "
            "feature is an exact linear combination of the others"
        )


def _assert_converged(fit, label: str) -> None:
    if not getattr(fit, "converged", True):
        raise ModelConvergenceError(f"{label} did not converge")


def fit_two_part(
    train: pd.DataFrame,
    chronic_columns: list[str],
    columns: list[str] | None = None,
    max_iter: int = 100,
) -> TwoPartModel:
    """Fits the participation and severity parts on the training year pair."""
    matrix = design_matrix(train, chronic_columns, columns)
    assert_full_rank(matrix)
    columns = list(matrix.columns)

    scaler = StandardScaler().fit(matrix)
    scaled = pd.DataFrame(scaler.transform(matrix), columns=columns, index=matrix.index)
    exog = sm.add_constant(scaled, has_constant="add")

    participation = sm.GLM(
        train["target_has_cost"].to_numpy(),
        exog,
        family=sm.families.Binomial(),
    ).fit(maxiter=max_iter)
    _assert_converged(participation, "participation model")

    positive = train["target_cost"] > 0
    severity = sm.GLM(
        train.loc[positive, "target_cost"].to_numpy(),
        exog.loc[positive.to_numpy()],
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(maxiter=max_iter)
    _assert_converged(severity, "severity model")

    return TwoPartModel(scaler=scaler, participation=participation, severity=severity, columns=columns)


def coefficient_table(model: TwoPartModel) -> pd.DataFrame:
    """Standardised coefficients from both parts, on one row per feature."""
    names = ["const"] + model.columns
    rows = []
    for feature, part_one, part_one_p, part_two, part_two_p in zip(
        names,
        model.participation.params,
        model.participation.pvalues,
        model.severity.params,
        model.severity.pvalues,
    ):
        rows.append(
            {
                "feature": feature,
                "participation_coef": round(float(part_one), 4),
                "participation_pvalue": float(part_one_p),
                "participation_odds_ratio": round(float(np.exp(part_one)), 4),
                "severity_coef": round(float(part_two), 4),
                "severity_pvalue": float(part_two_p),
                "severity_cost_multiplier": round(float(np.exp(part_two)), 4),
            }
        )
    table = pd.DataFrame(rows)
    table = table[table["feature"] != "const"]
    table["combined_abs_effect"] = table["participation_coef"].abs() + table["severity_coef"].abs()
    return table.sort_values("combined_abs_effect", ascending=False).reset_index(drop=True)


def fit_challenger(
    train: pd.DataFrame,
    chronic_columns: list[str],
    columns: list[str] | None = None,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> HistGradientBoostingRegressor:
    """A single stage gradient boosted challenger fitted with a Poisson deviance loss.

    This is the natural comparison for the two part structure: one model, trained on
    raw cost including the zeros, using a loss that tolerates a point mass at zero and
    a long tail.
    """
    matrix = design_matrix(train, chronic_columns, columns)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    model.fit(matrix, train["target_cost"].clip(lower=0))
    return model


def predict_challenger(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    chronic_columns: list[str],
    columns: list[str] | None = None,
) -> np.ndarray:
    return model.predict(design_matrix(frame, chronic_columns, columns))
