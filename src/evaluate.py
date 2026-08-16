"""Evaluation metrics for the cost model and the targeting decision it supports.

Two different questions are being asked of the same predictions. The statistical
question is how close the predicted cost is to the realised cost. The operational
question is whether ranking members by predicted cost puts the expensive ones near
the top, because that is all a care management team actually needs to target
outreach. The second question is answered with concentration and capture curves,
which depend only on the ranking and are therefore unaffected by a shift in the
overall cost level between years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def participation_metrics(actual_flag: pd.Series, probability: np.ndarray) -> dict:
    return {
        "auc": round(float(roc_auc_score(actual_flag, probability)), 4),
        "observed_rate": round(float(actual_flag.mean()), 4),
        "predicted_rate": round(float(np.mean(probability)), 4),
    }


def severity_metrics(actual_cost: pd.Series, predicted_conditional: np.ndarray) -> dict:
    positive = actual_cost > 0
    actual = actual_cost[positive]
    predicted = predicted_conditional[positive.to_numpy()]
    return {
        "members_with_cost": int(positive.sum()),
        "mean_actual": round(float(actual.mean()), 2),
        "mean_predicted": round(float(predicted.mean()), 2),
        "mean_absolute_error": round(float(np.mean(np.abs(actual - predicted))), 2),
        "spearman": round(float(pd.Series(predicted, index=actual.index).corr(actual, method="spearman")), 4),
    }


def combined_metrics(actual_cost: pd.Series, expected_cost: np.ndarray) -> dict:
    """Overall accuracy and calibration of the combined expected cost prediction."""
    actual = actual_cost.to_numpy(dtype=float)
    predicted = np.asarray(expected_cost, dtype=float)
    residual = actual - predicted
    total_variance = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total_variance if total_variance else np.nan
    return {
        "r_squared": round(r_squared, 4),
        "mean_absolute_error": round(float(np.mean(np.abs(residual))), 2),
        "mean_actual": round(float(actual.mean()), 2),
        "mean_predicted": round(float(predicted.mean()), 2),
        "predictive_ratio": round(float(predicted.mean() / actual.mean()), 4),
        "spearman": round(float(pd.Series(predicted).corr(pd.Series(actual), method="spearman")), 4),
        "pearson": round(float(pd.Series(predicted).corr(pd.Series(actual))), 4),
    }


def high_cost_auc(actual_cost: pd.Series, expected_cost: np.ndarray, quantile: float) -> dict:
    threshold = float(actual_cost.quantile(quantile))
    flag = (actual_cost >= threshold).astype(int)
    return {
        "quantile": quantile,
        "cost_threshold": round(threshold, 2),
        "members_flagged": int(flag.sum()),
        "auc": round(float(roc_auc_score(flag, expected_cost)), 4),
    }


def concentration_curve(
    actual_cost: pd.Series,
    expected_cost: np.ndarray,
    top_k_grid: list[float],
    quantile: float,
) -> pd.DataFrame:
    """Share of next year spend and of high cost members captured in the top K percent.

    Random targeting is the reference line: selecting K percent of members at random
    captures K percent of spend in expectation. The perfect line is what an oracle
    ranking on realised cost would capture, which bounds what any model could do.
    """
    frame = pd.DataFrame({"actual": actual_cost.to_numpy(dtype=float), "predicted": np.asarray(expected_cost, dtype=float)})
    threshold = frame["actual"].quantile(quantile)
    frame["is_high_cost"] = (frame["actual"] >= threshold).astype(int)

    by_prediction = frame.sort_values("predicted", ascending=False).reset_index(drop=True)
    by_actual = frame.sort_values("actual", ascending=False).reset_index(drop=True)

    total_cost = frame["actual"].sum()
    total_high = frame["is_high_cost"].sum()
    members = len(frame)

    rows = []
    for share in top_k_grid:
        cut = max(int(round(members * share)), 1)
        captured_cost = by_prediction["actual"].head(cut).sum()
        captured_high = by_prediction["is_high_cost"].head(cut).sum()
        oracle_cost = by_actual["actual"].head(cut).sum()
        rows.append(
            {
                "top_k_pct": round(share * 100, 2),
                "members_targeted": cut,
                "pct_of_spend_captured": round(float(captured_cost / total_cost * 100), 2),
                "pct_of_spend_random": round(share * 100, 2),
                "pct_of_spend_oracle": round(float(oracle_cost / total_cost * 100), 2),
                "lift_vs_random": round(float((captured_cost / total_cost) / share), 2),
                "pct_of_high_cost_members_captured": round(float(captured_high / total_high * 100), 2),
                "capture_lift_vs_random": round(float((captured_high / total_high) / share), 2),
            }
        )
    return pd.DataFrame(rows)


def calibration_by_decile(actual_cost: pd.Series, expected_cost: np.ndarray) -> pd.DataFrame:
    """Mean predicted against mean actual within deciles of predicted risk."""
    frame = pd.DataFrame({"actual": actual_cost.to_numpy(dtype=float), "predicted": np.asarray(expected_cost, dtype=float)})
    frame["decile"] = pd.qcut(frame["predicted"].rank(method="first"), 10, labels=False) + 1
    grouped = frame.groupby("decile").agg(
        members=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        mean_actual=("actual", "mean"),
        total_actual=("actual", "sum"),
    )
    grouped["predictive_ratio"] = grouped["mean_predicted"] / grouped["mean_actual"]
    grouped["share_of_total_spend_pct"] = grouped["total_actual"] / frame["actual"].sum() * 100
    return grouped.round(4).reset_index()
