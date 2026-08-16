"""Driver explainability for the risk score.

Two views are produced because they answer different questions and neither is
sufficient alone.

The coefficient table says how the model uses a feature holding the others fixed.
Several utilisation features here are strongly correlated with each other, so an
individual partial coefficient can carry a counterintuitive sign without meaning that
the underlying condition lowers cost. Permutation importance sidesteps that by asking
what the ranking actually loses when a feature is scrambled, and the marginal profile
reports the plain observed cost difference a stakeholder can sanity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import design_matrix


def permutation_importance(
    predict_fn,
    frame: pd.DataFrame,
    chronic_columns: list[str],
    columns: list[str],
    actual_cost: pd.Series,
    quantile: float,
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Drop in high cost ranking quality when each feature is randomly permuted.

    The metric is the area under the curve for identifying the top decile of realised
    cost, because that is the decision the score is used for.
    """
    rng = np.random.default_rng(seed)
    threshold = float(actual_cost.quantile(quantile))
    flag = (actual_cost >= threshold).astype(int).to_numpy()

    working = frame.copy()
    baseline = float(roc_auc_score(flag, predict_fn(working)))

    rows = []
    for column in columns:
        original = working[column].to_numpy().copy()
        drops = []
        for _ in range(repeats):
            working[column] = rng.permutation(original)
            score = float(roc_auc_score(flag, predict_fn(working)))
            drops.append(baseline - score)
        working[column] = original
        rows.append(
            {
                "feature": column,
                "mean_auc_drop": round(float(np.mean(drops)), 5),
                "std_auc_drop": round(float(np.std(drops)), 5),
            }
        )

    table = pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)
    table.insert(0, "baseline_auc", round(baseline, 4))
    return table


def ablation_study(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    chronic_columns: list[str],
    blocks: dict[str, list[str]],
    quantile: float,
    fit_fn,
) -> pd.DataFrame:
    """Refits the model on feature blocks in isolation to price each block's contribution.

    Permutation importance says what a fitted model leans on. It cannot say what a
    feature would be worth if the model had nothing else, because a feature whose
    information is duplicated elsewhere looks worthless either way. Refitting on each
    block separately answers the question a stakeholder actually asks, which is
    whether the recorded conditions carry independent signal at all.
    """
    threshold = float(holdout["target_cost"].quantile(quantile))
    flag = (holdout["target_cost"] >= threshold).astype(int)
    total_spend = float(holdout["target_cost"].sum())
    cut = max(int(round(len(holdout) * 0.10)), 1)

    rows = []
    for name, columns in blocks.items():
        model = fit_fn(train, chronic_columns, columns)
        predicted = model.predict_parts(holdout, chronic_columns)["expected_cost"].to_numpy()
        ordered = holdout.assign(score=predicted).sort_values("score", ascending=False)
        captured = float(ordered["target_cost"].head(cut).sum() / total_spend * 100)
        rows.append(
            {
                "feature_block": name,
                "features_used": len(columns),
                "high_cost_auc": round(float(roc_auc_score(flag, predicted)), 4),
                "pct_spend_captured_top_10": round(captured, 2),
                "spearman_with_actual": round(
                    float(pd.Series(predicted).corr(holdout["target_cost"].reset_index(drop=True), method="spearman")), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def marginal_condition_profile(
    frame: pd.DataFrame, chronic_columns: list[str], actual_column: str
) -> pd.DataFrame:
    """Observed next year cost with and without each chronic condition.

    This is a raw comparison and makes no attempt to hold other conditions constant,
    which is exactly why it is readable: it is the number a care manager would get by
    filtering the member list on that condition.
    """
    rows = []
    for column in chronic_columns:
        present = frame.loc[frame[column] == 1, actual_column]
        absent = frame.loc[frame[column] == 0, actual_column]
        rows.append(
            {
                "condition": column,
                "members_with_condition": int(len(present)),
                "prevalence_pct": round(float(len(present) / len(frame) * 100), 2),
                "mean_cost_with": round(float(present.mean()), 2),
                "mean_cost_without": round(float(absent.mean()), 2),
                "cost_ratio": round(float(present.mean() / absent.mean()), 3) if absent.mean() else np.nan,
                "cost_difference": round(float(present.mean() - absent.mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cost_ratio", ascending=False).reset_index(drop=True)


def condition_count_profile(frame: pd.DataFrame, count_column: str, actual_column: str) -> pd.DataFrame:
    """Mean realised cost by the number of chronic conditions carried."""
    grouped = frame.groupby(count_column).agg(
        members=(actual_column, "size"),
        mean_cost=(actual_column, "mean"),
        median_cost=(actual_column, "median"),
    )
    grouped["pct_of_members"] = grouped["members"] / len(frame) * 100
    return grouped.round(2).reset_index()
