"""Phase 2: features, the two part model, and prospective evaluation.

The design is strictly prospective. The model is fitted on 2008 features against
realised 2009 cost, and then evaluated on 2009 features against realised 2010 cost.
The evaluation year pair is never seen during fitting, by either part.

Two reference policies are carried through the whole evaluation because they are what
the model has to beat to be worth deploying. Ranking members by last year's spend is
what a plan can do with no model at all, and ranking by counted chronic conditions is
the simple clinical rule a care management team would reach for otherwise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import evaluate
from .config import Config
from .features import build_modeling_frame, select_modeling_columns
from .model import coefficient_table, fit_challenger, fit_two_part, predict_challenger


def _reference_rankings(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "prior_year_cost": frame["total_cost"].to_numpy(dtype=float),
        "chronic_condition_count": frame["chronic_condition_count"].to_numpy(dtype=float),
    }


def run(config: Config, force: bool = False) -> dict:
    config.ensure_dirs()
    interim = config.path("interim_dir")
    tables = config.path("reports_dir") / "tables"

    member_year = pd.read_parquet(interim / "member_year.parquet")
    chronic_columns = list(config["data"]["chronic_flag_columns"])

    train_pair = config["model"]["train_pair"]
    eval_pair = config["model"]["eval_pair"]
    quantile = float(config["model"]["high_cost_quantile"])

    train = build_modeling_frame(member_year, train_pair[0], train_pair[1], chronic_columns)
    holdout = build_modeling_frame(member_year, eval_pair[0], eval_pair[1], chronic_columns)

    columns, collinearity_ledger = select_modeling_columns(
        train, chronic_columns, float(config["model"]["collinearity_threshold"])
    )
    collinearity_ledger.to_csv(tables / "dropped_collinear_features.csv", index=False)

    model = fit_two_part(train, chronic_columns, columns)
    challenger = fit_challenger(
        train,
        chronic_columns,
        columns,
        seed=int(config["model"]["random_seed"]),
        n_estimators=int(config["model"]["challenger_n_estimators"]),
        max_depth=int(config["model"]["challenger_max_depth"]),
        learning_rate=float(config["model"]["challenger_learning_rate"]),
    )

    parts = model.predict_parts(holdout, chronic_columns)
    holdout = pd.concat([holdout.reset_index(drop=True), parts.reset_index(drop=True)], axis=1)
    holdout["challenger_cost"] = predict_challenger(challenger, holdout, chronic_columns, columns)

    train_parts = model.predict_parts(train, chronic_columns)
    train_metrics = evaluate.combined_metrics(train["target_cost"], train_parts["expected_cost"].to_numpy())

    actual = holdout["target_cost"]
    expected = holdout["expected_cost"].to_numpy()

    metrics = {
        "train_pair": f"{train_pair[0]} to {train_pair[1]}",
        "eval_pair": f"{eval_pair[0]} to {eval_pair[1]}",
        "train_members": int(len(train)),
        "holdout_members": int(len(holdout)),
        "in_sample_combined": train_metrics,
        "participation": evaluate.participation_metrics(holdout["target_has_cost"], holdout["probability_any_cost"].to_numpy()),
        "severity": evaluate.severity_metrics(actual, holdout["conditional_cost"].to_numpy()),
        "combined": evaluate.combined_metrics(actual, expected),
        "high_cost": evaluate.high_cost_auc(actual, expected, quantile),
    }

    trend_factor = float(actual.mean() / expected.mean())
    metrics["level_shift"] = {
        "trend_factor_applied": round(trend_factor, 4),
        "note": (
            "mean spend fell 40.7 percent between the fitting year and the target year, "
            "so a model fitted on the earlier level over predicts the later one. This "
            "row rescales predictions by a single constant to separate the level error "
            "from the ranking quality. The constant is derived from the target year and "
            "is therefore a diagnostic decomposition, not a prospective result."
        ),
        "recalibrated": evaluate.combined_metrics(actual, expected * trend_factor),
    }

    ranking_rows = []
    curves = {}
    candidates = {"two_part_model": expected, "challenger_boosted": holdout["challenger_cost"].to_numpy()}
    candidates.update(_reference_rankings(holdout))

    for name, score in candidates.items():
        curve = evaluate.concentration_curve(actual, score, config["evaluation"]["top_k_grid"], quantile)
        curve.insert(0, "policy", name)
        curves[name] = curve
        at_ten = curve[curve["top_k_pct"] == 10.0].iloc[0]
        ranking_rows.append(
            {
                "policy": name,
                "high_cost_auc": round(float(evaluate.high_cost_auc(actual, score, quantile)["auc"]), 4),
                "spearman_with_actual": round(float(pd.Series(score).corr(actual.reset_index(drop=True), method="spearman")), 4),
                "pct_spend_captured_top_10": float(at_ten["pct_of_spend_captured"]),
                "pct_high_cost_captured_top_10": float(at_ten["pct_of_high_cost_members_captured"]),
                "lift_vs_random_top_10": float(at_ten["lift_vs_random"]),
            }
        )

    comparison = pd.DataFrame(ranking_rows).sort_values("pct_spend_captured_top_10", ascending=False)
    all_curves = pd.concat(curves.values(), ignore_index=True)
    calibration = evaluate.calibration_by_decile(actual, expected)
    coefficients = coefficient_table(model)

    comparison.to_csv(tables / "policy_comparison.csv", index=False)
    all_curves.to_csv(tables / "concentration_curves.csv", index=False)
    calibration.to_csv(tables / "calibration_by_decile.csv", index=False)
    coefficients.to_csv(tables / "model_coefficients.csv", index=False)

    keep = [
        "DESYNPUF_ID",
        "target_cost",
        "target_has_cost",
        "total_cost",
        "chronic_condition_count",
        "probability_any_cost",
        "conditional_cost",
        "expected_cost",
        "challenger_cost",
        "target_year_death",
    ] + chronic_columns
    holdout[keep].to_parquet(interim / "holdout_predictions.parquet", index=False)
    train.to_parquet(interim / "train_frame.parquet", index=False)

    (config.path("reports_dir") / "phase2_findings.json").write_text(json.dumps(metrics, indent=2, default=str))

    return {
        "metrics": metrics,
        "policy_comparison": comparison.to_dict("records"),
        "top_coefficients": coefficients.head(10).to_dict("records"),
    }
