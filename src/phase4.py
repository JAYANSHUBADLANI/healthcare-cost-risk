"""Phase 4: driver explainability and figure generation."""

from __future__ import annotations

import json

import pandas as pd

from . import explain, features, figures
from .config import Config
from .features import build_modeling_frame, select_modeling_columns
from .model import fit_challenger, fit_two_part, predict_challenger


def run(config: Config, force: bool = False) -> dict:
    config.ensure_dirs()
    interim = config.path("interim_dir")
    reports = config.path("reports_dir")
    tables = reports / "tables"

    member_year = pd.read_parquet(interim / "member_year.parquet")
    tiered = pd.read_parquet(interim / "holdout_tiered.parquet")
    chronic_columns = list(config["data"]["chronic_flag_columns"])

    train_pair = config["model"]["train_pair"]
    eval_pair = config["model"]["eval_pair"]
    quantile = float(config["model"]["high_cost_quantile"])

    train = build_modeling_frame(member_year, train_pair[0], train_pair[1], chronic_columns)
    holdout = build_modeling_frame(member_year, eval_pair[0], eval_pair[1], chronic_columns)
    columns, _ = select_modeling_columns(train, chronic_columns, float(config["model"]["collinearity_threshold"]))

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

    model_importance = explain.permutation_importance(
        lambda frame: model.predict_parts(frame, chronic_columns)["expected_cost"].to_numpy(),
        holdout,
        chronic_columns,
        columns,
        holdout["target_cost"],
        quantile,
        seed=int(config["model"]["random_seed"]),
    )
    challenger_importance = explain.permutation_importance(
        lambda frame: predict_challenger(challenger, frame, chronic_columns, columns),
        holdout,
        chronic_columns,
        columns,
        holdout["target_cost"],
        quantile,
        seed=int(config["model"]["random_seed"]),
    )

    clinical_block = [
        column
        for column in columns
        if column in set(features.DEMOGRAPHIC_FEATURES + features.COVERAGE_FEATURES + chronic_columns)
    ]
    utilisation_block = [column for column in columns if column not in set(clinical_block)]
    ablation = explain.ablation_study(
        train,
        holdout,
        chronic_columns,
        {
            "clinical_profile_only": clinical_block,
            "prior_utilisation_only": utilisation_block,
            "full_model": columns,
        },
        quantile,
        fit_two_part,
    )
    ablation.to_csv(tables / "feature_block_ablation.csv", index=False)

    condition_profile = explain.marginal_condition_profile(holdout, chronic_columns, "target_cost")
    count_profile = explain.condition_count_profile(holdout, "chronic_condition_count", "target_cost")

    model_importance.to_csv(tables / "permutation_importance.csv", index=False)
    challenger_importance.to_csv(tables / "permutation_importance_challenger.csv", index=False)
    condition_profile.to_csv(tables / "condition_cost_profile.csv", index=False)
    count_profile.to_csv(tables / "condition_count_profile.csv", index=False)

    written = figures.build_all(tables, reports / "figures")

    findings = {
        "feature_block_ablation": ablation.to_dict("records"),
        "top_features_two_part": model_importance.head(8).to_dict("records"),
        "top_features_challenger": challenger_importance.head(8).to_dict("records"),
        "figures_written": written,
    }
    (reports / "phase4_findings.json").write_text(json.dumps(findings, indent=2, default=str))

    return {
        "feature_block_ablation": ablation.to_dict("records"),
        "top_features": model_importance.head(10).to_dict("records"),
        "condition_profile": condition_profile.to_dict("records"),
        "condition_count_profile": count_profile.to_dict("records"),
        "figures": written,
    }
