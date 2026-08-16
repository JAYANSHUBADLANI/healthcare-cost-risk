"""Phase 3: high cost identification, concentration, tiering and the business case."""

from __future__ import annotations

import json

import pandas as pd

from . import evaluate, tiering
from .config import Config


def run(config: Config, force: bool = False) -> dict:
    config.ensure_dirs()
    interim = config.path("interim_dir")
    tables = config.path("reports_dir") / "tables"

    holdout = pd.read_parquet(interim / "holdout_predictions.parquet")
    quantile = float(config["model"]["high_cost_quantile"])
    settings = config["business_case"]

    threshold = float(holdout["target_cost"].quantile(quantile))
    holdout["is_high_cost"] = (holdout["target_cost"] >= threshold).astype(int)
    holdout["risk_tier"] = tiering.assign_tiers(holdout["expected_cost"], settings["tier_cutoffs"])

    high_cost_summary = {
        "definition": f"top {round((1 - quantile) * 100)} percent of realised target year cost",
        "cost_threshold": round(threshold, 2),
        "members_above_threshold": int(holdout["is_high_cost"].sum()),
        "share_of_total_spend_pct": round(
            float(holdout.loc[holdout["is_high_cost"] == 1, "target_cost"].sum() / holdout["target_cost"].sum() * 100),
            2,
        ),
        "mean_cost_high_group": round(float(holdout.loc[holdout["is_high_cost"] == 1, "target_cost"].mean()), 2),
        "mean_cost_rest": round(float(holdout.loc[holdout["is_high_cost"] == 0, "target_cost"].mean()), 2),
    }

    curve = evaluate.concentration_curve(
        holdout["target_cost"], holdout["expected_cost"].to_numpy(), config["evaluation"]["top_k_grid"], quantile
    )
    profile = tiering.tier_profile(holdout, "risk_tier", "target_cost", "expected_cost")

    case = tiering.business_case(
        holdout,
        "risk_tier",
        "target_cost",
        settings["target_tier"],
        float(settings["outreach_cost_per_member"]),
        float(settings["avoidable_cost_fraction"]),
        float(settings["engagement_rate"]),
    )
    break_even = tiering.break_even_avoidable_fraction(
        holdout,
        "risk_tier",
        "target_cost",
        settings["target_tier"],
        float(settings["outreach_cost_per_member"]),
        float(settings["engagement_rate"]),
    )
    grid = tiering.sensitivity_grid(
        holdout,
        "risk_tier",
        "target_cost",
        settings["target_tier"],
        list(settings["sensitivity_outreach_costs"]),
        list(settings["sensitivity_avoidable_fractions"]),
        float(settings["engagement_rate"]),
    )

    curve.to_csv(tables / "high_cost_concentration.csv", index=False)
    profile.to_csv(tables / "risk_tier_profile.csv", index=False)
    case.to_csv(tables / "business_case.csv", index=False)
    grid.to_csv(tables / "business_case_sensitivity.csv", index=False)
    holdout.to_parquet(interim / "holdout_tiered.parquet", index=False)

    findings = {
        "high_cost": high_cost_summary,
        "assumptions": {
            "outreach_cost_per_member": float(settings["outreach_cost_per_member"]),
            "avoidable_cost_fraction": float(settings["avoidable_cost_fraction"]),
            "engagement_rate": float(settings["engagement_rate"]),
            "source": "assumed, not observed in the claims data",
        },
        "break_even_avoidable_fraction": round(break_even, 4),
        "business_case": case.to_dict("records"),
    }
    (config.path("reports_dir") / "phase3_findings.json").write_text(json.dumps(findings, indent=2, default=str))

    return {
        "high_cost": high_cost_summary,
        "tier_profile": profile.to_dict("records"),
        "business_case": case.to_dict("records"),
        "break_even_avoidable_fraction": round(break_even, 4),
        "concentration_at_selected_points": curve[curve["top_k_pct"].isin([5.0, 10.0, 20.0])].to_dict("records"),
    }
