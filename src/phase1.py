"""Phase 1: build the member-year panel and audit what the data can support."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import audit
from .claims import build_member_year_claims
from .cohort import build_beneficiary_panel, build_member_year
from .config import Config
from .data_io import (
    beneficiary_path,
    carrier_paths,
    inpatient_path,
    outpatient_path,
    pde_path,
)


def _count_data_rows(path: Path) -> int:
    with open(path, "rb") as handle:
        total = sum(1 for _ in handle)
    return max(total - 1, 0)


def collect_row_counts(config: Config) -> dict[str, int]:
    counts = {}
    for year in config.years:
        counts[f"beneficiary_{year}"] = _count_data_rows(beneficiary_path(config, year))
    counts["inpatient_claims"] = _count_data_rows(inpatient_path(config))
    counts["outpatient_claims"] = _count_data_rows(outpatient_path(config))
    counts["carrier_claims"] = sum(_count_data_rows(path) for path in carrier_paths(config))
    counts["pde_events"] = _count_data_rows(pde_path(config))
    return counts


def run(config: Config, force: bool = False) -> dict:
    config.ensure_dirs()
    interim = config.path("interim_dir")
    tables = config.path("reports_dir") / "tables"

    claims_cache = interim / "member_year_claims.parquet"
    if claims_cache.exists() and not force:
        claims = pd.read_parquet(claims_cache)
    else:
        claims = build_member_year_claims(config)
        claims.to_parquet(claims_cache, index=False)

    panel = build_beneficiary_panel(config)
    member_year = build_member_year(config, panel, claims)
    member_year.to_parquet(interim / "member_year.parquet", index=False)

    counts_cache = interim / "row_counts.json"
    if counts_cache.exists() and not force:
        counts = json.loads(counts_cache.read_text())
    else:
        counts = collect_row_counts(config)
        counts_cache.write_text(json.dumps(counts, indent=2))

    row_check = audit.check_row_counts(config, counts)
    prevalence = audit.check_chronic_prevalence(config, member_year)
    reconciliation = audit.reconcile_costs(member_year)
    distribution = audit.cost_distribution_profile(member_year)
    signal = audit.longitudinal_signal(member_year, config["model"]["high_cost_quantile"])
    level_shift = audit.year_level_shift(member_year)
    emergency = audit.emergency_code_coverage(member_year)

    prevalence.to_csv(tables / "chronic_prevalence_check.csv", index=False)
    reconciliation.to_csv(tables / "cost_reconciliation.csv", index=False)
    distribution.to_csv(tables / "cost_distribution.csv", index=False)
    signal.to_csv(tables / "longitudinal_signal.csv", index=False)
    level_shift.to_csv(tables / "year_level_shift.csv", index=False)

    findings = {
        "row_counts": row_check,
        "emergency_proxy": emergency,
        "members_2008": int((member_year["year"] == 2008).sum()),
        "panel_rows": int(len(member_year)),
    }
    (config.path("reports_dir") / "phase1_findings.json").write_text(json.dumps(findings, indent=2))

    return {
        "panel_rows": len(member_year),
        "row_counts_match_published": row_check["all_match"],
        "cost_distribution": distribution.to_dict("records"),
        "longitudinal_signal": signal.to_dict("records"),
        "level_shift": level_shift.to_dict("records"),
        "emergency_proxy": emergency,
    }
