"""Beneficiary panel construction from the annual summary files.

The summary file is the enrolment denominator. Working from the claim files alone
would silently drop every member who used no services, and those members are
exactly the negative class the first part of the two part model has to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data_io import read_beneficiary

DEMOGRAPHIC_COLUMNS = [
    "DESYNPUF_ID",
    "BENE_BIRTH_DT",
    "BENE_DEATH_DT",
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "BENE_ESRD_IND",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
]

SUMMARY_COST_COLUMNS = [
    "MEDREIMB_IP",
    "BENRES_IP",
    "PPPYMT_IP",
    "MEDREIMB_OP",
    "BENRES_OP",
    "PPPYMT_OP",
    "MEDREIMB_CAR",
    "BENRES_CAR",
    "PPPYMT_CAR",
]


def _age_at_year_end(birth_date: pd.Series, year: int) -> pd.Series:
    birth_year = (pd.to_numeric(birth_date, errors="coerce") // 10000).astype("Float64")
    return (year - birth_year).astype("Float64")


def load_beneficiary_year(config: Config, year: int) -> pd.DataFrame:
    """Reads one annual summary file and returns tidy demographics and condition flags."""
    chronic_columns = list(config["data"]["chronic_flag_columns"])
    columns = DEMOGRAPHIC_COLUMNS + chronic_columns + SUMMARY_COST_COLUMNS
    frame = read_beneficiary(config, year, usecols=columns)

    frame["year"] = year
    frame["age"] = _age_at_year_end(frame["BENE_BIRTH_DT"], year)
    frame["is_female"] = (frame["BENE_SEX_IDENT_CD"] == 2).astype(int)
    frame["has_esrd"] = (frame["BENE_ESRD_IND"].astype(str).str.upper() == "Y").astype(int)
    frame["died_in_year"] = frame["BENE_DEATH_DT"].notna().astype(int)

    yes_value = config["data"]["chronic_flag_yes_value"]
    for column in chronic_columns:
        frame[column] = (frame[column] == yes_value).astype(int)
    frame["chronic_condition_count"] = frame[chronic_columns].sum(axis=1)

    frame["part_a_months"] = frame["BENE_HI_CVRAGE_TOT_MONS"].fillna(0)
    frame["part_b_months"] = frame["BENE_SMI_CVRAGE_TOT_MONS"].fillna(0)
    frame["hmo_months"] = frame["BENE_HMO_CVRAGE_TOT_MONS"].fillna(0)
    frame["part_d_months"] = frame["PLAN_CVRG_MOS_NUM"].fillna(0)

    frame["summary_reimbursement"] = frame[["MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR"]].sum(axis=1)

    keep = (
        ["DESYNPUF_ID", "year", "age", "is_female", "has_esrd", "died_in_year"]
        + chronic_columns
        + [
            "chronic_condition_count",
            "part_a_months",
            "part_b_months",
            "hmo_months",
            "part_d_months",
            "summary_reimbursement",
        ]
        + SUMMARY_COST_COLUMNS
    )
    return frame[keep]


def build_beneficiary_panel(config: Config) -> pd.DataFrame:
    """Stacks every available year into one member-year enrolment panel."""
    frames = [load_beneficiary_year(config, year) for year in config.years]
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["DESYNPUF_ID", "year"]).reset_index(drop=True)


def build_member_year(config: Config, panel: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Left joins claim aggregates onto the enrolment panel so non users survive as zeros."""
    from .claims import COST_COLUMNS, UTILISATION_COLUMNS

    merged = panel.merge(claims, on=["DESYNPUF_ID", "year"], how="left")
    for column in COST_COLUMNS + UTILISATION_COLUMNS + ["total_cost"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0.0)
    merged["has_any_cost"] = (merged["total_cost"] > 0).astype(int)
    return merged
