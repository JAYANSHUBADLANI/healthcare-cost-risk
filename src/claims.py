"""Aggregation of the four claim sources to member-year totals.

Cost is measured as the amount borne by the payer, not billed charges. For the
institutional and professional files that is the claim payment amount. For the
drug file the payer-borne share is the total drug cost less the patient payment,
because DE-SynPUF carries no third party payment field.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data_io import (
    carrier_paths,
    inpatient_path,
    numbered_columns,
    outpatient_path,
    pde_path,
    year_from_date,
)

CARRIER_LINE_COUNT = 13
CARRIER_CHUNK_ROWS = 400_000
PDE_CHUNK_ROWS = 1_000_000
OUTPATIENT_HCPCS_COUNT = 45


def _accumulate_distinct(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=keys)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates()


def aggregate_inpatient(config: Config) -> pd.DataFrame:
    """Inpatient cost, admission count, length of stay and distinct facilities."""
    columns = [
        "DESYNPUF_ID",
        "CLM_ID",
        "CLM_THRU_DT",
        "CLM_PMT_AMT",
        "CLM_UTLZTN_DAY_CNT",
        "PRVDR_NUM",
    ]
    frame = pd.read_csv(inpatient_path(config), usecols=columns)
    frame["year"] = year_from_date(frame["CLM_THRU_DT"])
    frame = frame.dropna(subset=["year"])

    grouped = frame.groupby(["DESYNPUF_ID", "year"], observed=True)
    result = grouped.agg(
        ip_cost=("CLM_PMT_AMT", "sum"),
        ip_admissions=("CLM_ID", "nunique"),
        ip_days=("CLM_UTLZTN_DAY_CNT", "sum"),
        ip_distinct_facilities=("PRVDR_NUM", "nunique"),
    )
    return result.reset_index()


def aggregate_outpatient(config: Config) -> pd.DataFrame:
    """Outpatient cost, visit count, emergency visit count and distinct facilities."""
    hcpcs_columns = numbered_columns("HCPCS_CD_", OUTPATIENT_HCPCS_COUNT)
    columns = ["DESYNPUF_ID", "CLM_ID", "CLM_THRU_DT", "CLM_PMT_AMT", "PRVDR_NUM"] + hcpcs_columns
    frame = pd.read_csv(outpatient_path(config), usecols=columns, dtype={c: "str" for c in hcpcs_columns})
    frame["year"] = year_from_date(frame["CLM_THRU_DT"])
    frame = frame.dropna(subset=["year"])

    emergency_codes = set(str(code) for code in config["data"]["emergency_hcpcs_codes"])
    hcpcs = frame[hcpcs_columns]
    is_emergency = hcpcs.isin(emergency_codes).any(axis=1)
    frame["is_emergency"] = is_emergency.astype(int)

    grouped = frame.groupby(["DESYNPUF_ID", "year"], observed=True)
    result = grouped.agg(
        op_cost=("CLM_PMT_AMT", "sum"),
        op_visits=("CLM_ID", "nunique"),
        op_emergency_visits=("is_emergency", "sum"),
        op_distinct_facilities=("PRVDR_NUM", "nunique"),
    )
    return result.reset_index()


def aggregate_carrier(config: Config) -> pd.DataFrame:
    """Professional cost, claim count and distinct performing physicians.

    The carrier file is line structured with thirteen line slots per claim, so the
    payer-borne amount is the row sum of the line payment columns. The file ships in
    two segments that together exceed two gigabytes, so it is read in chunks and the
    distinct physician count is built by deduplicating member, year and physician
    triples across every chunk rather than summing per chunk counts.
    """
    payment_columns = numbered_columns("LINE_NCH_PMT_AMT_", CARRIER_LINE_COUNT)
    npi_columns = numbered_columns("PRF_PHYSN_NPI_", CARRIER_LINE_COUNT)
    columns = ["DESYNPUF_ID", "CLM_ID", "CLM_THRU_DT"] + payment_columns + npi_columns

    totals: list[pd.DataFrame] = []
    physician_pairs: list[pd.DataFrame] = []

    for path in carrier_paths(config):
        reader = pd.read_csv(path, usecols=columns, chunksize=CARRIER_CHUNK_ROWS)
        for chunk in reader:
            chunk["year"] = year_from_date(chunk["CLM_THRU_DT"])
            chunk = chunk.dropna(subset=["year"])
            if chunk.empty:
                continue
            chunk["carrier_cost"] = chunk[payment_columns].sum(axis=1)
            totals.append(
                chunk.groupby(["DESYNPUF_ID", "year"], observed=True)
                .agg(carrier_cost=("carrier_cost", "sum"), carrier_claims=("CLM_ID", "nunique"))
                .reset_index()
            )
            melted = chunk.melt(
                id_vars=["DESYNPUF_ID", "year"],
                value_vars=npi_columns,
                value_name="npi",
            )[["DESYNPUF_ID", "year", "npi"]]
            melted = melted.dropna(subset=["npi"]).drop_duplicates()
            physician_pairs.append(melted)

    combined = pd.concat(totals, ignore_index=True)
    result = (
        combined.groupby(["DESYNPUF_ID", "year"], observed=True)
        .agg(carrier_cost=("carrier_cost", "sum"), carrier_claims=("carrier_claims", "sum"))
        .reset_index()
    )

    distinct = _accumulate_distinct(physician_pairs, ["DESYNPUF_ID", "year", "npi"])
    physician_counts = (
        distinct.groupby(["DESYNPUF_ID", "year"], observed=True)
        .size()
        .rename("carrier_distinct_physicians")
        .reset_index()
    )
    result = result.merge(physician_counts, on=["DESYNPUF_ID", "year"], how="left")
    result["carrier_distinct_physicians"] = result["carrier_distinct_physicians"].fillna(0)
    return result


def aggregate_pde(config: Config) -> pd.DataFrame:
    """Drug cost borne by the payer, fill count, distinct products and days supply."""
    columns = [
        "DESYNPUF_ID",
        "PDE_ID",
        "SRVC_DT",
        "PROD_SRVC_ID",
        "DAYS_SUPLY_NUM",
        "PTNT_PAY_AMT",
        "TOT_RX_CST_AMT",
    ]
    totals: list[pd.DataFrame] = []
    product_pairs: list[pd.DataFrame] = []

    reader = pd.read_csv(
        pde_path(config),
        usecols=columns,
        chunksize=PDE_CHUNK_ROWS,
        dtype={"PROD_SRVC_ID": "str"},
    )
    for chunk in reader:
        chunk["year"] = year_from_date(chunk["SRVC_DT"])
        chunk = chunk.dropna(subset=["year"])
        if chunk.empty:
            continue
        chunk["rx_cost"] = chunk["TOT_RX_CST_AMT"] - chunk["PTNT_PAY_AMT"]
        totals.append(
            chunk.groupby(["DESYNPUF_ID", "year"], observed=True)
            .agg(
                rx_cost=("rx_cost", "sum"),
                rx_fills=("PDE_ID", "nunique"),
                rx_days_supply=("DAYS_SUPLY_NUM", "sum"),
            )
            .reset_index()
        )
        pairs = chunk[["DESYNPUF_ID", "year", "PROD_SRVC_ID"]].drop_duplicates()
        product_pairs.append(pairs)

    combined = pd.concat(totals, ignore_index=True)
    result = (
        combined.groupby(["DESYNPUF_ID", "year"], observed=True)
        .agg(
            rx_cost=("rx_cost", "sum"),
            rx_fills=("rx_fills", "sum"),
            rx_days_supply=("rx_days_supply", "sum"),
        )
        .reset_index()
    )

    distinct = _accumulate_distinct(product_pairs, ["DESYNPUF_ID", "year", "PROD_SRVC_ID"])
    product_counts = (
        distinct.groupby(["DESYNPUF_ID", "year"], observed=True)
        .size()
        .rename("rx_distinct_products")
        .reset_index()
    )
    result = result.merge(product_counts, on=["DESYNPUF_ID", "year"], how="left")
    result["rx_distinct_products"] = result["rx_distinct_products"].fillna(0)
    return result


COST_COLUMNS = ["ip_cost", "op_cost", "carrier_cost", "rx_cost"]

UTILISATION_COLUMNS = [
    "ip_admissions",
    "ip_days",
    "ip_distinct_facilities",
    "op_visits",
    "op_emergency_visits",
    "op_distinct_facilities",
    "carrier_claims",
    "carrier_distinct_physicians",
    "rx_fills",
    "rx_days_supply",
    "rx_distinct_products",
]


def build_member_year_claims(config: Config) -> pd.DataFrame:
    """Outer joins the four sources into one member-year table of cost and utilisation."""
    parts = [
        aggregate_inpatient(config),
        aggregate_outpatient(config),
        aggregate_carrier(config),
        aggregate_pde(config),
    ]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["DESYNPUF_ID", "year"], how="outer")

    fill_columns = COST_COLUMNS + UTILISATION_COLUMNS
    for column in fill_columns:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0)

    merged["total_cost"] = merged[COST_COLUMNS].sum(axis=1)
    merged["year"] = merged["year"].astype(int)
    return merged.sort_values(["DESYNPUF_ID", "year"]).reset_index(drop=True)
