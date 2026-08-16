"""Locating and reading the DE-SynPUF extract files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config

BENE_TEMPLATE = "DE1_0_{year}_Beneficiary_Summary_File_Sample_{sample}.csv"
INPATIENT_TEMPLATE = "DE1_0_2008_to_2010_Inpatient_Claims_Sample_{sample}.csv"
OUTPATIENT_TEMPLATE = "DE1_0_2008_to_2010_Outpatient_Claims_Sample_{sample}.csv"
PDE_TEMPLATE = "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_{sample}.csv"
CARRIER_TEMPLATE = "DE1_0_2008_to_2010_Carrier_Claims_Sample_{sample}{segment}.csv"

CARRIER_SEGMENTS = ("A", "B")


class MissingExtractError(FileNotFoundError):
    """Raised when an expected DE-SynPUF file is not on disk."""


def _resolve(config: Config, filename: str) -> Path:
    target = config.path("unzipped_dir") / filename
    if not target.exists():
        raise MissingExtractError(
            f"expected {target}. Run `make data` to download and unpack the CMS extract."
        )
    return target


def beneficiary_path(config: Config, year: int) -> Path:
    return _resolve(config, BENE_TEMPLATE.format(year=year, sample=config.sample))


def inpatient_path(config: Config) -> Path:
    return _resolve(config, INPATIENT_TEMPLATE.format(sample=config.sample))


def outpatient_path(config: Config) -> Path:
    return _resolve(config, OUTPATIENT_TEMPLATE.format(sample=config.sample))


def pde_path(config: Config) -> Path:
    return _resolve(config, PDE_TEMPLATE.format(sample=config.sample))


def carrier_paths(config: Config) -> list[Path]:
    return [
        _resolve(config, CARRIER_TEMPLATE.format(sample=config.sample, segment=segment))
        for segment in CARRIER_SEGMENTS
    ]


def read_beneficiary(config: Config, year: int, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(beneficiary_path(config, year), usecols=usecols)


def year_from_date(series: pd.Series) -> pd.Series:
    """Converts a YYYYMMDD numeric claim date into a calendar year."""
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric // 10000).astype("Int64")


def numbered_columns(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(1, count + 1)]
