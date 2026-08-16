"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Wraps the YAML configuration and resolves paths against the project root."""

    def __init__(self, values: dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self.values = values
        self.root = root

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(target, "r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        return cls(values)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def path(self, key: str) -> Path:
        resolved = self.root / self.values["paths"][key]
        return resolved

    def ensure_dirs(self) -> None:
        for key in ("interim_dir", "reports_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)
        for sub in ("tables", "figures"):
            (self.path("reports_dir") / sub).mkdir(parents=True, exist_ok=True)

    @property
    def sample(self) -> int:
        return int(self.values["data"]["sample"])

    @property
    def years(self) -> list[int]:
        return list(self.values["data"]["years"])
