"""Locations of analysis outputs and trained artifacts, with cached loaders.

Artifacts are produced by `scripts/build_artifacts.py`; analysis tables by the run_rq*.py scripts.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from d2s.config import get_settings

ROOT = Path(__file__).resolve().parents[4]
SAS_DIR = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"
CATALOGUE = ROOT / "reports" / "rq4" / "course_catalogue_assumptions.csv"


def processed() -> Path:
    return get_settings().dataset_dir / "processed"


def artifacts() -> Path:
    return get_settings().dataset_dir / "artifacts"


class MissingArtifact(FileNotFoundError):
    """Raised with a hint about which script produces the missing file."""


def _need(path: Path, producer: str) -> Path:
    if not path.exists():
        raise MissingArtifact(f"{path} not found; run `uv run python scripts/{producer}` first")
    return path


@lru_cache
def table(rel: str, producer: str = "run_rq1.py", **kw) -> pd.DataFrame:
    return pd.read_csv(_need(processed() / rel, producer), **kw)


@lru_cache
def summary(rel: str, producer: str) -> dict:
    return json.loads(_need(processed() / rel, producer).read_text())


@lru_cache
def artifact_table(name: str) -> pd.DataFrame:
    return pd.read_csv(_need(artifacts() / name, "build_artifacts.py"))


def artifact_path(name: str) -> Path:
    return _need(artifacts() / name, "build_artifacts.py")


def clear_caches() -> None:
    table.cache_clear()
    summary.cache_clear()
    artifact_table.cache_clear()
