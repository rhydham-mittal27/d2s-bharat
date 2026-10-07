"""Runtime settings, read from environment variables prefixed with ``D2S_``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="D2S_", env_file=".env", extra="ignore")

    # Workers
    queue_backend: Literal["inline", "arq"] = "inline"
    redis_url: str = "redis://localhost:6379/0"
    job_result_ttl_s: int = 86_400

    # ML
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: Literal["torch", "onnx", "openvino", "hashing"] = "torch"
    embedding_batch_size: int = 64
    dataset_dir: Path = Path(__file__).resolve().parents[3] / "dataset"

    # Decision engine
    solver_time_limit_s: float = 10.0
    solver_workers: int = 8
    solver_gap_limit: float = 0.005
    solver_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()
