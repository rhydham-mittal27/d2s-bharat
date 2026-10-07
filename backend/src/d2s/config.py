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
    match_threshold: float = 0.75  # semantic cut-off; set from the TechWolf sweep for the chosen model
    dataset_dir: Path = Path(__file__).resolve().parents[3] / "dataset"

    # Database (Supabase Postgres). Unset = run from the analysis files only.
    database_url: str | None = None
    database_echo: bool = False

    # Vector store (ChromaDB): embedded at chroma_path, or a Chroma server when chroma_host is set
    chroma_path: Path | None = None  # default: <dataset_dir>/chroma
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_ssl: bool = False

    # Agents (LangGraph + a local model served by Ollama; offline, no API key)
    ollama_url: str = "http://127.0.0.1:11434"
    agent_model: str = "qwen3:1.7b"
    agent_keep_alive: str = "30m"  # keep the model in memory between questions
    agent_llm_timeout_s: float = 60.0
    agent_llm_enabled: bool = True  # false: fast path / deterministic steps only

    # Auth (JWT). With auth on, every /api route except health and /api/auth/* needs a bearer token.
    auth_required: bool = True
    jwt_secret: str | None = None  # >= 32 random bytes; set in .env (see .env.example)
    jwt_ttl_minutes: int = 720  # 12 h access tokens; no refresh tokens in the prototype

    # Decision engine
    solver_time_limit_s: float = 10.0
    solver_workers: int = 8
    solver_gap_limit: float = 0.005
    solver_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.chroma_path is None:
        s.chroma_path = s.dataset_dir / "chroma"
    return s
