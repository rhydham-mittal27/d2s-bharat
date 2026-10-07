"""ChromaDB collection of market skills: one record per canonical skill seen in the SAS postings, with
its embedding and market evidence as metadata, so a search answers in one call.

* One collection per embedding model (``market_skills__<model>``): switching models never mixes
  vectors from different spaces.
* Cosine space; Chroma returns cosine distance, reported here as similarity = 1 - distance.
* Embeddings are supplied by us (the same model that embeds queries); Chroma's own embedding
  function is disabled.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from d2s.config import get_settings
from d2s.services import store

log = logging.getLogger(__name__)

BATCH = 2000  # below Chroma's maximum batch size
META_FIELDS = ("posts_all", "posts_data", "share_all", "share_data", "median_salary_mid", "top_family",
               "pay_odds_ratio", "pay_ci_low", "pay_ci_high", "pay_q_fdr")


def chroma_client(path: Path | None = None):
    """Embedded persistent client by default; HTTP client when D2S_CHROMA_HOST is set."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    s = get_settings()
    opts = ChromaSettings(anonymized_telemetry=False)
    if s.chroma_host:
        return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port, ssl=s.chroma_ssl, settings=opts)
    p = Path(path or s.chroma_path)
    p.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(p), settings=opts)


def _collection_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name.split("/")[-1]).strip("-.")
    return f"market_skills__{slug}"[:63]


class SkillVectorStore:
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self.name = _collection_name(model_name)
        self.col = client.get_or_create_collection(
            self.name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
            metadata={"model_name": model_name, "source": "SAS job postings"})

    def count(self) -> int:
        return self.col.count()

    def rebuild(self, records: list[dict], embeddings: np.ndarray) -> int:
        """Replace the collection's contents. records: skill + META_FIELDS; embeddings row-aligned."""
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings differ in length")
        self.client.delete_collection(self.name)
        self.__init__(self.client, self.model_name)
        ids = [r["skill"] for r in records]
        # Chroma drops None-valued keys; leave them out explicitly and restore them on read (_full)
        metas = [{"skill": r["skill"], "skill_lower": r["skill"].lower(),
                  **{k: r[k] for k in META_FIELDS if r.get(k) is not None}} for r in records]
        vecs = np.asarray(embeddings, dtype=np.float32)
        for i in range(0, len(ids), BATCH):
            self.col.upsert(ids=ids[i:i + BATCH], embeddings=vecs[i:i + BATCH].tolist(),
                            metadatas=metas[i:i + BATCH])
        return self.count()

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        """k nearest skills as (metadata, cosine similarity), best first."""
        if self.count() == 0:
            return []
        r = self.col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).ravel().tolist()],
                           n_results=min(k, self.count()), include=["metadatas", "distances"])
        return [(_full(m), 1.0 - float(d)) for m, d in zip(r["metadatas"][0], r["distances"][0], strict=True)]

    def by_name(self, skill: str) -> dict | None:
        r = self.col.get(where={"skill_lower": skill.strip().lower()}, limit=1, include=["metadatas"])
        return _full(r["metadatas"][0]) if r["ids"] else None


def _full(meta: dict) -> dict:
    """Every field present; missing (None) values come back as None."""
    return {"skill": meta["skill"], **{k: meta.get(k) for k in META_FIELDS}}


# ---- building the index from the analysis outputs ---------------------------------------------------------
def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def market_skill_records(min_posts: int = 3) -> list[dict]:
    stats = store.artifact_table("skill_stats.csv")
    stats = stats[stats["posts_all"] >= min_posts].reset_index(drop=True)
    pay = store.table("rq1/pay_adjusted_all.csv").set_index("skill")
    out = []
    for r in stats.to_dict("records"):
        p = pay.loc[r["skill"]] if r["skill"] in pay.index else None
        out.append({
            "skill": str(r["skill"]), "posts_all": int(r["posts_all"]), "posts_data": int(r["posts_data"]),
            "share_all": float(r["share_all"]), "share_data": float(r["share_data"]),
            "median_salary_mid": _num(r["median_salary_mid"]) if pd.notna(r["median_salary_mid"]) else None,
            "top_family": r["top_family"] if isinstance(r["top_family"], str) else None,
            "pay_odds_ratio": _num(p["odds_ratio"]) if p is not None else None,
            "pay_ci_low": _num(p["or_ci_low"]) if p is not None else None,
            "pay_ci_high": _num(p["or_ci_high"]) if p is not None else None,
            "pay_q_fdr": _num(p["q_fdr"]) if p is not None else None,
        })
    return out


def _cached_embeddings(names: list[str], model_name: str) -> np.ndarray | None:
    cache = store.artifacts() / "skill_lookup.npz"  # written by build_artifacts.py
    if not cache.exists():
        return None
    z = np.load(cache, allow_pickle=False)
    if str(z["model_name"]) == model_name and list(z["names"]) == names:
        return z["matrix"]
    return None


def build_skill_index(vs: SkillVectorStore, embedder=None) -> int:
    """(Re)build the collection: cached embeddings when they match the model, else embed now."""
    records = market_skill_records()
    names = [r["skill"] for r in records]
    matrix = _cached_embeddings(names, vs.model_name)
    if matrix is None:
        if embedder is None:
            raise ValueError(f"no cached embeddings for {vs.model_name}; pass an embedder")
        matrix = embedder.embed_documents(names)
    n = vs.rebuild(records, matrix)
    log.info("chroma: %s holds %d skills", vs.name, n)
    return n
