"""Hybrid skill extraction: semantic (embeddings) + lexical (exact labels), fused with RRF.

Semantic search finds paraphrases ("built dashboards for leadership" -> data visualization);
lexical search nails exact tool names ("PostgreSQL", "AutoCAD") that vectors can blur.
Each match keeps the clause it came from as evidence for the audit trail.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from d2s.ml.embeddings import Embedder
from d2s.ml.taxonomy import Skill

RRF_K = 60
_CLAUSE_SPLIT = re.compile(r"(?:[\n\r]+|[•·▪●\-\*]\s+|(?<=[.;!?])\s+)")
_WORD = re.compile(r"[a-z0-9+#.]+(?:[-/][a-z0-9+#.]+)*")
MAX_LABEL_WORDS = 5


class SkillMatch(BaseModel):
    skill_id: str
    label: str
    score: float  # fused RRF score, comparable within one extraction
    similarity: float  # best cosine similarity of any clause (0 if lexical only)
    lexical: bool
    evidence: str


def split_clauses(text: str, min_chars: int = 12) -> list[str]:
    parts = (p.strip(" \t-•*·:") for p in _CLAUSE_SPLIT.split(text))
    return [p for p in parts if len(p) >= min_chars]


def _norm(text: str) -> str:
    # Keep inner dots ("node.js", "asp.net") but drop sentence punctuation ("python." -> "python").
    tokens = (t.rstrip(".") for t in _WORD.findall(text.lower()))
    return " ".join(t for t in tokens if t)


@dataclass
class SkillIndex:
    ids: list[str]
    labels: list[str]
    matrix: np.ndarray  # (n_skills, dim), L2-normalised
    model_name: str

    @classmethod
    def build(cls, skills: list[Skill], embedder: Embedder, batch: int = 512) -> "SkillIndex":
        texts = [s.embedding_text for s in skills]
        chunks = [embedder.embed_documents(texts[i:i + batch]) for i in range(0, len(texts), batch)]
        matrix = np.vstack(chunks) if chunks else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls([s.id for s in skills], [s.label for s in skills], matrix, embedder.name)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids), labels=np.array(self.labels),
                            matrix=self.matrix, model_name=np.array(self.model_name))

    @classmethod
    def load(cls, path: Path) -> "SkillIndex":
        z = np.load(path, allow_pickle=False)
        return cls(list(z["ids"]), list(z["labels"]), z["matrix"], str(z["model_name"]))


class SkillMatcher:
    def __init__(
        self,
        skills: list[Skill],
        embedder: Embedder,
        index: SkillIndex | None = None,
        threshold: float = 0.6,
        top_k_per_clause: int = 3,
        min_lexical_len: int = 3,
    ):
        self.embedder = embedder
        self.index = index or SkillIndex.build(skills, embedder)
        if self.index.model_name != embedder.name:
            raise ValueError(f"index built with {self.index.model_name}, embedder is {embedder.name}")
        self.threshold = threshold
        self.top_k = top_k_per_clause
        self._labels = dict(zip(self.index.ids, self.index.labels, strict=True))
        self._lexicon: dict[str, str] = {}
        for s in skills:
            for name in (s.label, *s.alt_labels):
                key = _norm(name)
                if len(key) >= min_lexical_len and len(key.split()) <= MAX_LABEL_WORDS:
                    self._lexicon.setdefault(key, s.id)

    def _semantic(self, clauses: list[str]) -> dict[str, tuple[float, str]]:
        if not clauses or not len(self.index.ids):
            return {}
        return self._semantic_from_vectors(clauses, self.embedder.embed_queries(clauses))

    def _semantic_from_vectors(self, clauses: list[str], vectors: np.ndarray) -> dict[str, tuple[float, str]]:
        if not clauses:
            return {}
        sims = vectors @ self.index.matrix.T
        k = min(self.top_k, sims.shape[1])
        best: dict[str, tuple[float, str]] = {}
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for ci, row in enumerate(top):
            for j in row:
                sim = float(sims[ci, j])
                if sim < self.threshold:
                    continue
                sid = self.index.ids[j]
                if sid not in best or sim > best[sid][0]:
                    best[sid] = (sim, clauses[ci])
        return best

    def _lexical(self, clauses: list[str]) -> dict[str, tuple[int, str]]:
        hits: dict[str, tuple[int, str]] = {}
        counts: dict[str, int] = defaultdict(int)
        for clause in clauses:
            words = _norm(clause).split()
            for n in range(1, MAX_LABEL_WORDS + 1):
                for i in range(len(words) - n + 1):
                    sid = self._lexicon.get(" ".join(words[i:i + n]))
                    if sid:
                        counts[sid] += 1
                        hits.setdefault(sid, (0, clause))
        return {sid: (counts[sid], ev) for sid, (_, ev) in hits.items()}

    def label_of(self, skill_id: str) -> str:
        return self._labels.get(skill_id, skill_id)

    def lookup(self, label: str) -> str | None:
        """Exact (normalised) label/alt-label lookup, e.g. for recruiter-supplied skill tags."""
        return self._lexicon.get(_norm(label))

    @staticmethod
    def _clauses(text: str) -> list[str]:
        return split_clauses(text) or ([text.strip()] if text.strip() else [])

    def extract(self, text: str, limit: int = 25) -> list[SkillMatch]:
        clauses = self._clauses(text)
        return self._fuse(self._semantic(clauses), self._lexical(clauses), limit)

    def extract_lexical(self, text: str, limit: int = 50) -> list[SkillMatch]:
        """Exact-label matches only; no model call, so it scales to millions of posts."""
        return self._fuse({}, self._lexical(self._clauses(text)), limit)

    def extract_batch(self, texts: list[str], limit: int = 25, semantic: bool = True) -> list[list[SkillMatch]]:
        """Like ``extract`` per text, but embeds all clauses in one batched model call."""
        per_text = [self._clauses(t) for t in texts]
        if not semantic:
            return [self._fuse({}, self._lexical(c), limit) for c in per_text]
        flat = [c for cs in per_text for c in cs]
        vectors = self.embedder.embed_queries(flat) if flat else np.zeros((0, 1), dtype=np.float32)
        out, start = [], 0
        for clauses in per_text:
            vecs = vectors[start:start + len(clauses)]
            start += len(clauses)
            out.append(self._fuse(self._semantic_from_vectors(clauses, vecs), self._lexical(clauses), limit))
        return out

    def _fuse(
        self,
        semantic: dict[str, tuple[float, str]],
        lexical: dict[str, tuple[int, str]],
        limit: int,
    ) -> list[SkillMatch]:
        fused: dict[str, float] = defaultdict(float)
        for rank, sid in enumerate(sorted(semantic, key=lambda s: -semantic[s][0]), start=1):
            fused[sid] += 1 / (RRF_K + rank)
        for rank, sid in enumerate(sorted(lexical, key=lambda s: -lexical[s][0]), start=1):
            fused[sid] += 1 / (RRF_K + rank)

        matches = [
            SkillMatch(
                skill_id=sid,
                label=self._labels.get(sid, sid),
                score=score,
                similarity=semantic.get(sid, (0.0, ""))[0],
                lexical=sid in lexical,
                evidence=(lexical.get(sid) or semantic[sid])[1],
            )
            for sid, score in fused.items()
        ]
        matches.sort(key=lambda m: (-m.score, -m.similarity))
        return matches[:limit]

    def extract_many(self, texts: list[str], limit: int = 25) -> list[list[SkillMatch]]:
        return self.extract_batch(texts, limit)

    def rank_skills(self, queries: list[str], candidate_ids: list[str] | None = None, k: int = 10):
        """Top-k (skill_id, similarity) per query; optionally restricted to a subset (e.g. ESCO only).

        Used for evaluation and crosswalks, where we need ranked candidates rather than fused matches.
        """
        matrix, ids = self.index.matrix, self.index.ids
        if candidate_ids is not None:
            pos = {sid: i for i, sid in enumerate(ids)}
            rows = [pos[s] for s in candidate_ids if s in pos]
            matrix, ids = matrix[rows], [ids[r] for r in rows]
        results = []
        q = self.embedder.embed_queries(queries)
        for start in range(0, len(queries), 1024):
            sims = q[start:start + 1024] @ matrix.T
            kk = min(k, sims.shape[1])
            top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            for row_i, row in enumerate(top):
                ordered = sorted(row, key=lambda j: -sims[row_i, j])
                results.append([(ids[j], float(sims[row_i, j])) for j in ordered])
        return results
