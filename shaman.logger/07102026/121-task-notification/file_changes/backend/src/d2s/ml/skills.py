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
    def build(
        cls, skills: list[Skill], embedder: Embedder, batch: int = 512, text: str = "label"
    ) -> "SkillIndex":
        """``text="label"`` measured best on TechWolf labelled sentences (hit@5 27.2% vs 23.9%
        for label+description with bge-small), and embeds ~5x faster."""
        if text == "label":
            texts = [s.label for s in skills]
        elif text == "label+description":
            texts = [s.embedding_text for s in skills]
        else:
            raise ValueError(f"unknown index text mode {text}")
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


# O*NET's rated elements ("Design", "Science", "Programming", "Writing") describe occupations;
# as free-text words they match almost every post, so they never match lexically.
LEXICAL_EXCLUDED_KINDS = frozenset({"essential_skill", "transferable_skill", "knowledge"})
# Section headers and boilerplate that carry no skill but attract semantic matches.
_BOILERPLATE = re.compile(
    r"^(preferred candidate profile|roles? (and|&) responsibilities|job (description|summary)|"
    r"key (skills|responsibilities)|requirements?|qualifications?|about (us|the (role|company))|"
    r"what you('ll| will) do|who you are|education|experience|perks and benefits)\b",
    re.I,
)


def is_boilerplate(clause: str, min_words: int = 4) -> bool:
    return len(clause.split()) < min_words or bool(_BOILERPLATE.match(clause.strip()))


# Single-word tool names that are also everyday words; found by reviewing the most frequent
# exact matches on 88k Naukri posts (e.g. "Analyze" matched the verb in 4,484 posts).
AMBIGUOUS_WORD_LABELS = frozenset({"analyze", "act", "act!", "reduce", "google"})
_PAREN = re.compile(r"\s*\([^)]*\)\s*")


_ACRONYM = re.compile(r"^[A-Z][A-Z0-9+#/]{1,7}$")
_ACRONYM_STOP = frozenset({"it", "as", "or", "am", "pm", "us", "go", "do", "is", "in", "on", "at", "an",
                           "to", "so", "no", "me", "my", "we", "he", "up", "be", "by", "of", "all", "one"})


def _onet_acronyms(skills: list[Skill]) -> dict[str, str]:
    """O*NET tool labels often end in their acronym ("Extensible markup language XML",
    "Management information systems MIS"). Expose the acronym as a key when it is unique."""
    found: dict[str, list[str]] = defaultdict(list)
    for s in skills:
        if s.source != "onet" or s.kind != "tool":
            continue
        parts = s.label.split()
        if len(parts) >= 3 and _ACRONYM.match(parts[-1]) and _expands(parts[-1], parts[:-1]):
            key = _norm(parts[-1])
            if key and key not in _ACRONYM_STOP:
                found[key].append(s.id)
    return {k: ids[0] for k, ids in found.items() if len(set(ids)) == 1}


def _expands(acronym: str, words: list[str]) -> bool:
    """True if the acronym's letters occur in order in the preceding words ("HTML" in
    "hypertext markup language"); rejects product names that merely end in a code
    ("Airline Pilots Daily Aviation Log PPC")."""
    letters = [c for c in acronym.lower() if c.isalnum()]
    text, pos = " ".join(words).lower(), 0
    for ch in letters:
        pos = text.find(ch, pos)
        if pos < 0:
            return False
        pos += 1
    # ...and all but one letter must be word initials ("Microsoft Dynamics CRM" fails: m, d).
    initials = [w[0].lower() for w in words if w]
    return _lcs(letters, initials) >= max(2, len(letters) - 1)


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], prev + 1 if x == y else max(dp[j], dp[j - 1])
    return dp[-1]


def build_lexicon(skills: list[Skill], min_len: int = 3, umbrella_min: int = 2) -> dict[str, str]:
    """Normalised surface form -> skill id, with ambiguous forms removed.

    Preferred labels always count, except reviewed everyday-word tool names. Alternative labels
    are dropped when they are
      * a single plain lowercase word ("interaction" -> communication), or
      * an umbrella term appearing inside >= ``umbrella_min`` other labels ("SAP" -> 36 SAP
        products, ".NET" -> several .NET tools),
    unless the alt is just the preferred label without its qualifier
    ("Java" for "Java (computer programming)").
    """
    # ESCO also has skillType "knowledge" (SQL, statistics, ...) - only O*NET descriptors are excluded.
    kept = [s for s in skills if not (s.source == "onet" and s.kind in LEXICAL_EXCLUDED_KINDS)]
    token_counts: dict[str, int] = defaultdict(int)
    for s in kept:
        words = _norm(s.label).split()
        for n in range(1, min(len(words), MAX_LABEL_WORDS) + 1):
            for gram in {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}:
                token_counts[gram] += 1

    lexicon: dict[str, str] = {}
    for s in kept:  # preferred labels first, so they win over any alt label
        key = _norm(s.label)
        if key in AMBIGUOUS_WORD_LABELS:
            continue
        if len(key) >= min_len and len(key.split()) <= MAX_LABEL_WORDS:
            lexicon.setdefault(key, s.id)
    for key, sid in _onet_acronyms(kept).items():  # "HTML" -> "Hypertext markup language HTML"
        lexicon.setdefault(key, sid)
    for s in kept:
        unqualified = _norm(_PAREN.sub(" ", s.label))
        for alt in s.alt_labels:
            key = _norm(alt)
            if len(key) < min_len or len(key.split()) > MAX_LABEL_WORDS or key in AMBIGUOUS_WORD_LABELS:
                continue
            if " " not in key and alt.isalpha() and alt.islower():
                continue
            if key != unqualified and token_counts.get(key, 0) >= umbrella_min:
                continue
            lexicon.setdefault(key, s.id)
    return lexicon


class SkillMatcher:
    def __init__(
        self,
        skills: list[Skill],
        embedder: Embedder,
        index: SkillIndex | None = None,
        threshold: float = 0.6,
        top_k_per_clause: int = 3,
        min_lexical_len: int = 3,
        semantic_ids: set[str] | None = None,
        skip_boilerplate: bool = True,
    ):
        """``semantic_ids`` restricts which skills semantic search may return (e.g. ESCO only;
        tool names are better matched lexically)."""
        self.embedder = embedder
        self.index = index or SkillIndex.build(skills, embedder)
        if self.index.model_name != embedder.name:
            raise ValueError(f"index built with {self.index.model_name}, embedder is {embedder.name}")
        self.threshold = threshold
        self.top_k = top_k_per_clause
        self.skip_boilerplate = skip_boilerplate
        self._labels = dict(zip(self.index.ids, self.index.labels, strict=True))
        self._lexicon = build_lexicon(skills, min_lexical_len)
        if semantic_ids is None:
            self._sem_ids, self._sem_matrix = self.index.ids, self.index.matrix
        else:
            rows = [i for i, sid in enumerate(self.index.ids) if sid in semantic_ids]
            self._sem_ids = [self.index.ids[i] for i in rows]
            self._sem_matrix = self.index.matrix[rows]

    def _semantic_clauses(self, clauses: list[str]) -> list[str]:
        return [c for c in clauses if not is_boilerplate(c)] if self.skip_boilerplate else clauses

    def _semantic(self, clauses: list[str]) -> dict[str, tuple[float, str]]:
        clauses = self._semantic_clauses(clauses)
        if not clauses or not len(self._sem_ids):
            return {}
        return self._semantic_from_vectors(clauses, self.embedder.embed_queries(clauses))

    def _semantic_from_vectors(self, clauses: list[str], vectors: np.ndarray) -> dict[str, tuple[float, str]]:
        if not clauses or not len(self._sem_ids):
            return {}
        sims = vectors @ self._sem_matrix.T
        k = min(self.top_k, sims.shape[1])
        best: dict[str, tuple[float, str]] = {}
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for ci, row in enumerate(top):
            for j in row:
                sim = float(sims[ci, j])
                if sim < self.threshold:
                    continue
                sid = self._sem_ids[j]
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

    def longest_contained(self, text: str) -> str | None:
        """Longest known skill name occurring inside a short phrase ("Core Java" -> Java)."""
        words = _norm(text).split()
        for n in range(min(len(words), MAX_LABEL_WORDS), 0, -1):
            for i in range(len(words) - n + 1):
                sid = self._lexicon.get(" ".join(words[i:i + n]))
                if sid:
                    return sid
        return None

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
        per_sem = [self._semantic_clauses(c) for c in per_text]
        flat = [c for cs in per_sem for c in cs]
        vectors = self.embedder.embed_queries(flat) if flat else np.zeros((0, 1), dtype=np.float32)
        out, start = [], 0
        for clauses, sem in zip(per_text, per_sem, strict=True):
            vecs = vectors[start:start + len(sem)]
            start += len(sem)
            out.append(self._fuse(self._semantic_from_vectors(sem, vecs), self._lexical(clauses), limit))
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
