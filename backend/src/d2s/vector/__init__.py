"""Vector store: ChromaDB holds the skill embeddings used for meaning-based skill search.

Relational data (users, courses, cohorts, plan history) stays in Postgres (``d2s.db``); vectors live
here. By default Chroma runs embedded and persists to ``<dataset>/chroma``; set ``D2S_CHROMA_HOST``
to use a Chroma server instead.
"""

from d2s.vector.store import SkillVectorStore, build_skill_index, chroma_client

__all__ = ["SkillVectorStore", "build_skill_index", "chroma_client"]
