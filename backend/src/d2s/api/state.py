"""Process-wide state loaded once at start-up: trained model, evidence tables, skill lookup.

The skill lookup needs the sentence-transformer (~15 s to load), so it is warmed in a background
thread; endpoints that need it answer 503 "warming up" until it is ready.
"""

import logging
import threading
from dataclasses import dataclass, field

import pandas as pd

from d2s.analysis.traits import load_jds
from d2s.models.hike import HikeModel
from d2s.services import cohort, store

log = logging.getLogger("d2s.api")


@dataclass
class AppState:
    model: HikeModel | None = None
    area_roles: pd.DataFrame | None = None
    catalogue: pd.DataFrame | None = None
    tri: pd.DataFrame | None = None
    sample_jds: pd.DataFrame | None = None
    sample_gaps: pd.DataFrame | None = None
    lookup: object | None = None
    matcher: object | None = None
    vectors: object | None = None  # d2s.vector.SkillVectorStore (ChromaDB)
    vector_error: str | None = None
    lookup_error: str | None = None
    load_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        try:
            self.model = HikeModel.load(store.artifact_path("hike_model.pkl"))
            self.area_roles = store.artifact_table("area_roles.csv")
            self.catalogue = pd.read_csv(store.CATALOGUE)
            self.tri = store.table("rq1/triangulation_rq1_rq2.csv")
            jds = load_jds(store.SAS_DIR / "JDS Skill Traits.xlsx").frame
            self.sample_jds = jds
            prof = cohort.profile(jds.drop(columns="salary_hike_high_or_low"), self.model,
                                  self.area_roles, self.catalogue)
            self.sample_gaps = cohort.gaps_frame(prof)
        except Exception as exc:  # surfaced via /api/health instead of crashing the server
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("start-up load failed")

    def warm_lookup(self) -> None:
        def run():
            try:
                from d2s.ml.embeddings import get_embedder
                from d2s.services.market import SkillLookup

                lk = SkillLookup(get_embedder())
                with self._lock:
                    self.lookup = lk
            except Exception as exc:
                self.lookup_error = f"{type(exc).__name__}: {exc}"
                log.exception("skill lookup warm-up failed")
                return
            self._open_vectors(lk.embedder)

        threading.Thread(target=run, name="warm-skill-lookup", daemon=True).start()

    def _open_vectors(self, embedder) -> None:
        """Open the ChromaDB skill collection; build it on first start (a few seconds from the cached
        embeddings). On failure search falls back to the in-memory index."""
        try:
            from d2s.vector import SkillVectorStore, build_skill_index, chroma_client

            vs = SkillVectorStore(chroma_client(), embedder.name)
            if vs.count() == 0:
                build_skill_index(vs, embedder)
            self.vectors = vs
        except Exception as exc:
            self.vector_error = f"{type(exc).__name__}: {exc}"
            log.exception("chroma vector store unavailable; using the in-memory index")


STATE = AppState()
