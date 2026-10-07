"""ESCO <-> O*NET crosswalk by embedding similarity (one unified skill backbone).

High-similarity pairs are auto-accepted; the middle band goes to the human review queue;
low ones stay unlinked. Thresholds are starting points to tune on reviewed samples.
"""

import pandas as pd

from d2s.ml.skills import SkillMatcher
from d2s.ml.taxonomy import Taxonomy

AUTO_ACCEPT = 0.85
REVIEW = 0.70


def esco_onet_crosswalk(
    taxonomy: Taxonomy, matcher: SkillMatcher, include_tools: bool = True
) -> pd.DataFrame:
    esco_ids = [s.id for s in taxonomy.skills.values() if s.source == "esco"]
    onet = [
        s for s in taxonomy.skills.values()
        if s.source == "onet" and (include_tools or s.kind != "tool")
    ]
    ranked = matcher.rank_skills([s.embedding_text for s in onet], candidate_ids=esco_ids, k=3)
    rows = []
    for s, cands in zip(onet, ranked, strict=True):
        (best_id, best_sim), runner = cands[0], cands[1] if len(cands) > 1 else (None, 0.0)
        rows.append({
            "onet_id": s.id, "onet_label": s.label, "onet_kind": s.kind,
            "esco_id": best_id, "esco_label": taxonomy.skills[best_id].label, "similarity": best_sim,
            "runner_up_label": taxonomy.skills[runner[0]].label if runner[0] else "",
            "margin": best_sim - runner[1],
        })
    df = pd.DataFrame(rows)
    df["decision"] = pd.cut(
        df["similarity"], [-1, REVIEW, AUTO_ACCEPT, 1.01],
        labels=["unlinked", "review", "auto_accept"], right=False,
    )
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)
