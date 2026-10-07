"""Measure skill-matching quality on TechWolf's hand-labelled job-post sentences (CC BY 4.0).

Each sentence has one gold ESCO skill (preferred label). We rank all ESCO skills per sentence
and report hit@k / MRR, plus the accuracy-vs-coverage trade-off of the similarity threshold.
"""

import numpy as np
import pandas as pd

from d2s.ml.skills import SkillMatcher
from d2s.ml.taxonomy import Taxonomy


def techwolf_eval(df: pd.DataFrame, taxonomy: Taxonomy, matcher: SkillMatcher, k: int = 10) -> dict:
    esco = {s.id: s for s in taxonomy.skills.values() if s.source == "esco"}
    by_label = {s.label.lower(): sid for sid, s in esco.items()}
    df = df.assign(gold=df["label"].str.lower().map(by_label))
    unmatched = int(df["gold"].isna().sum())
    df = df.dropna(subset=["gold"]).reset_index(drop=True)

    sentences = df["sentence"].str.lstrip("*•- ").tolist()
    ranked = matcher.rank_skills(sentences, candidate_ids=list(esco), k=k)

    ranks, top1_sim, top1_ok, rows = [], [], [], []
    for gold, sent, cands in zip(df["gold"], sentences, ranked, strict=True):
        ids = [c[0] for c in cands]
        rank = ids.index(gold) + 1 if gold in ids else None
        ranks.append(rank)
        top1_sim.append(cands[0][1])
        top1_ok.append(rank == 1)
        rows.append({"sentence": sent, "gold": esco[gold].label, "top1": esco[ids[0]].label,
                     "top1_similarity": cands[0][1], "gold_rank": rank})

    n = len(ranks)
    hit = {f"hit@{j}": sum(1 for r in ranks if r and r <= j) / n for j in (1, 3, 5, 10) if j <= k}
    mrr = sum(1 / r for r in ranks if r) / n

    sims, ok = np.array(top1_sim), np.array(top1_ok)
    sweep = []
    for t in np.round(np.arange(0.30, 0.951, 0.025), 3):
        keep = sims >= t
        sweep.append({
            "threshold": float(t),
            "coverage": float(keep.mean()),               # share of sentences we'd output a skill for
            "precision@1": float(ok[keep].mean()) if keep.any() else float("nan"),
        })
    return {
        "n_sentences": n,
        "gold_not_in_esco": unmatched,
        **hit,
        "mrr": mrr,
        "sweep": pd.DataFrame(sweep),
        "examples": pd.DataFrame(rows),
    }
