"""Steps 2-5 - skill extraction -> Sentence-Transformer embeddings -> semantic matching ->
ESCO/O*NET mapping, plus the aggregates the forecaster and dashboards consume.
"""

import time
from collections import Counter
from collections.abc import Callable

import pandas as pd

from d2s.ml.skills import SkillMatcher
from d2s.ml.taxonomy import Taxonomy

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def extract_matches(
    posts: pd.DataFrame,
    matcher: SkillMatcher,
    semantic: bool,
    batch_posts: int = 64,
    limit: int = 30,
    log: Log = _noop,
) -> pd.DataFrame:
    """One row per (post, skill) with the match evidence, for every post in ``posts``."""
    rows = []
    started = time.perf_counter()
    texts, ids = posts["text"].tolist(), posts["post_id"].tolist()
    step = batch_posts if semantic else 2_000
    for start in range(0, len(texts), step):
        chunk = texts[start:start + step]
        results = matcher.extract_batch(chunk, limit=limit, semantic=semantic)
        for pid, matches in zip(ids[start:start + step], results, strict=True):
            for m in matches:
                rows.append((pid, m.skill_id, m.label, m.score, m.similarity, m.lexical, m.evidence))
        done = min(start + step, len(texts))
        if done == len(texts) or (done // step) % max(1, 1_000 // step) == 0:
            rate = done / max(time.perf_counter() - started, 1e-9)
            log(f"{'semantic' if semantic else 'lexical'} extraction: {done:,}/{len(texts):,} posts "
                f"({rate:,.0f} posts/s)")
    return pd.DataFrame(rows, columns=["post_id", "skill_id", "label", "score", "similarity",
                                       "lexical", "evidence"])


def match_type(df: pd.DataFrame) -> pd.Series:
    sem = df["similarity"] > 0
    return pd.Series(
        ["both" if s and lex else "semantic" if s else "lexical"
         for s, lex in zip(sem, df["lexical"], strict=True)],
        index=df.index,
    )


def annotate(matches: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    skills = taxonomy.skills
    out = matches.copy()
    out["taxonomy"] = out["skill_id"].map(lambda s: skills[s].source if s in skills else "?")
    out["kind"] = out["skill_id"].map(lambda s: skills[s].kind if s in skills else "?")
    out["hot_technology"] = out["skill_id"].map(lambda s: bool(s in skills and skills[s].hot_technology))
    out["match_type"] = match_type(out)
    return out


def skill_demand(matches: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """Share of posts mentioning each skill (the demand signal per skill)."""
    n_posts = posts["post_id"].nunique()
    g = matches.groupby(["skill_id", "label", "taxonomy", "kind", "hot_technology"], as_index=False)
    out = g.agg(posts=("post_id", "nunique"), mean_similarity=("similarity", "mean"))
    out["share_of_posts"] = out["posts"] / n_posts
    return out.sort_values("posts", ascending=False).reset_index(drop=True)


def demand_by_city(matches: pd.DataFrame, posts: pd.DataFrame, top_skills: list[str]) -> pd.DataFrame:
    m = matches[matches["skill_id"].isin(top_skills)].merge(posts[["post_id", "city"]], on="post_id")
    return m.groupby(["city", "label"]).post_id.nunique().rename("posts").reset_index()


def map_tags(posts: pd.DataFrame, matcher: SkillMatcher, min_count: int = 5, log: Log = _noop) -> pd.DataFrame:
    """Map recruiter tags (Naukri `tagsAndSkills`) onto the taxonomy: exact label first,
    otherwise the nearest skill by embedding similarity. Tags are a free, noisy, human signal."""
    counts = Counter(t for tags in posts["tags"] for t in tags)
    tags = [t for t, c in counts.items() if c >= min_count]
    log(f"mapping {len(tags):,} distinct recruiter tags (used ≥{min_count}×)")
    exact = {t: matcher.lookup(t) for t in tags}
    ranked = matcher.rank_skills(tags, k=1)
    rows = []
    for t, best in zip(tags, ranked, strict=True):
        sid, sim = best[0]
        rows.append({
            "tag": t, "count": counts[t],
            "exact_skill_id": exact[t],
            "nearest_skill_id": sid, "nearest_label": matcher.label_of(sid), "similarity": sim,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def tag_recall(posts: pd.DataFrame, matches: pd.DataFrame, matcher: SkillMatcher) -> dict:
    """Of recruiter tags that map exactly to a taxonomy skill, how many did we extract from the JD text?"""
    found = matches.groupby("post_id")["skill_id"].agg(set).to_dict()
    mappable = recovered = total = 0
    for pid, tags in zip(posts["post_id"], posts["tags"], strict=True):
        for t in tags:
            total += 1
            sid = matcher.lookup(t)
            if sid:
                mappable += 1
                recovered += sid in found.get(pid, set())
    return {
        "tags_total": total,
        "tags_exact_in_taxonomy": mappable,
        "tags_exact_in_taxonomy_pct": 100 * mappable / total if total else 0.0,
        "recovered_from_text_pct": 100 * recovered / mappable if mappable else 0.0,
    }
