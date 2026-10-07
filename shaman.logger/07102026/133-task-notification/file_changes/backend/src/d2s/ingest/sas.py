"""SAS hackathon ingestion: map recruiter key-skills and job text onto ESCO/O*NET, calibrate the
semantic threshold *inside* the SAS data, and build analysis-ready skill x salary/city tables.

No external labelled data is used: recruiter `key_skills` tags act as weak in-dataset labels.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from d2s.ml.embeddings import Embedder
from d2s.ml.skills import SkillMatcher, _norm
from d2s.ml.taxonomy import Taxonomy

Log = Callable[[str], None]

# Tag -> skill resolution order: exact label/acronym -> longest known skill name *inside* the tag
# ("Core Java" -> Java) -> nearest embedding. Embedding matches are accepted only above TAG_HIGH
# (TAG_HIGH_SINGLE for one-word tags, which otherwise land on over-specific skills, e.g.
# "Operations" -> "maintenance operations" at 0.88). The 0.78-band goes to a review queue:
# a spot check showed errors like "HTML" -> PHP and "Core Java" -> "core apples".
TAG_HIGH = 0.85
TAG_HIGH_SINGLE = 0.90
TAG_MEDIUM = 0.78


def map_tags(posts: pd.DataFrame, matcher: SkillMatcher, log: Log = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map every distinct recruiter tag once, then expand to (post, skill) rows.

    Returns (tag_table, post_tag_matches).
    """
    exploded = posts[["post_id", "tags"]].explode("tags").dropna(subset=["tags"])
    exploded["key"] = exploded["tags"].map(_norm)
    exploded = exploded[exploded["key"] != ""]
    counts = exploded.groupby("key").agg(tag=("tags", "first"), posts=("post_id", "nunique"))
    keys = counts.index.tolist()
    log(f"mapping {len(keys):,} distinct recruiter tags ({len(exploded):,} tag uses)")

    exact = {k: matcher.lookup(counts.at[k, "tag"]) for k in keys}
    contains = {k: matcher.longest_contained(counts.at[k, "tag"]) for k in keys if not exact[k]}
    pending = [k for k in keys if not exact[k] and not contains.get(k)]
    ranked = matcher.rank_skills([counts.at[k, "tag"] for k in pending], k=2) if pending else []
    nearest = dict(zip(pending, ranked, strict=True))

    rows = []
    for k in keys:
        runner = None
        if exact[k]:
            sid, sim, method = exact[k], 1.0, "exact"
        elif contains.get(k):
            sid, sim, method = contains[k], 1.0, "contains"
        else:
            (sid, sim), runner = nearest[k][0], (nearest[k][1] if len(nearest[k]) > 1 else None)
            bar = TAG_HIGH_SINGLE if " " not in k else TAG_HIGH
            method = "semantic" if sim >= bar else "review" if sim >= TAG_MEDIUM else "unmapped"
        rows.append({
            "key": k, "tag": counts.at[k, "tag"], "posts": int(counts.at[k, "posts"]),
            "skill_id": sid if method in ("exact", "contains", "semantic") else None,
            "nearest_skill_id": sid, "nearest_label": matcher.label_of(sid), "similarity": sim,
            "runner_up_label": matcher.label_of(runner[0]) if runner else "", "method": method,
        })
    table = pd.DataFrame(rows).sort_values("posts", ascending=False).reset_index(drop=True)

    mapped = exploded.merge(table[table["skill_id"].notna()][["key", "skill_id", "similarity", "method"]], on="key")
    matches = mapped.rename(columns={"tags": "evidence"})[["post_id", "skill_id", "similarity", "method", "evidence"]]
    matches["label"] = matches["skill_id"].map(matcher.label_of)
    matches = matches.drop_duplicates(["post_id", "skill_id"])
    return table, matches


def text_matches(posts: pd.DataFrame, matcher: SkillMatcher, log: Log = print, batch: int = 256) -> pd.DataFrame:
    """Hybrid extraction from title + (truncated) description, keeping raw similarities so the
    threshold can be calibrated afterwards without re-embedding."""
    rows = []
    texts, ids = posts["text"].tolist(), posts["post_id"].tolist()
    for start in range(0, len(texts), batch):
        res = matcher.extract_batch(texts[start:start + batch], limit=30)
        for pid, ms in zip(ids[start:start + batch], res, strict=True):
            rows += [(pid, m.skill_id, m.label, m.similarity, m.lexical, m.evidence) for m in ms]
        if (start // batch) % 10 == 0 or start + batch >= len(texts):
            log(f"text extraction: {min(start + batch, len(texts)):,}/{len(texts):,} posts")
    df = pd.DataFrame(rows, columns=["post_id", "skill_id", "label", "similarity", "lexical", "evidence"])
    df["method"] = np.where(df["lexical"] & (df["similarity"] > 0), "both",
                            np.where(df["lexical"], "lexical", "semantic"))
    return df


def calibrate_threshold(text: pd.DataFrame, tags: pd.DataFrame, grid=None) -> tuple[float, pd.DataFrame]:
    """Pick the semantic threshold that best agrees with recruiters' own key-skills.

    precision = share of semantic text matches that are also among the post's tag skills
    recall    = share of tag skills that the semantic text matching recovers
    Only posts that have both text and mapped tags are scored. Weak labels: recruiters omit
    many skills, so absolute precision is a lower bound; the *shape* picks the threshold.
    """
    grid = grid if grid is not None else np.round(np.arange(0.60, 0.901, 0.01), 2)
    tag_sets = tags.groupby("post_id")["skill_id"].agg(set)
    sem = text[text["method"] != "lexical"]
    sem = sem[sem["post_id"].isin(tag_sets.index)]
    n_tag = int(tag_sets[tag_sets.index.isin(sem["post_id"].unique())].map(len).sum())
    hit = np.array([sid in tag_sets.get(pid, ()) for pid, sid in zip(sem["post_id"], sem["skill_id"], strict=True)])
    sims = sem["similarity"].to_numpy()
    rows = []
    for t in grid:
        keep = sims >= t
        tp = int(hit[keep].sum())
        p = tp / keep.sum() if keep.any() else float("nan")
        r = tp / n_tag if n_tag else float("nan")
        f1 = 2 * p * r / (p + r) if p and r and (p + r) else 0.0
        rows.append({"threshold": float(t), "matches_kept": int(keep.sum()), "precision": p, "recall": r, "f1": f1})
    sweep = pd.DataFrame(rows)
    best = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])
    return best, sweep


MIN_TEXT_SEMANTIC_F1 = 0.20  # below this, semantic text matches are noise vs recruiter tags


def combine(
    tags: pd.DataFrame, text: pd.DataFrame, threshold: float | None, taxonomy: Taxonomy
) -> pd.DataFrame:
    """Final (post, skill) table: tag matches + exact-name text matches, plus semantic text
    matches above ``threshold`` (pass None to exclude semantic text matches entirely)."""
    t = tags.assign(signal="key_skills")
    keep = text["method"] != "semantic"
    if threshold is not None:
        keep |= text["similarity"] >= threshold
    x = text[keep].assign(signal="text")
    cols = ["post_id", "skill_id", "label", "similarity", "method", "evidence", "signal"]
    both = pd.concat([t[cols], x[cols]], ignore_index=True)
    agg = both.groupby(["post_id", "skill_id"], as_index=False).agg(
        label=("label", "first"), similarity=("similarity", "max"),
        signals=("signal", lambda s: "+".join(sorted(set(s)))), evidence=("evidence", "first"),
        method=("method", "first"),
    )
    sk = taxonomy.skills
    agg["taxonomy"] = agg["skill_id"].map(lambda s: sk[s].source if s in sk else "?")
    agg["kind"] = agg["skill_id"].map(lambda s: sk[s].kind if s in sk else "?")
    agg["hot_technology"] = agg["skill_id"].map(lambda s: bool(s in sk and sk[s].hot_technology))
    return agg


def skill_profile(matches: pd.DataFrame, posts: pd.DataFrame, min_posts: int = 40) -> pd.DataFrame:
    """Per skill: demand (share of posts), salary-band midpoint and experience, for analysis."""
    m = matches.merge(posts[["post_id", "sal_mid_lakh", "exp_min", "city"]], on="post_id")
    n = posts["post_id"].nunique()
    overall = posts["sal_mid_lakh"].median()
    g = m.groupby(["skill_id", "label", "taxonomy", "hot_technology"], as_index=False).agg(
        posts=("post_id", "nunique"),
        salary_median_lakh=("sal_mid_lakh", "median"),
        salary_mean_lakh=("sal_mid_lakh", "mean"),
        exp_min_median=("exp_min", "median"),
        high_band_share=("sal_mid_lakh", lambda s: float((s >= 20).mean())),
    )
    g["share_of_posts"] = g["posts"] / n
    g["salary_premium_lakh"] = g["salary_median_lakh"] - overall
    return g[g["posts"] >= min_posts].sort_values("posts", ascending=False).reset_index(drop=True)


def skill_city(matches: pd.DataFrame, posts: pd.DataFrame, top_skills: list[str], top_cities: int = 10) -> pd.DataFrame:
    loc = posts[["post_id", "locations"]].explode("locations").rename(columns={"locations": "city"})
    cities = loc["city"].value_counts().head(top_cities).index
    m = matches[matches["skill_id"].isin(top_skills)].merge(loc[loc["city"].isin(cities)], on="post_id")
    return m.groupby(["city", "label"]).post_id.nunique().rename("posts").reset_index()


def map_titles(titles: list[str], taxonomy: Taxonomy, embedder: Embedder, k: int = 3) -> pd.DataFrame:
    """Job titles -> nearest ESCO/O*NET occupations (occupation titles embedded on the fly)."""
    occs = list(taxonomy.occupations.values())
    occ_vecs = embedder.embed_documents([o.title for o in occs])
    q = embedder.embed_queries(titles)
    sims = q @ occ_vecs.T
    rows = []
    for i, t in enumerate(titles):
        top = np.argsort(-sims[i])[:k]
        rows.append({"title": t, **{
            f"occupation_{j + 1}": f"{occs[o].title} ({occs[o].source})" for j, o in enumerate(top)},
            "similarity_1": float(sims[i, top[0]])})
    return pd.DataFrame(rows)
