"""Ingestion & normalization on the SAS hackathon data ONLY.

    Data parsing -> Skill extraction -> Sentence Transformers -> Semantic matching -> ESCO/O*NET mapping

Inputs : hackathon/SAS Data Problem Statement and Instructions Hackathon/{Analytics Jobs.csv, DataScience Jobs.csv}
Outputs: dataset/processed/sas/*  (read by make_sas_report.py)
Usage  : uv run python scripts/run_sas_ingestion.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from d2s.config import get_settings
from d2s.ingest import parsing, sas
from d2s.ml.embeddings import SentenceTransformerEmbedder
from d2s.ml.skills import SkillIndex, SkillMatcher
from d2s.ml.taxonomy import load_esco, load_onet
from d2s.workers.tasks import index_path

T0 = time.perf_counter()
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SAS_DIR = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    s = get_settings()
    out = s.dataset_dir / "processed" / "sas"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": s.embedding_model, "source": str(SAS_DIR.relative_to(ROOT))}

    log("STEP 1/5 data parsing & cleaning (SAS files only)")
    posts, clean = parsing.load_sas_analytics(SAS_DIR / "Analytics Jobs.csv")
    ds = parsing.load_sas_ds_jobs(SAS_DIR / "DataScience Jobs.csv")
    summary["cleaning"] = clean.as_dict()
    summary["ds_jobs"] = {"rows": len(ds), "companies": int(ds["company_name"].nunique()),
                          "jobs_total": int(ds["num_of_jobs"].sum())}
    posts.drop(columns=["text"]).to_csv(out / "posts_clean.csv.gz", index=False)
    ds.to_csv(out / "ds_jobs_clean.csv", index=False)
    log(f"analytics posts {clean.raw_rows:,} -> {clean.kept_rows:,}; ds_jobs {len(ds):,} rows")

    log("STEP 2/5 taxonomy vocabulary + sentence-transformer index")
    taxonomy = load_onet().merge(load_esco())
    skills = list(taxonomy.skills.values())
    embedder = SentenceTransformerEmbedder(s.embedding_model, s.embedding_backend, s.embedding_batch_size)
    cache = index_path(s.embedding_model, taxonomy.version, "label")
    index = SkillIndex.load(cache) if cache.exists() else None
    if index is None or index.ids != [x.id for x in skills]:
        log(f"building label index for {len(skills):,} skills")
        index = SkillIndex.build(skills, embedder, text="label")
        index.save(cache)
    log(f"index ready ({len(index.ids):,} skills, {cache.name})")
    esco_ids = {x.id for x in skills if x.source == "esco"}
    # Low threshold here on purpose: raw similarities are kept and the cut-off is calibrated below.
    matcher = SkillMatcher(skills, embedder, index=index, threshold=0.60, semantic_ids=esco_ids)

    log("STEP 3/5 ESCO/O*NET mapping of recruiter key_skills")
    tag_table, tag_matches = sas.map_tags(posts, matcher, log)
    tag_table.to_csv(out / "tag_mapping.csv", index=False)
    use = tag_table.groupby("method")["posts"].sum()
    summary["tags"] = {
        "distinct": len(tag_table),
        "distinct_by_method": tag_table["method"].value_counts().to_dict(),
        "tag_uses_by_method": {k: int(v) for k, v in use.items()},
        "posts_with_mapped_tag": int(tag_matches["post_id"].nunique()),
    }
    log(f"tags: {summary['tags']['distinct_by_method']}")

    log("STEP 4/5 skill extraction from title + description (semantic matching)")
    text = sas.text_matches(posts, matcher, log)
    threshold, sweep = sas.calibrate_threshold(text, tag_matches)
    sweep.to_csv(out / "threshold_calibration.csv", index=False)
    summary["threshold"] = threshold
    summary["calibration_at_threshold"] = sweep.set_index("threshold").loc[threshold].to_dict()
    log(f"calibrated semantic threshold = {threshold} {summary['calibration_at_threshold']}")
    text.to_csv(out / "text_matches_raw.csv.gz", index=False)

    final = sas.combine(tag_matches, text, threshold, taxonomy)
    final.to_csv(out / "post_skills.csv.gz", index=False)
    summary["post_skills"] = {
        "rows": len(final), "posts_with_skill": int(final["post_id"].nunique()),
        "by_signal": final["signals"].value_counts().to_dict(),
        "by_taxonomy": final["taxonomy"].value_counts().to_dict(),
    }
    log(f"final post-skill rows {len(final):,}: {summary['post_skills']['by_signal']}")

    log("STEP 5/5 analysis-ready tables")
    prof = sas.skill_profile(final, posts)
    prof.to_csv(out / "skill_profile.csv", index=False)
    sas.skill_city(final, posts, prof.head(15)["skill_id"].tolist()).to_csv(out / "skill_city.csv", index=False)
    by_title = ds.groupby("job_title").apply(lambda g: pd.Series({
        "jobs": int(g["num_of_jobs"].sum()), "companies": g["company_name"].nunique(),
        "avg_lakh_weighted": float((g["avg_lakh"] * g["num_of_jobs"]).sum() / g["num_of_jobs"].sum()),
        "min_lakh_p25": float(g["min_lakh"].quantile(0.25)), "max_lakh_p75": float(g["max_lakh"].quantile(0.75)),
    }), include_groups=False).sort_values("jobs", ascending=False)
    by_title.to_csv(out / "ds_jobs_by_title.csv")
    titles = by_title.index.tolist() + posts["title"].value_counts().head(25).index.tolist()
    sas.map_titles(list(dict.fromkeys(titles)), taxonomy, embedder).to_csv(out / "title_occupations.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"DONE outputs in {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
