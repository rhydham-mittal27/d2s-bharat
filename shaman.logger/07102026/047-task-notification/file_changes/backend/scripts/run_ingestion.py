"""Run the ingestion & normalization pipeline end to end on real data.

    Data parsing -> Skill extraction -> Sentence Transformers -> Semantic matching -> ESCO/O*NET mapping

Usage:  uv run python scripts/run_ingestion.py [--semantic-sample 3000] [--threshold 0.6]
Outputs: dataset/processed/ingest/*.csv.gz + summary.json (read by make_ingestion_report.py)
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from d2s.config import get_settings
from d2s.ingest import crosswalk, evaluate, parsing, pipeline
from d2s.ml.embeddings import SentenceTransformerEmbedder
from d2s.ml.skills import SkillIndex, SkillMatcher
from d2s.ml.taxonomy import load_esco, load_onet

T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-sample", type=int, default=3000, help="posts for full semantic matching")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--model", default=get_settings().embedding_model)
    args = ap.parse_args()

    s = get_settings()
    data = s.dataset_dir
    out = data / "processed" / "ingest"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": args.model, "threshold": args.threshold}

    # 1. Data parsing ------------------------------------------------------------------------
    log("STEP 1/5 data parsing")
    files = sorted((data / "jobs" / "naukri").glob("naukri_*.jsonl"))
    posts, stats = parsing.load_naukri(files)
    summary["parse"] = stats.as_dict()
    summary["segments"] = posts["segment"].value_counts().to_dict()
    log(f"parsed {stats.raw:,} raw -> {stats.kept:,} clean posts {stats.as_dict()}")
    posts.drop(columns=["text"]).to_csv(out / "posts.csv.gz", index=False)

    # 2-3. Taxonomy + Sentence-Transformer index ------------------------------------------------
    log("STEP 2/5 taxonomy + sentence-transformer embeddings")
    taxonomy = load_onet().merge(load_esco())
    skills = list(taxonomy.skills.values())
    log(f"taxonomy {taxonomy.version}: {len(skills):,} skills")
    embedder = SentenceTransformerEmbedder(args.model, s.embedding_backend, s.embedding_batch_size)
    log(f"model {args.model} loaded (dim={embedder.dim})")
    cache = data / "index" / f"{args.model.replace('/', '__')}__{taxonomy.version}.npz"
    index = SkillIndex.load(cache) if cache.exists() else None
    if index is None or index.ids != [x.id for x in skills]:
        log(f"embedding {len(skills):,} skill descriptions (one-off, cached afterwards)")
        t = time.perf_counter()
        index = SkillIndex.build(skills, embedder)
        index.save(cache)
        log(f"index built in {time.perf_counter() - t:.0f}s -> {cache.name}")
    else:
        log(f"index loaded from cache {cache.name}")
    matcher = SkillMatcher(skills, embedder, index=index, threshold=args.threshold)

    # 4a. Evaluation on hand-labelled data (sets expectations for the threshold) -----------------
    log("STEP 3/5 evaluation on TechWolf labelled sentences")
    tw = pd.read_csv(data / "eval" / "techwolf" / "test.csv")
    ev = evaluate.techwolf_eval(tw, taxonomy, matcher)
    ev["sweep"].to_csv(out / "eval_threshold_sweep.csv", index=False)
    ev["examples"].to_csv(out / "eval_examples.csv", index=False)
    summary["eval"] = {k: v for k, v in ev.items() if not isinstance(v, pd.DataFrame)}
    log(f"TechWolf: {summary['eval']}")

    # 4b. Skill extraction: lexical on all posts, hybrid semantic on a stratified sample ----------
    log("STEP 4/5 skill extraction + semantic matching")
    lex = pipeline.extract_matches(posts, matcher, semantic=False, log=log)
    lex = pipeline.annotate(lex, taxonomy)
    lex.to_csv(out / "matches_lexical_all.csv.gz", index=False)
    summary["lexical_all"] = {"posts": len(posts), "matches": len(lex),
                              "posts_with_match": int(lex["post_id"].nunique())}

    sample = parsing.stratified_sample(posts, args.semantic_sample)
    log(f"semantic sample: {len(sample):,} posts {sample['segment'].value_counts().to_dict()}")
    hyb = pipeline.extract_matches(sample, matcher, semantic=True, log=log)
    hyb = pipeline.annotate(hyb, taxonomy)
    hyb.to_csv(out / "matches_hybrid_sample.csv.gz", index=False)
    summary["hybrid_sample"] = {
        "posts": len(sample), "matches": len(hyb),
        "match_type": hyb["match_type"].value_counts().to_dict(),
        "taxonomy": hyb["taxonomy"].value_counts().to_dict(),
        "tag_recall": pipeline.tag_recall(sample, hyb, matcher),
        "tag_recall_lexical_only": pipeline.tag_recall(
            sample, lex[lex["post_id"].isin(sample["post_id"])], matcher),
    }
    pipeline.skill_demand(hyb, sample).to_csv(out / "skill_demand_sample.csv", index=False)
    demand_all = pipeline.skill_demand(lex, posts)
    demand_all.to_csv(out / "skill_demand_lexical_all.csv", index=False)
    top = demand_all.head(12)["skill_id"].tolist()
    pipeline.demand_by_city(lex, posts, top).to_csv(out / "demand_by_city.csv", index=False)
    log(f"hybrid sample: {summary['hybrid_sample']['match_type']}")

    # 5. ESCO/O*NET mapping: recruiter tags + taxonomy crosswalk --------------------------------
    log("STEP 5/5 ESCO/O*NET mapping")
    tags = pipeline.map_tags(posts, matcher, log=log)
    tags.to_csv(out / "tag_mapping.csv", index=False)
    summary["tags"] = {"distinct_mapped": len(tags),
                       "exact": int(tags["exact_skill_id"].notna().sum())}
    cw = crosswalk.esco_onet_crosswalk(taxonomy, matcher)
    cw.to_csv(out / "crosswalk_onet_esco.csv", index=False)
    summary["crosswalk"] = cw["decision"].value_counts().to_dict()
    log(f"crosswalk O*NET->ESCO: {summary['crosswalk']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"DONE outputs in {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # make failures visible to the progress monitor
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
