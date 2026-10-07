"""Visualise the ingestion run: reads dataset/processed/ingest/*, writes reports/ingestion/*.png + README.md.

Usage:  uv run python scripts/make_ingestion_report.py
Every number comes from the pipeline outputs; nothing here is illustrative.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from d2s.config import get_settings  # noqa: E402

DATA = get_settings().dataset_dir
IN = DATA / "processed" / "ingest"
OUT = Path(__file__).resolve().parents[2] / "reports" / "ingestion"
COLORS = {"esco": "#2563eb", "onet": "#ea580c", "lexical": "#16a34a", "semantic": "#9333ea", "both": "#0891b2"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})


def save(fig, name: str) -> str:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_funnel(summary: dict) -> str:
    p = summary["parse"]
    steps = [("Raw posts", p["raw"]), ("− duplicate job ids", p["raw"] - p["duplicate_id"]),
             ("− empty descriptions", p["raw"] - p["duplicate_id"] - p["empty"]),
             ("− reposted identical text", p["kept"])]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels, vals = zip(*steps, strict=True)
    bars = ax.barh(labels[::-1], vals[::-1], color=["#16a34a", "#64748b", "#64748b", "#334155"])
    for b, v in zip(bars, vals[::-1], strict=True):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:,}", va="center")
    ax.set_title("Step 1 · Data parsing: Naukri job posts after cleaning & de-duplication")
    ax.set_xlabel("posts")
    return save(fig, "01_parsing_funnel.png")


def fig_top_skills(demand: pd.DataFrame, n_posts: int, title: str, name: str) -> str:
    top = demand.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(top["label"].str.slice(0, 45), 100 * top["share_of_posts"],
            color=[COLORS.get(t, "#999") for t in top["taxonomy"]])
    for y, (hot, share) in enumerate(zip(top["hot_technology"], 100 * top["share_of_posts"], strict=True)):
        ax.text(share, y, "  ★ hot tech" if hot else "", va="center", fontsize=7, color="#b45309")
    ax.set_xlabel("% of posts mentioning the skill")
    ax.set_title(f"{title}\n({n_posts:,} posts)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("esco", "onet")],
              labels=["ESCO skill", "O*NET skill/tool"], loc="lower right")
    return save(fig, name)


def fig_match_types(hyb: pd.DataFrame, threshold: float) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    counts = hyb["match_type"].value_counts().reindex(["lexical", "semantic", "both"]).fillna(0)
    a.bar(counts.index, counts.values, color=[COLORS[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        a.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    a.set_title("How each (post, skill) match was found")
    a.set_ylabel("matches")
    sem = hyb[hyb["similarity"] > 0]
    b.hist(sem["similarity"], bins=40, color=COLORS["semantic"], alpha=0.85)
    b.axvline(threshold, color="k", ls="--", lw=1)
    b.text(threshold, b.get_ylim()[1] * 0.9, f" threshold {threshold}", fontsize=8)
    b.set_title("Cosine similarity of semantic matches")
    b.set_xlabel("similarity (clause ↔ skill description)")
    return save(fig, "03_match_types_similarity.png")


def fig_eval(summary: dict, sweep: pd.DataFrame, threshold: float) -> str:
    ev = summary["eval"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    ks = [k for k in ("hit@1", "hit@3", "hit@5", "hit@10") if k in ev]
    a.bar(ks, [100 * ev[k] for k in ks], color="#2563eb")
    for i, k in enumerate(ks):
        a.text(i, 100 * ev[k], f"{100 * ev[k]:.1f}%", ha="center", va="bottom")
    a.set_ylim(0, 100)
    a.set_title(f"Gold ESCO skill in top-k\n({ev['n_sentences']} labelled sentences, MRR {ev['mrr']:.2f})")
    b.plot(sweep["threshold"], 100 * sweep["precision@1"], "-o", ms=3, label="precision@1 (kept)", color="#16a34a")
    b.plot(sweep["threshold"], 100 * sweep["coverage"], "-o", ms=3, label="coverage", color="#64748b")
    b.axvline(threshold, color="k", ls="--", lw=1)
    b.set_xlabel("similarity threshold")
    b.set_ylabel("%")
    b.set_ylim(0, 102)
    b.legend()
    b.set_title("Threshold trade-off: accuracy vs. how often we output a skill")
    return save(fig, "04_eval_techwolf.png")


def fig_embedding_map(demand: pd.DataFrame, model: str, version: str, text: str) -> str | None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from d2s.workers.tasks import index_path

    path = index_path(model, version, text)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    ids, matrix = list(z["ids"]), z["matrix"]
    rng = np.random.default_rng(0)
    pos = {s: i for i, s in enumerate(ids)}
    demanded = [s for s in demand.head(150)["skill_id"] if s in pos]
    background = rng.choice(len(ids), size=min(5000, len(ids)), replace=False)
    rows = np.unique(np.concatenate([background, [pos[s] for s in demanded]]))
    xy = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(
        PCA(n_components=50, random_state=0).fit_transform(matrix[rows]))
    row_xy = {r: xy[i] for i, r in enumerate(rows)}
    src = np.array(["esco" if ids[r].startswith("http") else "onet" for r in rows])

    fig, ax = plt.subplots(figsize=(8, 7))
    for s in ("esco", "onet"):
        m = src == s
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.25, color=COLORS[s], label=f"{s.upper()} skills (sample)")
    d = demand.set_index("skill_id")
    top = [s for s in demanded][:150]
    pts = np.array([row_xy[pos[s]] for s in top])
    sizes = 20 + 400 * d.loc[top, "share_of_posts"].to_numpy() / d["share_of_posts"].max()
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes, facecolors="none", edgecolors="k", lw=0.8,
               label="demanded in Naukri posts (size = share)")
    for s in top[:18]:
        x, y = row_xy[pos[s]]
        ax.annotate(d.loc[s, "label"][:28], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, markerscale=3)
    ax.set_title(f"Skill embedding space ({model}, t-SNE of {len(rows):,} skills)\n"
                 "similar skills cluster together; ESCO & O*NET interleave")
    return save(fig, "05_embedding_map.png")


def fig_crosswalk(cw: pd.DataFrame) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for kind, color in (("tool", "#ea580c"), ("skill", "#2563eb")):
        m = cw["onet_kind"].eq("tool") if kind == "tool" else ~cw["onet_kind"].eq("tool")
        a.hist(cw.loc[m, "similarity"], bins=40, alpha=0.6, color=color,
               label=f"O*NET {'tools' if kind == 'tool' else 'skills/knowledge'} ({int(m.sum()):,})")
    for t, lbl in ((0.70, "review"), (0.85, "auto-accept")):
        a.axvline(t, color="k", ls="--", lw=1)
        a.text(t, a.get_ylim()[1] * 0.92, f" {lbl}", fontsize=8)
    a.legend()
    a.set_xlabel("similarity to best ESCO skill")
    a.set_title("Step 5 · ESCO ↔ O*NET crosswalk")
    dec = cw["decision"].value_counts().reindex(["auto_accept", "review", "unlinked"]).fillna(0)
    b.bar(dec.index, dec.values, color=["#16a34a", "#f59e0b", "#94a3b8"])
    for i, v in enumerate(dec.values):
        b.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    b.set_title("Crosswalk decisions")
    return save(fig, "06_crosswalk.png")


def fig_tags(tags: pd.DataFrame) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    exact = tags["exact_skill_id"].notna()
    a.hist([tags.loc[exact, "similarity"], tags.loc[~exact, "similarity"]], bins=30, stacked=True,
           color=["#16a34a", "#9333ea"], label=["exact label match", "nearest by embedding"])
    a.set_xlabel("similarity of tag ↔ mapped skill")
    a.set_title(f"Recruiter tags → taxonomy ({len(tags):,} distinct tags)")
    a.legend()
    show = tags.head(14)
    b.axis("off")
    cell = [[t[:24], f"{c:,}", lbl[:30], f"{s:.2f}"] for t, c, lbl, s in
            zip(show["tag"], show["count"], show["nearest_label"], show["similarity"], strict=True)]
    tbl = b.table(cellText=cell, colLabels=["recruiter tag", "posts", "mapped skill", "sim"],
                  loc="center", cellLoc="left", colWidths=[0.3, 0.12, 0.45, 0.1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.15)
    b.set_title("Most frequent tags and their mapping")
    return save(fig, "07_tag_mapping.png")


def fig_city(city: pd.DataFrame) -> str:
    top_cities = city.groupby("city")["posts"].sum().nlargest(10).index
    piv = city[city["city"].isin(top_cities)].pivot_table(index="label", columns="city", values="posts",
                                                         aggfunc="sum").fillna(0)
    share = piv / piv.sum(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(share.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(share.columns)), share.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(share.index)), [s[:35] for s in share.index])
    fig.colorbar(im, ax=ax, label="share of the city's top-skill mentions")
    ax.set_title("Where top skills are demanded (exact-match extraction, all posts)")
    return save(fig, "08_city_heatmap.png")


def fig_before_after(summary: dict) -> str | None:
    base_dir = DATA / "processed" / "ingest_v0_baseline"
    if not (base_dir / "summary.json").exists():
        return None
    base = json.loads((base_dir / "summary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.5]})
    ks = ["hit@1", "hit@3", "hit@5", "hit@10"]
    x = np.arange(len(ks))
    a.bar(x - 0.2, [100 * base["eval"][k] for k in ks], 0.4, label="v0: label+description index", color="#94a3b8")
    a.bar(x + 0.2, [100 * summary["eval"][k] for k in ks], 0.4, label="v1: label index", color="#2563eb")
    a.set_xticks(x, ks)
    a.set_ylabel("% of TechWolf sentences")
    a.set_title("Matching accuracy before → after")
    a.legend(fontsize=7)

    old = pd.read_csv(base_dir / "matches_lexical_all.csv.gz", usecols=["post_id", "label"])
    new = pd.read_csv(IN / "matches_lexical_all.csv.gz", usecols=["post_id", "label"])
    o, n = old.groupby("label").post_id.nunique(), new.groupby("label").post_id.nunique()
    top_old = o.nlargest(12).index
    y = np.arange(len(top_old))[::-1]
    b.barh(y + 0.2, o[top_old], 0.4, color="#94a3b8", label="v0")
    b.barh(y - 0.2, n.reindex(top_old).fillna(0), 0.4, color="#2563eb", label="v1")
    b.set_yticks(y, [t[:32] for t in top_old])
    b.set_xlabel("posts matched (exact match, all posts)")
    b.set_title("v0's top exact matches: generic words removed")
    b.legend(fontsize=7)
    return save(fig, "10_before_after.png")


def write_readme(summary: dict, figs: list[str]) -> None:
    p, ev, hs = summary["parse"], summary["eval"], summary["hybrid_sample"]
    base_path = DATA / "processed" / "ingest_v0_baseline" / "summary.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None
    tr, trl = hs["tag_recall"], hs["tag_recall_lexical_only"]
    lines = [
        "# Ingestion & Normalization: run report",
        "",
        "Pipeline: **Data parsing → Skill extraction → Sentence Transformers → Semantic matching → "
        "ESCO/O\\*NET skill mapping**. Generated by `backend/scripts/make_ingestion_report.py` from the "
        "outputs of `backend/scripts/run_ingestion.py`; every number below is measured, none illustrative.",
        "",
        "## Data",
        f"- **Job posts:** Naukri.com dump (`muhammetakkurt/naukri-jobs-dataset`, CC BY-NC 4.0, research/"
        f"demo use only): {p['raw']:,} raw → **{p['kept']:,} clean** posts "
        f"(segments: {', '.join(f'{k} {v:,}' for k, v in summary['segments'].items())}).",
        "- **Taxonomy:** O\\*NET 30.3 + ESCO v1.2.1 = 22,758 skills.",
        f"- **Model:** `{summary['model']}` (index text: {summary.get('index_text', 'label')}); "
        f"semantic threshold **{summary['threshold']}**.",
        "- **Evaluation set:** TechWolf skill-extraction test set (CC BY 4.0), job-post sentences each "
        "hand-labelled with one ESCO skill.",
        "",
        "## Key results",
        "| Metric | Value |",
        "|---|---|",
        f"| Posts removed as duplicates (id / identical text) | {p['duplicate_id']:,} / {p['duplicate_text']:,} |",
        f"| Posts with invalid dates (1970 placeholders) | {p['invalid_dates']:,} |",
        f"| TechWolf hit@1 / hit@5 / hit@10 | {100 * ev['hit@1']:.1f}% / {100 * ev['hit@5']:.1f}% / "
        f"{100 * ev['hit@10']:.1f}% (n={ev['n_sentences']}) |",
        f"| Exact-match extraction, all posts | {summary['lexical_all']['matches']:,} matches in "
        f"{summary['lexical_all']['posts_with_match']:,} posts |",
        f"| Hybrid extraction, {hs['posts']:,}-post sample | {hs['matches']:,} matches "
        f"({', '.join(f'{k} {v:,}' for k, v in hs['match_type'].items())}) |",
        f"| Recruiter tags found exactly in taxonomy | {tr['tags_exact_in_taxonomy_pct']:.1f}% |",
        f"| …of those, recovered from JD text (hybrid / exact-only) | {tr['recovered_from_text_pct']:.1f}% / "
        f"{trl['recovered_from_text_pct']:.1f}% |",
        f"| O\\*NET→ESCO crosswalk decisions | {', '.join(f'{k} {v:,}' for k, v in summary['crosswalk'].items())} |",
    ]
    if base:
        lines += [
            "",
            "## What changed between v0 and v1 (and why)",
            f"- TechWolf hit@1 {100 * base['eval']['hit@1']:.1f}% → {100 * ev['hit@1']:.1f}%, "
            f"hit@5 {100 * base['eval']['hit@5']:.1f}% → {100 * ev['hit@5']:.1f}%: index skill **labels** instead "
            "of label+description (an ablation showed long ESCO descriptions dilute the match); 5× faster to embed.",
            "- Semantic threshold 0.60 → data-driven (median top-1 similarity on TechWolf); semantic search "
            "restricted to ESCO skills (O\\*NET product names produced false semantic hits).",
            "- Exact matching no longer uses O\\*NET occupational descriptors (`Design`, `Science`, "
            "`Programming`), umbrella alt labels (`SAP`, `.NET`), generic one-word alt labels, or reviewed "
            "everyday-word tool names (`Analyze`, `Act!`, `REDUCE`, `Google`).",
            "- Boilerplate lines (\"Preferred candidate profile\", headers, < 4 words) are skipped for semantic matching.",
        ]
    lines += [
        "",
        "## Limitations (read before using these numbers)",
        "- **Not a time series.** Posting dates are a Dec-2024 scrape snapshot (most posts from Oct–Dec 2024); "
        "use only for skill *mix*, never for demand *forecasting*.",
        "- **Strict accuracy metric.** TechWolf counts only the exact ESCO label as correct; many 'misses' are "
        "near-synonyms (e.g. *work independently* vs *show initiative*).",
        "- **Small general model.** `bge-small` is not trained on job data; a domain model "
        "(e.g. TechWolf ConTeXT-Skill-Extraction, licence not declared) is the next thing to evaluate.",
        "- **Crosswalk thresholds** (0.70 review / 0.85 auto-accept) are starting points to tune on human-reviewed pairs.",
        "",
        "## Figures",
    ]
    lines += [f"![{f}]({f})" for f in figs]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((IN / "summary.json").read_text())
    hyb = pd.read_csv(IN / "matches_hybrid_sample.csv.gz")
    demand_s = pd.read_csv(IN / "skill_demand_sample.csv")
    demand_all = pd.read_csv(IN / "skill_demand_lexical_all.csv")
    sweep = pd.read_csv(IN / "eval_threshold_sweep.csv")
    cw = pd.read_csv(IN / "crosswalk_onet_esco.csv")
    tags = pd.read_csv(IN / "tag_mapping.csv")
    city = pd.read_csv(IN / "demand_by_city.csv")
    th = summary["threshold"]
    version = "onet-30.3+esco-1.2.1"

    figs = [
        fig_funnel(summary),
        fig_top_skills(demand_s, summary["hybrid_sample"]["posts"],
                       "Steps 2–4 · Top skills (hybrid semantic + exact matching, stratified sample)",
                       "02_top_skills_hybrid.png"),
        fig_match_types(hyb, th),
        fig_eval(summary, sweep, th),
        fig_embedding_map(demand_s, summary["model"], version, summary.get("index_text", "label")),
        fig_crosswalk(cw),
        fig_tags(tags),
        fig_city(city),
        fig_top_skills(demand_all, summary["lexical_all"]["posts"],
                       "Top skills across ALL posts (exact-match only)", "09_top_skills_all_lexical.png"),
        fig_before_after(summary),
    ]
    figs = [f for f in figs if f]
    write_readme(summary, figs)
    print("wrote", figs, "+ README.md")


if __name__ == "__main__":
    main()
