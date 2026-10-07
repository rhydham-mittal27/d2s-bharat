"""Ablation on TechWolf's labelled sentences: which skill text and query encoding match best?

Variants (ESCO candidates only, same gold labels as the pipeline eval):
  doc text : label | label + alt labels | label + description (current index)
  query    : encode_query (model's query prompt, if any) | encode_document (no prompt)

Usage:  uv run python scripts/eval_matching_variants.py [--model BAAI/bge-small-en-v1.5]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from d2s.config import get_settings
from d2s.ml.taxonomy import load_esco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=get_settings().embedding_model)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)
    print(f"model {args.model}; prompts configured: {model.prompts or 'none'}", flush=True)

    esco = load_esco()
    skills = list(esco.skills.values())
    by_label = {s.label.lower(): i for i, s in enumerate(skills)}
    tw = pd.read_csv(get_settings().dataset_dir / "eval" / "techwolf" / "test.csv")
    tw["gold"] = tw["label"].str.lower().map(by_label)
    tw = tw.dropna(subset=["gold"])
    gold = tw["gold"].astype(int).to_numpy()
    sentences = tw["sentence"].str.lstrip("*•- ").tolist()

    doc_variants = {
        "label": [s.label for s in skills],
        "label+alt": [s.label + ("; " + "; ".join(s.alt_labels[:8]) if s.alt_labels else "") for s in skills],
        "label+description": [s.embedding_text for s in skills],
    }
    enc = dict(batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    queries = {
        "query-prompt": model.encode_query(sentences, **enc),
        "no-prompt": model.encode_document(sentences, **enc),
    }

    rows = []
    for dname, texts in doc_variants.items():
        t = time.perf_counter()
        docs = model.encode_document(texts, **enc)
        print(f"embedded {len(texts):,} docs [{dname}] in {time.perf_counter() - t:.0f}s", flush=True)
        for qname, q in queries.items():
            sims = q @ docs.T
            gold_sim = sims[np.arange(len(gold)), gold]
            rank = (sims > gold_sim[:, None]).sum(axis=1) + 1
            rows.append({
                "model": args.model, "doc_text": dname, "query": qname,
                "hit@1": float((rank <= 1).mean()), "hit@5": float((rank <= 5).mean()),
                "hit@10": float((rank <= 10).mean()), "mrr": float((1 / rank).mean()),
            })
            print(json.dumps(rows[-1]), flush=True)

    out = get_settings().dataset_dir / "processed" / "ingest"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out / f"eval_variants__{args.model.replace('/', '__')}.csv"
    df.to_csv(path, index=False)
    print(df.sort_values("mrr", ascending=False).to_string(index=False))
    print(f"DONE in {time.perf_counter() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
