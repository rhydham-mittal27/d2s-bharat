"""(Re)build the ChromaDB skill collection from the analysis outputs.

    uv run python scripts/build_vector_index.py          # embedded store at <dataset>/chroma
    uv run python scripts/build_vector_index.py --query "pytorch"

The API builds the collection automatically on first start when it is empty; run this after
re-running the analysis (new skill_stats.csv) or switching the embedding model. Uses the cached
embeddings in skill_lookup.npz when they match the model, otherwise embeds the skills now.
"""

import argparse
import sys
import time

from d2s.config import get_settings
from d2s.vector import SkillVectorStore, build_skill_index, chroma_client


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=None, help="run one search afterwards to check the index")
    args = ap.parse_args()
    s = get_settings()
    t = time.perf_counter()
    vs = SkillVectorStore(chroma_client(), s.embedding_model)
    where = f"{s.chroma_host}:{s.chroma_port}" if s.chroma_host else str(s.chroma_path)
    print(f"chroma: {where} / collection {vs.name}")
    embedder = None
    try:
        n = build_skill_index(vs)
    except ValueError:  # no cached embeddings for this model
        from d2s.ml.embeddings import get_embedder

        embedder = get_embedder()
        n = build_skill_index(vs, embedder)
    print(f"indexed {n} skills in {time.perf_counter() - t:.1f}s")
    if args.query:
        if embedder is None:
            from d2s.ml.embeddings import get_embedder

            embedder = get_embedder()
        for meta, sim in vs.query(embedder.embed_queries([args.query])[0], 5):
            print(f"  {sim:.3f}  {meta['skill']}  ({meta['posts_data']} data-role postings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
