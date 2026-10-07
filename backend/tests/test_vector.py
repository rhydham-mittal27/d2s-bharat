"""ChromaDB skill store: cosine ranking, exact-name lookup, rebuild, per-model collections, and the
real index built from the analysis outputs."""

import numpy as np
import pytest
from tests_support import artifacts_ready

from d2s.vector import SkillVectorStore, build_skill_index, chroma_client
from d2s.vector.store import _collection_name, market_skill_records

DIM = 384


def _rec(name, posts=10, pay=None):
    return {"skill": name, "posts_all": posts, "posts_data": posts // 2, "share_all": 0.01, "share_data": 0.02,
            "median_salary_mid": None, "top_family": None, "pay_odds_ratio": pay,
            "pay_ci_low": None if pay is None else pay - 0.2, "pay_ci_high": None if pay is None else pay + 0.2,
            "pay_q_fdr": None}


@pytest.fixture
def client(tmp_path):
    return chroma_client(tmp_path / "chroma")


def test_query_ranks_by_cosine_and_finds_exact_names(client):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, DIM))
    vs = SkillVectorStore(client, "test-model")
    assert vs.rebuild([_rec("SQL"), _rec("Python", pay=1.5), _rec("Power BI"), _rec("Hadoop")], vecs) == 4

    q = vecs[1] + 0.05 * rng.normal(size=DIM)
    hits = vs.query(q, k=2)
    assert hits[0][0]["skill"] == "Python" and hits[0][1] > 0.99 and len(hits) == 2
    assert hits[0][0]["pay_odds_ratio"] == 1.5 and hits[0][0]["median_salary_mid"] is None

    assert vs.by_name("power bi")["skill"] == "Power BI"  # case-insensitive
    assert vs.by_name("cobol") is None
    assert vs.query(q, k=50)[-1][1] <= hits[0][1]  # k larger than the collection is fine


def test_rebuild_replaces_and_models_are_separate(client):
    rng = np.random.default_rng(1)
    a = SkillVectorStore(client, "BAAI/bge-small-en-v1.5")
    a.rebuild([_rec("x"), _rec("y")], rng.normal(size=(2, DIM)))
    a.rebuild([_rec("z")], rng.normal(size=(1, DIM)))
    assert a.count() == 1 and a.by_name("x") is None  # old contents gone

    b = SkillVectorStore(client, "other/model")
    assert b.count() == 0 and b.name != a.name  # different embedding space, different collection
    assert _collection_name("BAAI/bge-small-en-v1.5") == "market_skills__bge-small-en-v1.5"
    with pytest.raises(ValueError, match="differ in length"):
        b.rebuild([_rec("q")], rng.normal(size=(2, DIM)))


def test_store_persists_across_clients(tmp_path):
    rng = np.random.default_rng(2)
    SkillVectorStore(chroma_client(tmp_path / "c"), "m").rebuild([_rec("SAS")], rng.normal(size=(1, DIM)))
    assert SkillVectorStore(chroma_client(tmp_path / "c"), "m").by_name("SAS") is not None


@pytest.mark.skipif(not artifacts_ready(), reason="analysis outputs / artifacts not built")
def test_real_index_from_cached_embeddings(client):
    """Builds the production collection from skill_lookup.npz (no model download) and checks that a
    skill's own embedding finds itself first."""
    import numpy as np

    from d2s.services import store

    cache = store.artifacts() / "skill_lookup.npz"
    if not cache.exists():
        pytest.skip("skill_lookup.npz not built")
    z = np.load(cache, allow_pickle=False)
    vs = SkillVectorStore(client, str(z["model_name"]))
    n = build_skill_index(vs)
    assert n == len(market_skill_records()) > 1000
    names = list(z["names"])
    i = names.index("machine learning")
    top = vs.query(z["matrix"][i], k=3)
    assert top[0][0]["skill"] == "machine learning" and top[0][1] > 0.999
