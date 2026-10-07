import numpy as np
import pandas as pd
import pytest

from d2s.ml.embeddings import HashingEmbedder
from d2s.ml.forecasting import DemandForecaster, Pattern, classify, demand_weights
from d2s.ml.skills import SkillIndex, SkillMatcher, split_clauses
from d2s.ml.taxonomy import Skill, find_onet_dir, load_onet

# ---- embeddings / skills ----------------------------------------------------------------

SKILLS = [
    Skill(id="s:python", label="Python", source="test", kind="tool", description="programming language"),
    Skill(id="s:sql", label="SQL", source="test", kind="tool", description="database query language"),
    Skill(id="s:tableau", label="Tableau", source="test", kind="tool", description="data visualization software"),
    Skill(id="s:negotiation", label="Negotiation", source="test", kind="skill",
          description="Bringing others together and trying to reconcile differences."),
]


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder()
    a = e.embed_queries(["data analysis with python", "data analysis with python"])
    assert a.shape == (2, e.dim)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a[0], a[1])


def test_split_clauses_handles_bullets_and_sentences():
    text = "We need an analyst.\n• Strong SQL and Python skills\n- Build Tableau dashboards; present to leadership."
    clauses = split_clauses(text)
    assert any("SQL" in c for c in clauses)
    assert any("Tableau" in c for c in clauses)


def test_matcher_finds_lexical_tools_with_evidence():
    m = SkillMatcher(SKILLS, HashingEmbedder(), threshold=0.99)  # semantic effectively off
    found = {x.skill_id: x for x in m.extract("Must know SQL and Python. Build Tableau dashboards for sales.")}
    assert {"s:python", "s:sql", "s:tableau"} <= set(found)
    assert found["s:sql"].lexical
    assert "SQL" in found["s:sql"].evidence
    assert "s:negotiation" not in found


def test_lexical_match_ignores_sentence_punctuation_but_keeps_dotted_names():
    skills = [*SKILLS, Skill(id="s:node", label="Node.js", source="test", kind="tool")]
    m = SkillMatcher(skills, HashingEmbedder(), threshold=0.99)
    found = {x.skill_id for x in m.extract("We use Node.js daily. Our stack is mostly Python.")}
    assert {"s:node", "s:python"} <= found


def test_index_roundtrip(tmp_path):
    e = HashingEmbedder()
    idx = SkillIndex.build(SKILLS, e)
    path = tmp_path / "idx.npz"
    idx.save(path)
    loaded = SkillIndex.load(path)
    assert loaded.ids == idx.ids and loaded.model_name == e.name
    np.testing.assert_allclose(loaded.matrix, idx.matrix)


def test_matcher_rejects_index_from_other_model():
    idx = SkillIndex.build(SKILLS, HashingEmbedder(dim=256))
    with pytest.raises(ValueError, match="index built with"):
        SkillMatcher(SKILLS, HashingEmbedder(dim=512), index=idx)


# ---- taxonomy (uses the downloaded O*NET files when present) ------------------------------

def _onet_available() -> bool:
    try:
        find_onet_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _onet_available(), reason="O*NET dataset not downloaded")
def test_load_onet_30_3():
    tax = load_onet()
    assert len(tax.occupations) > 900
    kinds = {s.kind for s in tax.skills.values()}
    assert {"essential_skill", "transferable_skill", "knowledge", "tool"} <= kinds
    assert any(s.hot_technology for s in tax.skills.values())
    rated = [link for link in tax.links if link.importance is not None]
    assert rated and all(1 <= link.importance <= 5 for link in rated[:500])
    assert tax.version.startswith("onet-30")


# ---- forecasting -------------------------------------------------------------------------

def _series(uid, values, start="2023-01-01"):
    return pd.DataFrame({"unique_id": uid, "ds": pd.date_range(start, periods=len(values), freq="MS"),
                         "y": values})


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17, 19, 18, 20], Pattern.SMOOTH),
        ([0, 0, 5, 0, 0, 0, 6, 0, 0, 5, 0, 0, 0, 5, 0, 0, 6, 0], Pattern.INTERMITTENT),
        ([0, 0, 1, 0, 0, 0, 30, 0, 0, 2, 0, 0, 0, 25, 0, 0, 1, 0], Pattern.LUMPY),
        ([0] * 18, Pattern.ZERO),
        ([3, 4, 5], Pattern.SHORT),
    ],
)
def test_classify(values, expected):
    assert classify(np.array(values), min_history=12) is expected


def test_forecaster_routes_and_produces_valid_intervals():
    rng = np.random.default_rng(0)
    t = np.arange(36)
    smooth = 50 + 0.8 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 36)
    sparse = np.where(rng.random(36) < 0.25, rng.integers(1, 6, 36), 0)
    df = pd.concat([
        _series("python", smooth.round()),
        _series("cobol", sparse),
        _series("new_skill", [1, 2, 4, 5]),
        _series("dead_skill", [0] * 36),
    ])
    fc = DemandForecaster(horizon=6, level=80)
    out = {r.unique_id: r for r in fc.fit_predict(df)}

    assert out["python"].pattern in (Pattern.SMOOTH, Pattern.ERRATIC)
    assert out["cobol"].pattern in (Pattern.INTERMITTENT, Pattern.LUMPY)
    assert out["new_skill"].pattern is Pattern.SHORT
    assert out["dead_skill"].pattern is Pattern.ZERO and out["dead_skill"].total() == 0

    for r in out.values():
        assert len(r.points) == 6
        for p in r.points:
            assert 0 <= p.lo <= p.point <= p.hi
    assert out["python"].backtest_mae is not None
    assert out["python"].total() > out["cobol"].total()

    w = demand_weights(list(out.values()))
    w_lo = demand_weights(list(out.values()), "lo")
    assert all(w_lo[k] <= w[k] + 1e-9 for k in w)


def test_forecaster_fills_missing_months_with_zero():
    df = pd.DataFrame({"unique_id": "x", "ds": pd.to_datetime(["2024-01-01", "2024-04-01"]), "y": [3, 4]})
    prepared = DemandForecaster()._prepare(df)
    assert list(prepared["y"]) == [3, 0, 0, 4]
