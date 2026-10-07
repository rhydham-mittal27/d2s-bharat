"""Explaining skill matches: why a recruiter tag or a piece of text maps to a skill.

Tags follow the deployed decision path (exact -> contains -> semantic -> review/unmapped); the
explanation names the step that fired, the evidence (matched words, similarity and threshold) and the
runner-up candidates, so a reviewer can correct a near-miss.
"""

import re

from d2s.ingest import sas as tagrules
from d2s.ml.skills import MAX_LABEL_WORDS, SkillMatcher, _norm, is_boilerplate, split_clauses
from d2s.xai.schemas import Explanation, Factor


def _span_regex(key: str) -> re.Pattern:
    words = [re.escape(w) for w in key.split()]
    return re.compile(r"(?<![\w])" + r"[\W_]+".join(words) + r"(?![\w])", re.I)


def explain_tag(matcher: SkillMatcher, tag: str, k: int = 3) -> Explanation:
    key = _norm(tag)
    exact = matcher.lookup(tag)
    ranked = matcher.rank_skills([tag], k=k)[0]
    runners = [Factor(name=matcher.label_of(sid), value=round(sim, 3), unit="cosine similarity",
                      source="embedding.rank", note="closest by meaning alone") for sid, sim in ranked]

    def others(chosen: str) -> list[Factor]:  # meaning-only runner-ups, without the skill already chosen
        return [f for (sid, _), f in zip(ranked, runners, strict=True) if sid != chosen]
    if exact:
        label = matcher.label_of(exact)
        how = "its preferred label" if _norm(label) == key else "an alternative label or acronym"
        return Explanation(subject=f"tag:{tag}", question="why",
                           summary=f"'{tag}' matches {how} of the skill '{label}' exactly.",
                           factors=[Factor(name=label, value=1.0, source="lexicon.exact")] + others(exact),
                           evidence=[f"normalised form '{key}' found in the skill lexicon"],
                           confidence=1.0, method="exact lexicon lookup (labels, filtered alt labels, O*NET acronyms)",
                           details={"step": "exact", "skill": label})
    inner = matcher.longest_contained(tag)
    if inner:
        label = matcher.label_of(inner)
        sub = next((" ".join(key.split()[i:i + n]) for n in range(MAX_LABEL_WORDS, 0, -1)
                    for i in range(len(key.split()) - n + 1) if matcher.lookup(" ".join(key.split()[i:i + n])) == inner), "")
        return Explanation(subject=f"tag:{tag}", question="why",
                           summary=f"'{tag}' contains the known skill name '{sub}', so it maps to '{label}'.",
                           factors=[Factor(name=label, value=1.0, source="lexicon.contains", note=f"matched '{sub}'")]
                           + others(inner),
                           evidence=[f"'{sub}' inside '{tag}'"], confidence=0.95,
                           method="longest known skill name contained in the tag",
                           details={"step": "contains", "skill": label, "matched_text": sub})
    sid, sim = ranked[0]
    bar = tagrules.TAG_HIGH_SINGLE if " " not in key else tagrules.TAG_HIGH
    margin = sim - (ranked[1][1] if len(ranked) > 1 else 0.0)
    if sim >= bar:
        step, verdict = "semantic", "accepted"
    elif sim >= tagrules.TAG_MEDIUM:
        step, verdict = "review", "sent to the review queue"
    else:
        step, verdict = "unmapped", "left unmapped"
    summary = (f"No exact or contained skill name; the closest skill by meaning is '{matcher.label_of(sid)}' "
               f"(similarity {sim:.2f}, threshold {bar:.2f} for {'one-word' if ' ' not in key else 'multi-word'} tags), "
               f"so the match was {verdict}. Margin over the runner-up: {margin:.2f}.")
    return Explanation(subject=f"tag:{tag}", question="why", summary=summary, factors=runners,
                       confidence=round(float(sim), 3),
                       method="sentence-embedding nearest neighbour (bge-small) with documented thresholds",
                       details={"step": step, "similarity": sim, "threshold": bar, "margin": margin,
                                "review_band": tagrules.TAG_MEDIUM})


def explain_text(matcher: SkillMatcher, text: str, limit: int = 15) -> Explanation:
    """Which words in a job text triggered which skills (exact-name matches, with character spans)."""
    clauses = split_clauses(text) or [text]
    hits: dict[str, dict] = {}
    for clause in clauses:
        words = _norm(clause).split()
        for n in range(MAX_LABEL_WORDS, 0, -1):
            for i in range(len(words) - n + 1):
                key = " ".join(words[i:i + n])
                sid = matcher.lookup(key)
                if not sid or sid in hits:
                    continue
                m = _span_regex(key).search(text)
                hits[sid] = {"skill": matcher.label_of(sid), "matched": key,
                             "span": [m.start(), m.end()] if m else None, "clause": clause,
                             "clause_is_boilerplate": is_boilerplate(clause)}
    items = list(hits.values())[:limit]
    factors = [Factor(name=h["skill"], value=h["matched"], source="lexicon.exact",
                      note=f"characters {h['span'][0]}-{h['span'][1]}" if h["span"] else "") for h in items]
    summary = (f"Found {len(hits)} skill name(s) in the text: " + ", ".join(h["skill"] for h in items[:6])
               + ("..." if len(hits) > 6 else "") + ".") if hits else "No known skill names found in the text."
    return Explanation(subject="text", question="why", summary=summary, factors=factors,
                       evidence=[h["clause"] for h in items[:5]],
                       method="exact skill-name matching with character spans (meaning-based text matching was "
                              "tested and rejected on the truncated SAS descriptions)",
                       details={"matches": items})
