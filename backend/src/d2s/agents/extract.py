"""Read intents, numbers, courses and skill areas from a planner's question, deterministically.

Every number an agent acts on comes from here, never from the language model. Where the question
implies a change without a number ("a trainer leaves"), a documented default is used and reported back
to the user as an assumption.
"""

import re
from dataclasses import dataclass, field

# ---- vocabulary ----------------------------------------------------------------------------------------
COURSE_ALIASES = {
    "py_sql": ["python & sql", "python and sql", "python/sql", "py sql", "sql course", "python course"],
    "sas_prog": ["sas programming", "sas viya", "sas course"],
    "stats": ["statistics", "stats", "probability", "applied statistics"],
    "ml": ["machine learning", "ml course", "ml foundations"],
    "dl_nlp": ["deep learning", "nlp", "natural language"],
    "bigdata": ["big data course", "big-data", "hadoop", "spark", "hive"],
    "story_tableau": ["tableau"],
    "story_powerbi": ["power bi", "powerbi"],
    "bootcamp": ["bootcamp", "boot camp", "capstone"],
}
AREA_ALIASES = {
    "coding_skills": ["coding", "programming", "code"],
    "maths_stats_skills": ["maths", "math", "statistics skills", "stats skills", "quantitative"],
    "ai_and_ml_skills": ["ai", "artificial intelligence", "ai & ml", "ai/ml", "genai", "gen ai"],
    "big_data_skills": ["big data", "data engineering"],
    "dashboard_and_storytelling_skills": ["dashboard", "storytelling", "visualisation", "visualization", "bi skills"],
}
INTENT_WORDS = {
    "plan_confidence": ["how sure", "confident", "confidence", "robust", "reliable", "certain", "trust", "how risky"],
    "compare_baselines": ["naive", "simple rule", "rule of thumb", "biggest gap", "cheapest", "baseline", "smarter",
                          "better than", "compared to", "compare with", "versus", " vs "],
    "alternative_plans": ["alternative", "other option", "options", "plan b", "other plans", "choices", "different plan"],
    "why_not_course": ["why not", "why didn't", "why did not", "why wasn't", "why was not", "why isn't", "why no ",
                       "left out", "not chosen", "excluded", "skip", "skipped"],
    "optimise_plan": ["best plan", "what should we teach", "what to teach", "recommend", "which courses", "plan for",
                      "what should we run", "what do we run"],
}
DOWN = ["cut", "reduc", "drop", "lose", "losing", "lost", "lower", "decreas", "less", "shrink", "fewer", "leav",
        "quit", "fall", "falls", "down", "slash", "minus", "short of"]
UP = ["increase", "more", "extra", "raise", "rise", "rises", "add", "grow", "boost", "jump", "up", "hot", "surge",
      "spike", "double", "higher", "booming", "boom", "plus", "gain"]

DEFAULTS = {  # used when the question implies a change but gives no number (always reported back)
    "trainer_leaves_hours": 60,
    "demand_shift_pct": 50,
    "cohort_shift_pct": 25,
    "price_shift_pct": 20,
    "budget_shift_pct": 20,
}


@dataclass
class Parsed:
    intents: list[str] = field(default_factory=list)
    scenario: dict = field(default_factory=dict)  # what_if arguments
    budget: int | None = None  # absolute budget mentioned (optimise_plan)
    trainer_hours: int | None = None  # absolute hours mentioned
    courses: list[str] = field(default_factory=list)  # course ids mentioned
    areas: list[str] = field(default_factory=list)  # skill areas mentioned
    assumptions: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return " " + re.sub(r"\s+", " ", text.lower().replace("’", "'")) + " "


def _has(t: str, words) -> bool:
    return any(w in t for w in words)


def _word(t: str, w: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", t) is not None


def _direction(window: str) -> int:
    down = any(_word(window, w) or w in window for w in DOWN)
    up = any(_word(window, w) for w in UP)
    if "half" in window or "halve" in window:
        return -1
    return -1 if down and not up else 1 if up and not down else (-1 if down else 1)


def _money(t: str) -> list[int]:
    """Rupee amounts: '2 lakh', '₹2L', '2.5 lakhs', '50k', '2,00,000', 'rs 150000'."""
    out = []
    for m in re.finditer(r"(?:₹|rs\.?\s*|inr\s*)?(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|l\b|k\b|thousand|crore|cr\b)", t):
        v, unit = float(m.group(1)), m.group(2)
        mult = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "l": 1e5, "k": 1e3, "thousand": 1e3, "crore": 1e7,
                "cr": 1e7}[unit.strip()]
        out.append(int(v * mult))
    for m in re.finditer(r"(?:₹|rs\.?\s*|inr\s*)(\d[\d,]{3,})", t):
        out.append(int(m.group(1).replace(",", "")))
    for m in re.finditer(r"(?<![\d.])(\d{1,3}(?:,\d{2})*,\d{3}|\d{5,8})(?![\d%])", t):  # bare 2,00,000 / 200000
        v = int(m.group(1).replace(",", ""))
        if v >= 10_000 and v not in out:
            out.append(v)
    return out


def _pct_near(t: str, keys: list[str], span: int = 45) -> int | None:
    """A percentage within `span` characters of any key word; sign from direction words around it."""
    for k in keys:
        for km in re.finditer(re.escape(k), t):
            window = t[max(0, km.start() - span): km.end() + span]
            m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent|pc\b)", window)
            if m:
                v = round(float(m.group(1)))
                return v * _direction(window)
            if "double" in window:
                return 100
            if _word(window, "half") or "halve" in window:
                return -50
    return None


def parse(question: str, course_names: dict[str, str], current_budget: int, current_hours: int | None) -> Parsed:
    t = _norm(question)
    p = Parsed()

    # courses and areas mentioned
    for cid, name in course_names.items():
        aliases = COURSE_ALIASES.get(cid, []) + [name.lower(), cid.replace("_", " ")]
        if any(a and a in t for a in aliases):
            p.courses.append(cid)
    for area, aliases in AREA_ALIASES.items():
        if any(_word(t, a) for a in aliases):
            p.areas.append(area)

    # explicit intents
    for intent, words in INTENT_WORDS.items():
        if _has(t, words):
            p.intents.append(intent)
    if "why_not_course" in p.intents and not p.courses:
        p.intents.remove("why_not_course")

    # ---- what-if scenario: each change is read from its own clause ------------------------------------
    sc: dict = {}
    clauses = [c for c in re.split(r"(?<!\d),|,(?!\d)|;|(?<!\d)\.(?!\d)|\?|!| and | but | while | plus | also ", t) if c.strip()]
    money = _money(t)
    hour_m = re.search(r"(\d+)\s*(?:trainer[- ]?)?(?:hours|hrs|hour|h\b)", t)
    change_words = ("cut", "cuts", "drop", "drops", "increase", "reduce", "lose", "loses", "lost", "what if",
                    "only have", "goes up", "go up", "rise", "rises")
    absolute_ask = "optimise_plan" in p.intents and not any(w in t for w in change_words)

    def clause_with(words):
        return [c for c in clauses if _has(c, words)]

    def pct_in(c):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent|pc\b)", c)
        if m:
            return round(float(m.group(1))) * _direction(c)
        if "double" in c:
            return 100
        if _word(c, "half") or "halve" in c:
            return -50
        return None

    budget_words = ["budget", "money", "funding", "funds", "spend", "grant", "only have", "we have"]
    if absolute_ask:
        if money:
            p.budget = money[0]
        if hour_m:
            p.trainer_hours = int(hour_m.group(1))
    else:
        for c in clause_with(budget_words) or ([cl for cl in clauses if _money(cl)] if money else []):
            pct = pct_in(c)
            m = _money(c)
            if pct is None and m and current_budget:
                target = m[0]
                if " by " in c and target < current_budget and _direction(c) < 0:  # "cut by 1 lakh"
                    target = current_budget - target
                elif " by " in c and _direction(c) > 0:  # "increase by 1 lakh"
                    target = current_budget + target
                pct = round(100 * (target - current_budget) / current_budget)
            if pct is None and any(_word(c, w) for w in ("cut", "cuts", "reduced", "reduce", "slashed", "more",
                                                          "extra", "increase", "increased")):
                pct = DEFAULTS["budget_shift_pct"] * _direction(c)
                p.assumptions.append(f"No size was given for the budget change, so {pct:+d}% was assumed.")
            if pct:
                sc["budget_pct"] = max(-95, min(500, pct))
                break

        hc = clause_with(["hour", "hrs"])
        hm = re.search(r"(\d+)\s*(?:trainer[- ]?)?(?:hours|hrs|hour|h\b)", hc[0]) if hc else None
        if hm:
            sc["trainer_hours_delta"] = int(hm.group(1)) * _direction(hc[0])
        elif clause_with(["trainer", "faculty", "instructor", "teacher"]):
            c = clause_with(["trainer", "faculty", "instructor", "teacher"])[0]
            if _has(c, ["leav", "quit", "lose", "lost", "resign", "sick", "unavailable", "retire"]):
                sc["trainer_hours_delta"] = -DEFAULTS["trainer_leaves_hours"]
                p.assumptions.append(f"A trainer leaving was taken as -{DEFAULTS['trainer_leaves_hours']} trainer-hours.")
            elif _has(c, ["hire", "new trainer", "another trainer", "extra trainer", "join"]):
                sc["trainer_hours_delta"] = DEFAULTS["trainer_leaves_hours"]
                p.assumptions.append(f"An extra trainer was taken as +{DEFAULTS['trainer_leaves_hours']} trainer-hours.")

        price_words = ["price", "course cost", "fees", "fee", "vendor", "inflation"]
        for c in clause_with(price_words):
            pct = pct_in(c)
            if pct is None and _has(c, DOWN + UP):
                pct = DEFAULTS["price_shift_pct"] * _direction(c)
                p.assumptions.append(f"No size was given for the price change, so {pct:+d}% was assumed.")
            if pct:
                sc["cost_pct"] = pct
                break

        cohort_words = ["learners", "students", "batch", "cohort", "intake", "enrol"]
        for c in clause_with(cohort_words):
            if not _has(c, DOWN + UP + ["bigger", "larger", "smaller"]):
                continue
            pct = pct_in(c)
            if pct is None:
                pct = DEFAULTS["cohort_shift_pct"] * (-1 if _has(c, ["smaller", "fewer", "less"]) else 1)
                p.assumptions.append(f"No size was given for the cohort change, so {pct:+d}% learners short was assumed.")
            elif _has(c, ["bigger", "larger"]):
                pct = abs(pct)
            sc["cohort_pct"] = pct
            break

        demand_words = ["demand", "hot", "popular", "hiring", "boom", "trending", "surge", "jump"]
        shifts = {}
        for c in clauses:
            areas = [a for a, al in AREA_ALIASES.items() if any(_word(c, x) for x in al)]
            if areas and _has(c, demand_words):
                pct = pct_in(c)
                if pct is None:
                    pct = DEFAULTS["demand_shift_pct"] * _direction(c)
                    p.assumptions.append(f"No size was given for the demand shift, so {pct:+d}% was assumed.")
                shifts.update({a: pct for a in areas})
        if shifts:
            sc["demand_pct"] = shifts

    if p.courses:
        remove_ctx = ["without", "unavailable", "not available", "cancel", "no longer", "drop the", "lose the",
                      "can't run", "cannot run", "won't run", "remove"]
        require_ctx = ["must run", "must include", "insist", "mandatory", "require", "force", "have to run",
                       "need to run", "include the", "add the"]
        if "why_not_course" not in p.intents:
            if _has(t, remove_ctx):
                sc["remove_courses"] = list(p.courses)
            elif _has(t, require_ctx):
                sc["require_courses"] = list(p.courses)

    if sc:
        p.scenario = sc
        if "what_if" not in p.intents:
            p.intents.insert(0, "what_if")

    if "what_if" in p.intents and "optimise_plan" in p.intents:
        p.intents.remove("optimise_plan")  # the what-if already re-optimises
    return p
