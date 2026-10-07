"""Step 1 - Data parsing: raw job posts -> clean, de-duplicated JobPost table.

Source today: the Naukri.com job-postings dump (Hugging Face `muhammetakkurt/naukri-jobs-dataset`,
CC BY-NC 4.0). Each loader returns the same columns, so more sources can be added later.
"""

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

COLUMNS = [
    "post_id", "source", "segment", "title", "company", "city", "created", "experience",
    "salary", "tags", "text",
]

_BLOCK_TAGS = re.compile(r"<\s*(br|/p|p|/li|/div|div|/h\d|h\d|/tr|tr)\b[^>]*>", re.I)
_LI = re.compile(r"<\s*li\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n\s*\n+")


def clean_html(raw: str) -> str:
    """HTML job description -> plain text that keeps line/bullet structure for clause splitting."""
    if not isinstance(raw, str):
        return ""
    text = _LI.sub("\n• ", raw)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKS.sub("\n", text).strip()


def _text_key(text: str) -> str:
    """Fingerprint for near-exact duplicate descriptions (same JD reposted under a new id)."""
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()


@dataclass
class ParseStats:
    raw: int = 0
    duplicate_id: int = 0
    empty: int = 0
    duplicate_text: int = 0
    kept: int = 0
    invalid_dates: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def load_naukri(paths: list[Path]) -> tuple[pd.DataFrame, ParseStats]:
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True, dtype={"jobId": str})
        df["segment"] = Path(p).stem.replace("naukri_", "")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    stats = ParseStats(raw=len(raw))

    raw = raw.drop_duplicates(subset="jobId")
    stats.duplicate_id = stats.raw - len(raw)

    created = pd.to_datetime(raw["createdDate"], errors="coerce")
    invalid = created.isna() | (created.dt.year < 2000)  # 1970-01-01 placeholders in the dump
    stats.invalid_dates = int(invalid.sum())
    created = created.mask(invalid)

    description = raw["jobDescription"].map(clean_html)
    posts = pd.DataFrame({
        "post_id": "naukri:" + raw["jobId"].astype(str),
        "source": "naukri",
        "segment": raw["segment"],
        "title": raw["title"].fillna("").str.strip(),
        "company": raw["companyName"].fillna(""),
        "city": raw["location"].fillna("").str.split(",").str[0].str.strip(),
        "created": created,
        "experience": raw.get("experience", pd.Series("", index=raw.index)).fillna(""),
        "salary": raw.get("salary", pd.Series("", index=raw.index)).fillna(""),
        "tags": raw["tagsAndSkills"].fillna("").map(
            lambda s: [t.strip() for t in s.split(",") if t.strip()]
        ),
        "text": (raw["title"].fillna("") + "\n" + description).str.strip(),
    })

    empty = description.str.len() < 40
    stats.empty = int(empty.sum())
    posts = posts[~empty.to_numpy()]

    keys = posts["text"].map(_text_key)
    dup = keys.duplicated()
    stats.duplicate_text = int(dup.sum())
    posts = posts[~dup.to_numpy()].reset_index(drop=True)
    stats.kept = len(posts)
    return posts[COLUMNS], stats


# ---- SAS hackathon files (the only job data used for results) ----------------------------------

@dataclass
class SasCleaningLog:
    """Every cleaning action with its row/value count, for the report's data-preparation section."""
    raw_rows: int = 0
    exact_duplicate_rows: int = 0
    kept_rows: int = 0
    missing_description: int = 0
    missing_job_type: int = 0
    job_type_variants_merged: int = 0
    multi_city_rows: int = 0
    missing_key_skills: int = 0
    tags_total: int = 0
    tags_truncated_fixed: int = 0
    tags_dropped_empty_or_ellipsis: int = 0
    tags_case_duplicates_dropped: int = 0
    descriptions_truncated: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")


def _range(text: str) -> tuple[float | None, float | None]:
    m = _RANGE.search(str(text))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _clean_tags(raw: str, log: SasCleaningLog) -> tuple[list[str], int]:
    if not isinstance(raw, str):
        return [], 0
    out, seen, truncated = [], set(), 0
    for t in raw.split(","):
        log.tags_total += 1
        t = t.strip()
        if t.endswith("..."):
            t = t[:-3].strip()
            truncated += 1
        if len(t) < 1 or set(t) <= {"."}:
            log.tags_dropped_empty_or_ellipsis += 1
            continue
        key = t.casefold()
        if key in seen:
            log.tags_case_duplicates_dropped += 1
            continue
        seen.add(key)
        out.append(t)
    log.tags_truncated_fixed += truncated
    return out, truncated


def load_sas_analytics(path: Path) -> tuple[pd.DataFrame, SasCleaningLog]:
    """SAS `Analytics Jobs.csv` -> one clean row per posting.

    Descriptions in this file are truncated (~100 chars, ';'-separated, ending in '...'),
    so recruiter `key_skills` are the primary skill signal and the text is secondary.
    """
    raw = pd.read_csv(path)
    log = SasCleaningLog(raw_rows=len(raw))
    dup = raw.drop(columns=["s_no"]).duplicated()
    log.exact_duplicate_rows = int(dup.sum())
    df = raw[~dup].copy()
    log.kept_rows = len(df)

    exp = df["experience"].map(_range)
    sal = df["salary"].map(_range)
    jt = df["job_type"].str.strip().str.lower()
    log.missing_job_type = int(jt.isna().sum())
    log.job_type_variants_merged = int((df["job_type"].dropna() != "Analytics").sum())
    jt = jt.replace({"analytic": "analytics"}).fillna("unknown")

    locations = df["location"].fillna("").map(lambda s: [c.strip() for c in s.split(",") if c.strip()])
    log.multi_city_rows = int((locations.map(len) > 1).sum())

    desc = df["job_description"]
    log.missing_description = int(desc.isna().sum())
    desc = desc.fillna("")
    log.descriptions_truncated = int(desc.str.rstrip().str.endswith("...").sum())
    desc = desc.str.replace(r"\.\.\.\s*$", "", regex=True).str.replace(";", "\n")

    log.missing_key_skills = int(df["key_skills"].isna().sum())
    tags = df["key_skills"].map(lambda s: _clean_tags(s, log)[0])

    posts = pd.DataFrame({
        "post_id": "sas:" + df["s_no"].astype(str),
        "source": "sas_analytics",
        "segment": jt,
        "title": df["job_desig"].fillna("").str.strip(),
        "company": "",
        "city": locations.map(lambda xs: xs[0] if xs else ""),
        "locations": locations,
        "created": pd.NaT,
        "experience": df["experience"],
        "exp_min": exp.map(lambda r: r[0]), "exp_max": exp.map(lambda r: r[1]),
        "salary": df["salary"],
        "sal_min_lakh": sal.map(lambda r: r[0]), "sal_max_lakh": sal.map(lambda r: r[1]),
        "tags": tags,
        "text": (df["job_desig"].fillna("") + "\n" + desc).str.strip(),
    }).reset_index(drop=True)
    posts["sal_mid_lakh"] = (posts["sal_min_lakh"] + posts["sal_max_lakh"]) / 2
    return posts, log


def _lakh(v) -> float | None:
    s = str(v).strip().upper().replace(",", "")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(L|CR)?$", s)
    if not m:
        return None
    return float(m.group(1)) * (100 if m.group(2) == "CR" else 1)


def load_sas_ds_jobs(path: Path) -> pd.DataFrame:
    """SAS `DataScience Jobs.csv`: company x title aggregates; salaries '7.8L' -> 7.8 (lakh/yr)."""
    df = pd.read_csv(path)
    for c in ("avg_salary", "min_salary", "max_salary"):
        df[c.replace("salary", "lakh")] = df[c].map(_lakh)
    df["job_title"] = df["job_title"].str.strip()
    df["company_name"] = df["company_name"].str.strip()
    return df


def stratified_sample(posts: pd.DataFrame, n: int, by: str = "segment", seed: int = 42) -> pd.DataFrame:
    """Equal-share sample per segment (capped by segment size), reproducible."""
    if n >= len(posts):
        return posts
    groups = posts.groupby(by)
    per = max(1, n // groups.ngroups)
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in groups]
    return pd.concat(parts).reset_index(drop=True)
