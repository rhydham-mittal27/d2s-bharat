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


def stratified_sample(posts: pd.DataFrame, n: int, by: str = "segment", seed: int = 42) -> pd.DataFrame:
    """Equal-share sample per segment (capped by segment size), reproducible."""
    if n >= len(posts):
        return posts
    groups = posts.groupby(by)
    per = max(1, n // groups.ngroups)
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in groups]
    return pd.concat(parts).reset_index(drop=True)
