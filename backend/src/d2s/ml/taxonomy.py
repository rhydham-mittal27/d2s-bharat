"""Load O*NET 30.3 (and ESCO v1.2 CSV, once downloaded) into plain taxonomy records.

O*NET 30.3 split the old "Skills"/"Technology Skills" files; we read the new layout:
Essential Skills, Transferable Skills, Knowledge (IM/LV-rated) and Software Skills (tools,
with Hot Technology / In Demand flags).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from d2s.config import get_settings

ONET_RATED_FILES = {
    "Essential Skills.txt": "essential_skill",
    "Transferable Skills.txt": "transferable_skill",
    "Knowledge.txt": "knowledge",
}


@dataclass
class Skill:
    id: str
    label: str
    source: str  # "onet" | "esco"
    kind: str
    description: str = ""
    alt_labels: list[str] = field(default_factory=list)
    hot_technology: bool = False
    in_demand: bool = False

    @property
    def embedding_text(self) -> str:
        return f"{self.label}. {self.description}".strip(". ") if self.description else self.label


@dataclass
class Occupation:
    id: str
    title: str
    description: str
    source: str


@dataclass
class OccupationSkill:
    occupation_id: str
    skill_id: str
    importance: float | None = None  # O*NET IM scale 1-5
    level: float | None = None  # O*NET LV scale 0-7
    relation: str = "essential"  # ESCO: essential | optional


@dataclass
class Taxonomy:
    skills: dict[str, Skill]
    occupations: dict[str, Occupation]
    links: list[OccupationSkill]
    version: str

    def merge(self, other: "Taxonomy") -> "Taxonomy":
        return Taxonomy(
            skills={**self.skills, **other.skills},
            occupations={**self.occupations, **other.occupations},
            links=[*self.links, *other.links],
            version=f"{self.version}+{other.version}",
        )


def _read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8", keep_default_na=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1", keep_default_na=False)


def find_onet_dir(dataset_dir: Path | None = None) -> Path:
    root = (dataset_dir or get_settings().dataset_dir) / "onet"
    hits = sorted(root.glob("**/Occupation Data.txt"))
    if not hits:
        raise FileNotFoundError(f"O*NET text files not found under {root}; see dataset/README.md")
    return hits[0].parent


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_onet(onet_dir: Path | None = None, include_tools: bool = True) -> Taxonomy:
    d = onet_dir or find_onet_dir()
    occ = _read_tsv(d / "Occupation Data.txt")
    occupations = {
        f"onet:{r['O*NET-SOC Code']}": Occupation(
            id=f"onet:{r['O*NET-SOC Code']}", title=r["Title"], description=r["Description"], source="onet"
        )
        for r in occ.to_dict("records")
    }

    ref = _read_tsv(d / "Content Model Reference.txt")
    descriptions = dict(zip(ref["Element ID"], ref["Description"], strict=True))

    skills: dict[str, Skill] = {}
    links: list[OccupationSkill] = []
    for fname, kind in ONET_RATED_FILES.items():
        path = d / fname
        if not path.exists():
            continue
        df = _read_tsv(path)
        df = df[df["Recommend Suppress"] != "Y"]
        for eid, name in df[["Element ID", "Element Name"]].drop_duplicates().itertuples(index=False):
            sid = f"onet:{eid}"
            skills[sid] = Skill(id=sid, label=name, source="onet", kind=kind,
                                description=descriptions.get(eid, ""))
        pivot = df.pivot_table(
            index=["O*NET-SOC Code", "Element ID"], columns="Scale ID", values="Data Value",
            aggfunc="first",
        )
        for (code, eid), row in pivot.iterrows():
            links.append(OccupationSkill(
                occupation_id=f"onet:{code}", skill_id=f"onet:{eid}",
                importance=_to_float(row.get("IM")), level=_to_float(row.get("LV")),
            ))

    if include_tools and (d / "Software Skills.txt").exists():
        tools = _read_tsv(d / "Software Skills.txt")
        for example, group in tools.groupby("Workplace Example"):
            sid = f"onet:tool:{_slug(example)}"
            skills[sid] = Skill(
                id=sid, label=example, source="onet", kind="tool",
                description=group["Element Name"].iloc[0],
                hot_technology=(group["Hot Technology"] == "Y").any(),
                in_demand=(group["In Demand"] == "Y").any(),
            )
            for code in group["O*NET-SOC Code"].unique():
                links.append(OccupationSkill(occupation_id=f"onet:{code}", skill_id=sid))

    return Taxonomy(skills=skills, occupations=occupations, links=links, version=_onet_version(d))


def _onet_version(d: Path) -> str:
    m = re.search(r"db_(\d+)_(\d+)", str(d))
    return f"onet-{m.group(1)}.{m.group(2)}" if m else "onet"


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_esco(esco_dir: Path | None = None, lang: str = "en") -> Taxonomy:
    """Read the ESCO CSV classification package (download per dataset/README.md)."""
    root = esco_dir or (get_settings().dataset_dir / "esco")
    hits = sorted(root.glob(f"**/skills_{lang}.csv"))
    if not hits:
        raise FileNotFoundError(f"ESCO skills_{lang}.csv not found under {root}; see dataset/README.md")
    d = hits[0].parent
    read = lambda name: pd.read_csv(d / name, dtype=str, keep_default_na=False)  # noqa: E731

    skills = {
        r["conceptUri"]: Skill(
            id=r["conceptUri"], label=r["preferredLabel"], source="esco",
            kind=r.get("skillType") or "skill", description=r.get("description", ""),
            alt_labels=[a for a in r.get("altLabels", "").split("\n") if a],
        )
        for r in read(f"skills_{lang}.csv").to_dict("records")
    }
    occupations = {
        r["conceptUri"]: Occupation(id=r["conceptUri"], title=r["preferredLabel"],
                                    description=r.get("description", ""), source="esco")
        for r in read(f"occupations_{lang}.csv").to_dict("records")
    }
    links = [
        OccupationSkill(occupation_id=r["occupationUri"], skill_id=r["skillUri"],
                        relation=r.get("relationType", "essential"))
        for r in read(f"occupationSkillRelations_{lang}.csv").to_dict("records")
    ]
    m = re.search(r"v?(\d+\.\d+\.\d+)", str(d))
    return Taxonomy(skills=skills, occupations=occupations, links=links,
                    version=f"esco-{m.group(1)}" if m else "esco")
