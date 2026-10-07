# Datasets

Raw taxonomy data for D2S Bharat. These files are **not** committed to git; re-download them using the steps below.

## O\*NET 30.3 (`onet/`)

Downloaded automatically from https://www.onetcenter.org/database.html

| File | Contents | Size |
|---|---|---|
| `db_30_3_text.zip` → `text/` | 45 tab-delimited text files | 13.2 MB zip → 98 MB |
| `db_30_3_mysql.zip` → `mysql/` | SQL scripts (MySQL/PostgreSQL-compatible) | 13.6 MB zip → 284 MB |

Key files for D2S: `Occupation Data.txt`, `Essential Skills.txt`, `Transferable Skills.txt`, `Software Skills.txt`, `Knowledge.txt`, `Task Statements.txt`, `Emerging Tasks.txt`, `Job Titles.txt`, `Job Zones.txt`, `Education.txt`, `Related Occupations.txt`.

License: Creative Commons Attribution 4.0. Attribute O\*NET in the UI and docs.

## ESCO v1.2.x (`esco/`)

ESCO can't be downloaded with a script; the portal emails a link.

1. Go to https://esco.ec.europa.eu/en/use-esco/download
2. Choose **version:** latest v1.2.x, **content:** Classification, **language:** English (`en`), **format:** CSV
3. Accept the privacy notice, enter your email address, and submit
4. Open the link in the email, download the zip, and place it in `dataset/esco/`

**Downloaded:** v1.2.1, English CSV → extracted to `esco/v1.2.1/` (10 MB zip → 50 MB, 19 files).

| File | Rows |
|---|---|
| `skills_en.csv` | 13,960 skills |
| `occupations_en.csv` | 3,043 occupations |
| `occupationSkillRelations_en.csv` | 126,051 occupation↔skill links (essential/optional) |
| `skillSkillRelations_en.csv` | 5,818 skill↔skill links |

Also included: skill hierarchy, ISCO groups, and the digital / green / transversal / language / research skill collections.

Key files: `skills_en.csv`, `occupations_en.csv`, `occupationSkillRelations_en.csv`, `skillsHierarchy_en.csv`, `broaderRelationsSkillPillar_en.csv`, `ISCOGroups_en.csv`.
