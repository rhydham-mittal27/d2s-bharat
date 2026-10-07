# Datasets

Raw taxonomy data for D2S Bharat. These files are **not** committed to git; re-download them using the steps below.

## O\*NET 30.3 (`onet/`)

Downloaded automatically from https://www.onetcenter.org/database.html

| File | Contents | Size |
|---|---|---|
| `db_30_3_text.zip` | 45 tab-delimited text files | 13.2 MB |
| `db_30_3_mysql.zip` | SQL scripts (MySQL/PostgreSQL-compatible) | 13.6 MB |

License: Creative Commons Attribution 4.0. Attribute O\*NET in the UI and docs.

## ESCO v1.2.x (`esco/`)

ESCO can't be downloaded with a script; the portal emails a link.

1. Go to https://esco.ec.europa.eu/en/use-esco/download
2. Choose **version:** latest v1.2.x, **content:** Classification, **language:** English (`en`), **format:** CSV
3. Accept the privacy notice, enter your email address, and submit
4. Open the link in the email, download the zip, and place it in `dataset/esco/`

Expected key files: `skills_en.csv`, `occupations_en.csv`, `occupationSkillRelations_en.csv`, `skillsHierarchy_en.csv`, `broaderRelationsSkillPillar_en.csv`, `ISCOGroups_en.csv`.
