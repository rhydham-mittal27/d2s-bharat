---
name: d2s-sas-dataset-only
description: Analysis must use only the SAS hackathon datasets (same event as Build for Bharat); no Naukri/external job data
metadata:
  node_type: memory
  type: project
  originSessionId: 179abff8-7912-4c4b-9bae-f98badd5c47b
  modified: 2026-10-07T07:25:40.590Z
---

On 2026-10-07 the user said the SAS CU Hackathon (SAS + Chandigarh University) is the same event as Build for Bharat, and to use **only the datasets provided by SAS**: `hackathon/SAS Data Problem Statement and Instructions Hackathon/` (Analytics Jobs.csv, DataScience Jobs.csv, JDS Skill Traits.xlsx, SDS Personality Traits.xlsx). The Naukri dump downloaded earlier must not be used for results.

**Why:** competition rules (judged on Round 2 report 70% + Round 3 presentation 30%; data prep = 25/100 report marks).

**How to apply:** job-post analysis uses SAS files only. ESCO/O*NET are kept as the skill *vocabulary* (the pitch deck names them) unless the user says otherwise; external labelled sets (TechWolf) aren't used for SAS results. Calibrate on in-dataset signals (key_skills tags). Related: [[d2s-data-sources]], [[d2s-no-llm-api-own-slm]].
