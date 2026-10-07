---
name: d2s-data-sources
description: D2S Bharat uses ESCO + O*NET as its skill/occupation taxonomy sources (decided 2026-10-07); NCS/NQR/PLFS deferred
metadata:
  node_type: memory
  type: project
  originSessionId: 179abff8-7912-4c4b-9bae-f98badd5c47b
  modified: 2026-10-07T05:25:43.735Z
---

D2S Bharat (hackathon project, Team Code5urge) chose ESCO + O*NET as the taxonomy backbone on 2026-10-07. ESCO is the primary skill vocabulary; O*NET adds importance/level and hot-technology data. NSQF/NQR, NCS vacancies and PLFS are deferred to V2+.

**Why:** both are free and usable immediately; NCS needs a partnership and NQR has no public API.

**How to apply:** design ingestion, schema and extraction around ESCO/O*NET (bulk-load first, pin ESCO `selectedVersion`). Don't propose NCS/NQR for the MVP. Stack: Next.js frontend + Python/FastAPI backend. Spec lives in features.md.
