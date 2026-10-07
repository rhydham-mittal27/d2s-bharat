---
name: d2s-no-llm-api-own-slm
description: D2S Bharat uses no third-party LLM API; explanation layer = own fine-tuned self-hosted SLM (decided 2026-10-07)
metadata:
  node_type: memory
  type: project
  originSessionId: 179abff8-7912-4c4b-9bae-f98badd5c47b
  modified: 2026-10-07T06:07:07.056Z
---

D2S Bharat will NOT call any external LLM API (Claude, OpenAI, etc.) or use Langfuse-as-a-service. Decision briefs come from the team's own small language model: an open 1–4B base fine-tuned with LoRA (Unsloth), served self-hosted (llama.cpp GGUF). Deterministic template briefs are the MVP and the permanent fallback when the SLM fails number-faithfulness checks.

**Why:** user's decision on 2026-10-07: "we will not use llm we will train our own slm".

**How to apply:** never propose LLM APIs or hosted LLM tooling for this project. "Train" = fine-tune an open base, not pre-train (flag that if from-scratch comes up). Related: [[d2s-data-sources]].
