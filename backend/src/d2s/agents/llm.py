"""The local model, used only for constrained choices.

A 1.7B model on a laptop CPU picks the right tool reliably when its answer is forced into a JSON
schema whose fields are enums (Ollama structured output), but it invents arguments when calling tools
freely. So the model never produces numbers: it chooses among options we give it, and our code reads
numbers and course names from the user's text (d2s.agents.extract).

``Chooser`` is the seam the agents depend on; tests pass a scripted chooser instead of Ollama.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel

from d2s.config import get_settings

log = logging.getLogger(__name__)

# (system prompt, user text, response schema) -> parsed response, or None if the model is unavailable
Chooser = Callable[[str, str, type[BaseModel]], BaseModel | None]


@dataclass
class TokenMeter:
    """Exact token counts reported by Ollama for every model call in one agent run.

    input_tokens = prompt tokens the model evaluated (Ollama reuses its cache for a repeated prefix, so a
    second identical system prompt can count fewer); output_tokens = generated tokens.
    """

    calls: list[dict] = field(default_factory=list)

    def add(self, purpose: str, input_tokens: int, output_tokens: int, ms: float) -> None:
        self.calls.append({"purpose": purpose, "input_tokens": input_tokens, "output_tokens": output_tokens,
                           "total_tokens": input_tokens + output_tokens, "ms": round(ms, 1)})

    def summary(self) -> dict:
        i = sum(c["input_tokens"] for c in self.calls)
        o = sum(c["output_tokens"] for c in self.calls)
        return {"model_calls": len(self.calls), "input_tokens": i, "output_tokens": o, "total_tokens": i + o,
                "calls": self.calls}


def ollama_status() -> dict[str, Any]:
    s = get_settings()
    out: dict[str, Any] = {"enabled": s.agent_llm_enabled, "model": s.agent_model, "url": s.ollama_url}
    if not s.agent_llm_enabled:
        return {**out, "available": False, "reason": "disabled (D2S_AGENT_LLM_ENABLED=false)"}
    try:
        tags = httpx.get(f"{s.ollama_url}/api/tags", timeout=2).json()
        names = {m["name"] for m in tags.get("models", [])}
        loaded = {m["name"] for m in httpx.get(f"{s.ollama_url}/api/ps", timeout=2).json().get("models", [])}
        ok = s.agent_model in names
        return {**out, "available": ok, "loaded": s.agent_model in loaded,
                "reason": None if ok else f"model not pulled: run `ollama pull {s.agent_model}`"}
    except Exception as exc:  # noqa: BLE001  (Ollama not running)
        return {**out, "available": False, "reason": f"Ollama not reachable at {s.ollama_url}: {type(exc).__name__}"}


@lru_cache(maxsize=1)
def _chat():
    from langchain_ollama import ChatOllama

    s = get_settings()
    return ChatOllama(model=s.agent_model, base_url=s.ollama_url, reasoning=False, temperature=0,
                      num_ctx=2048, keep_alive=s.agent_keep_alive,
                      client_kwargs={"timeout": s.agent_llm_timeout_s})


def ollama_chooser(system: str, user: str, schema: type[BaseModel], meter: TokenMeter | None = None
                   ) -> BaseModel | None:
    """Constrained choice from the local model; None when the model is off or fails (callers fall back).
    With a meter, the call's exact token counts are recorded."""
    if not get_settings().agent_llm_enabled:
        return None
    try:
        t = time.perf_counter()
        out = _chat().with_structured_output(schema, method="json_schema", include_raw=True).invoke(
            [("system", system), ("user", user)])
        ms = (time.perf_counter() - t) * 1000
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        if meter is not None:
            meter.add(schema.__name__, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), ms)
        log.info("llm %s in %.0f ms, tokens %s: %s", schema.__name__, ms, usage, out["parsed"])
        return out["parsed"]
    except Exception:  # any model or transport failure -> deterministic fallback
        log.exception("local model call failed; falling back to deterministic behaviour")
        return None


def metered(meter: TokenMeter) -> Chooser:
    """The local-model chooser, recording every call's tokens into `meter`."""
    return lambda system, user, schema: ollama_chooser(system, user, schema, meter)


def warm_up() -> None:
    """Load the model into memory at server start so the first question does not pay the load time."""
    if not get_settings().agent_llm_enabled:
        return
    try:
        s = get_settings()
        httpx.post(f"{s.ollama_url}/api/chat", timeout=60, json={
            "model": s.agent_model, "messages": [{"role": "user", "content": "ok"}], "stream": False,
            "think": False, "keep_alive": s.agent_keep_alive, "options": {"num_predict": 1}})
    except Exception:  # noqa: BLE001
        log.info("agent model warm-up skipped (Ollama not reachable)")
