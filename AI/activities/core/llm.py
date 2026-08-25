"""Everything that talks to a model. Previously duplicated in full across the
activity and task pipelines; there is now one copy.

Nothing here raises: a failed call returns None and the caller falls back to
code-rendered output, which is the whole failure policy of this codebase.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AGENT_TIMEOUT, MODEL, PRICING
from .grounding import grounding_check
from .logging_setup import get_log

log = get_log("llm")

try:
    from agents import Agent, AsyncOpenAI, OpenAIResponsesModel, Runner

    llm_model = OpenAIResponsesModel(model=MODEL, openai_client=AsyncOpenAI())
except Exception as exc:  # SDK or credentials missing -> every call falls back
    Agent = Runner = None  # type: ignore[assignment]
    llm_model = None
    log.warning("agents SDK unavailable (%s); reports will be code-rendered", exc)

try:
    from AI.utils import calculate_cost  # type: ignore
except Exception:
    calculate_cost = None


# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates token usage across every call in one report run. Per-stage
    rows are kept so it is obvious which agent is costing money."""

    model: str = MODEL
    calls: list[dict] = field(default_factory=list)

    def add(self, stage: str, result: Any, seconds: float) -> dict:
        usage = extract_usage(result)
        row = {"stage": stage, "seconds": round(seconds, 2), **usage}
        self.calls.append(row)
        log.info("  usage[%s]: in=%s out=%s total=%s (%.2fs)", stage, usage["input_tokens"],
                 usage["output_tokens"], usage["total_tokens"], seconds)
        if calculate_cost:
            try:
                calculate_cost(result, model=self.model)
            except Exception as exc:
                log.warning("  calculate_cost failed for %s: %s", stage, exc)
        return row

    def totals(self) -> dict:
        inp = sum(c["input_tokens"] for c in self.calls)
        out = sum(c["output_tokens"] for c in self.calls)
        rate_in, rate_out = PRICING.get(self.model, PRICING["default"])
        return {
            "model": self.model,
            "llm_calls": len(self.calls),
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "estimated_cost_usd": round(inp / 1e6 * rate_in + out / 1e6 * rate_out, 6),
            "by_stage": self.calls,
        }


def extract_usage(result: Any) -> dict[str, int]:
    """Pull token counts out of a Runner result across SDK shapes. A missing
    usage object costs a metric, not the report."""
    def read(usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None
        get = ((lambda k: usage.get(k)) if isinstance(usage, dict)
               else (lambda k: getattr(usage, k, None)))
        inp = get("input_tokens") or get("prompt_tokens") or 0
        out = get("output_tokens") or get("completion_tokens") or 0
        tot = get("total_tokens") or (inp + out)
        if inp or out or tot:
            return {"input_tokens": int(inp), "output_tokens": int(out), "total_tokens": int(tot)}
        return None

    try:
        for path in (lambda r: getattr(getattr(r, "context_wrapper", None), "usage", None),
                     lambda r: getattr(r, "usage", None)):
            found = read(path(result))
            if found:
                return found
        acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for raw in getattr(result, "raw_responses", None) or []:
            part = read(getattr(raw, "usage", None))
            if part:
                for key in acc:
                    acc[key] += part[key]
        if any(acc.values()):
            return acc
    except Exception as exc:
        log.warning("  could not read token usage: %s", exc)
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def run_agent(name: str, instructions: str, user_input: str, usage: UsageTracker,
                    timeout: int = AGENT_TIMEOUT) -> str | None:
    """One guarded call: build the agent, run it under a timeout, record usage,
    return the text or None."""
    if Agent is None or llm_model is None:
        log.error("  <- agent %s skipped: no model client available", name)
        return None

    started = time.perf_counter()
    try:
        log.info("  -> agent %s starting (instructions %s chars)", name, len(instructions))
        agent = Agent(name=name, instructions=instructions, model=llm_model)
        result = await asyncio.wait_for(Runner.run(agent, input=user_input), timeout=timeout)
        usage.add(name, result, time.perf_counter() - started)
        output = getattr(result, "final_output", None)
        if not output:
            log.warning("  <- agent %s returned empty output", name)
            return None
        log.info("  <- agent %s OK (%s chars, %.2fs)", name, len(str(output)),
                 time.perf_counter() - started)
        return str(output)
    except asyncio.TimeoutError:
        log.error("  <- agent %s TIMED OUT after %ss", name, timeout)
    except Exception as exc:
        log.exception("  <- agent %s FAILED: %s: %s", name, type(exc).__name__, exc)
    return None


FENCE_RE = re.compile(r"^\s*```(?:json|markdown|md)?|```\s*$", re.M)


def strip_fences(raw: str) -> str:
    return FENCE_RE.sub("", str(raw or "")).strip()


def parse_json_output(raw: str | None, fallback: Any = None) -> Any:
    """Models add fences and prose whatever the prompt says. Strip fences, then
    fall back to the outermost {...} span, then to `fallback`."""
    if not raw:
        return fallback
    text = strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    log.warning("  JSON parse failed (%s chars); using fallback", len(text))
    return fallback


async def write_with_grounding(name: str, instructions: str, user_input: str, allowed: str,
                               usage: UsageTracker, json_mode: bool,
                               fallback: Any) -> tuple[Any, list[str]]:
    """Run one agent, check every figure it wrote against what it was given, and
    give it exactly one chance to repair the ones that do not trace back."""
    from .prompts import prompt_repair_input

    raw = await run_agent(name, instructions, user_input, usage)
    if not raw:
        return fallback, []

    bad = grounding_check(raw, allowed)
    if bad:
        log.warning("  grounding[%s]: %s unsupported figure(s) %s - running repair pass",
                    name, len(bad), sorted(set(bad))[:10])
        repaired = await run_agent(f"{name}_Repair", instructions,
                                   await prompt_repair_input(bad, raw), usage)
        if repaired:
            raw = repaired
            bad = grounding_check(raw, allowed)
    if bad:
        log.warning("  grounding[%s]: %s figure(s) still unsupported: %s",
                    name, len(bad), sorted(set(bad))[:10])
    else:
        log.info("  grounding[%s]: OK - every figure traced to the inputs", name)

    return (parse_json_output(raw, fallback) if json_mode else strip_fences(raw)), bad
