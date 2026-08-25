"""Everything that talks to a model. Previously duplicated in full across the
activity and task pipelines; there is now one copy.

Nothing here raises: a failed call returns None and the caller falls back to
code-rendered output, which is the whole failure policy of this codebase.

UNCHANGED from the version you supplied apart from un-escaping the `<-` log
arrows that HTML-escaping had turned into `&lt;-`.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import AGENT_TIMEOUT, FORMS_COST_PRINT_TO_LOG, MODEL
from .cost import calculate_cost, rates_for
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

# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates token usage across every call in one report run. Per-stage
    rows are kept so it is obvious which agent is costing money.

    Cost comes from `cost.calculate_cost` - the project's own function - called
    once per agent run and summed. Nothing here re-derives a price, so cached-input
    discounts and the pricing table are handled in exactly one place.
    """

    model: str = MODEL
    calls: list[dict] = field(default_factory=list)

    def add(self, stage: str, result: Any, seconds: float) -> dict:
        usage = extract_usage(result)
        row = {"stage": stage, "seconds": round(seconds, 2), **usage,
               "cost_usd": self._cost_of(stage, result)}
        self.calls.append(row)
        log.info("  usage[%s]: in=%s (cached %s) out=%s total=%s cost=$%s (%.2fs)",
                 stage, usage["input_tokens"], usage["cached_input_tokens"],
                 usage["output_tokens"], usage["total_tokens"],
                 "?" if row["cost_usd"] is None else f"{row['cost_usd']:.6f}", seconds)
        return row

    def _cost_of(self, stage: str, result: Any) -> float | None:
        """Run the project's `calculate_cost` on one result.

        It prints its totals by design; that output is captured and re-emitted
        through the logger so importing this module never turns a service's stdout
        into a token log. The function itself is used exactly as written.
        """
        try:
            if not FORMS_COST_PRINT_TO_LOG:
                return float(calculate_cost(result, model=self.model))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                cost = float(calculate_cost(result, model=self.model))
            for line in buffer.getvalue().splitlines():
                if line.strip():
                    log.info("  cost[%s]: %s", stage, line.strip())
            return cost
        except Exception as exc:
            log.warning("  calculate_cost failed for %s: %s", stage, exc)
            return None

    def totals(self) -> dict:
        inp = sum(c["input_tokens"] for c in self.calls)
        cached = sum(c["cached_input_tokens"] for c in self.calls)
        out = sum(c["output_tokens"] for c in self.calls)
        costs = [c["cost_usd"] for c in self.calls if c["cost_usd"] is not None]
        return {
            "model": self.model,
            "llm_calls": len(self.calls),
            "input_tokens": inp,
            "cached_input_tokens": cached,
            "output_tokens": out,
            "total_tokens": inp + out,
            "estimated_cost_usd": round(sum(costs), 6),
            "cost_priced_calls": len(costs),
            "rates_per_1m_tokens": rates_for(self.model),
            "by_stage": self.calls,
        }


_EMPTY_USAGE = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0}


def extract_usage(result: Any) -> dict[str, int]:
    """Pull token counts out of a Runner result across SDK shapes. A missing
    usage object costs a metric, not the report.

    Cached input tokens are read too - they are what makes a repeated prompt
    prefix cheap, so they belong in the report next to the total.
    """
    def cached_of(usage: Any) -> int:
        details = (usage.get("input_tokens_details") if isinstance(usage, dict)
                   else getattr(usage, "input_tokens_details", None))
        if not details:
            return 0
        value = (details.get("cached_tokens") if isinstance(details, dict)
                 else getattr(details, "cached_tokens", 0))
        return int(value or 0)

    def read(usage: Any) -> dict[str, int] | None:
        if usage is None:
            return None
        get = ((lambda k: usage.get(k)) if isinstance(usage, dict)
               else (lambda k: getattr(usage, k, None)))
        inp = get("input_tokens") or get("prompt_tokens") or 0
        out = get("output_tokens") or get("completion_tokens") or 0
        tot = get("total_tokens") or (inp + out)
        if inp or out or tot:
            return {"input_tokens": int(inp), "cached_input_tokens": cached_of(usage),
                    "output_tokens": int(out), "total_tokens": int(tot)}
        return None

    try:
        for path in (lambda r: getattr(getattr(r, "context_wrapper", None), "usage", None),
                     lambda r: getattr(r, "usage", None)):
            found = read(path(result))
            if found:
                return found
        acc = dict(_EMPTY_USAGE)
        for raw in getattr(result, "raw_responses", None) or []:
            part = read(getattr(raw, "usage", None))
            if part:
                for key in acc:
                    acc[key] += part[key]
        if any(acc.values()):
            return acc
    except Exception as exc:
        log.warning("  could not read token usage: %s", exc)
    return dict(_EMPTY_USAGE)


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
                               fallback: Any,
                               wording: Callable[[str], list[str]] | None = None
                               ) -> tuple[Any, list[str]]:
    """Run one agent, check what it wrote, and give it exactly one chance to repair.

    Two things are checked: every figure must trace back to what the agent was
    given, and - when a `wording` checker is supplied - the text must be free of
    internal names and statistical jargon. Both are folded into the same repair
    pass, so a run with one problem of each still costs one extra call, not two.
    """
    from .prompts import prompt_repair_input

    raw = await run_agent(name, instructions, user_input, usage)
    if not raw:
        return fallback, []

    def check(text: str) -> tuple[list[str], list[str]]:
        return grounding_check(text, allowed), (wording(text) if wording else [])

    bad, jargon = check(raw)
    if bad or jargon:
        log.warning("  grounding[%s]: %s unsupported figure(s) %s, %s jargon term(s) %s"
                    " - running repair pass", name, len(bad), sorted(set(bad))[:10],
                    len(jargon), sorted(set(jargon))[:10])
        repaired = await run_agent(f"{name}_Repair", instructions,
                                   await prompt_repair_input(bad, raw, jargon), usage)
        if repaired:
            raw = repaired
            bad, jargon = check(raw)
    if bad or jargon:
        log.warning("  grounding[%s]: still unsupported figure(s) %s / jargon %s",
                    name, sorted(set(bad))[:10], sorted(set(jargon))[:10])
    else:
        log.info("  grounding[%s]: OK - figures traced, wording clean", name)

    return (parse_json_output(raw, fallback) if json_mode else strip_fences(raw)), bad
