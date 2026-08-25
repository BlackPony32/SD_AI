"""The two prompts used by the activity report.

  [A] Statistics Analyst -- reads the calculated statistics and produces the
      seven key points, plus a one-paragraph reading of each table.
  [B] Situation Writer   -- writes the narrative: executive summary, what the
      activity shows, who is doing the work, what looks wrong, what to do.

A explains the tables the reader is looking at; B explains the situation those
tables describe. Neither waits on the other, so wall clock is one call, and a
failure in one still leaves the report with the other half intact.

Beyond the shared rules, both agents are held to the confidence rules: the
deterministic layer attaches `reliability` and `basis` to every comparison, and
ignoring either is a rule violation rather than a style preference -- "activity
fell 62%" off three events is the most likely way this report misleads someone.
"""

from __future__ import annotations

from ..core.prompts import build_rules, json_payload

# Dropped before the metrics reach a model: long, and low-signal for narrative
# purposes. Every token here is paid on every run.
_PROMPT_DROP = {
    "temporal": ("hourly_counts", "day_of_week_counts", "monthly_counts"),
    "metadata": ("columns",),
}
_MAX_SALESPEOPLE_IN_PROMPT = 12
_MAX_REPRESENTATIVES_IN_PROMPT = 10
_REPRESENTATIVE_FIELDS = ("label", "kind", "events", "share_of_all_events_pct",
                          "events_last_90d", "status", "days_since_last_event",
                          "main_activity", "main_activity_share_pct",
                          "activity_diversity_index", "median_days_between_events")
_SALESPERSON_FIELDS = ("salesperson", "orders_total", "revenue_total", "avg_order_value",
                       "median_order_value", "share_of_revenue_pct", "orders_last_90d",
                       "revenue_last_90d", "orders_last_30d", "revenue_last_30d",
                       "days_since_last_order", "revenue_mom")

_RULES = build_rules("no_invented_numbers", "respect_reliability", "respect_basis",
                     "human_labels", "channel_is_not_a_person", "dull_truth")


def _pick(rows: list[dict], fields: tuple[str, ...], limit: int) -> list[dict]:
    return [{k: r.get(k) for k in fields} for r in rows[:limit]]


def trim_metrics(metrics: dict) -> dict:
    """Compact copy of the metrics for prompt use. Never mutates the original."""
    out: dict = {}
    for section, body in metrics.items():
        if not isinstance(body, dict):
            out[section] = body
            continue
        trimmed = {k: v for k, v in body.items() if k not in _PROMPT_DROP.get(section, ())}
        if section == "representatives" and isinstance(trimmed.get("top_representatives"), list):
            trimmed["top_representatives"] = _pick(trimmed["top_representatives"],
                                                   _REPRESENTATIVE_FIELDS,
                                                   _MAX_REPRESENTATIVES_IN_PROMPT)
        if section == "salesperson_orders" and isinstance(trimmed.get("by_salesperson"), list):
            trimmed["salespeople_total"] = len(trimmed["by_salesperson"])
            trimmed["by_salesperson"] = _pick(trimmed["by_salesperson"], _SALESPERSON_FIELDS,
                                              _MAX_SALESPEOPLE_IN_PROMPT)
        out[section] = trimmed
    return out


async def prompt_statistics_analyst(metrics: dict, table_keys: list[str],
                                    target: int = 7) -> str:
    """Agent A. Calculated statistics in; key points and per-table readings out.
    Doubles as the allow-list for its grounding check, so every figure it may
    write is inside this string."""
    keys = ", ".join(table_keys) or "none"
    return f"""You are a data analyst for a B2B distribution business. Below is a \
statistics package computed from the platform's activity log (what users did in the \
system and when) and its order book (orders and revenue per salesperson).

The reader is looking at these tables printed above your text. Your job is twofold: \
pull out the {target} points that matter most, and write a short reading of each table so \
the reader knows what they are looking at. Output JSON only.

{_RULES}

For the key points, cover the ground rather than {target} variations on one theme. \
Draw on: overall volume and its direction (`overview`, `temporal`), which activities \
grew or stopped (`activity_types`), sales concentration and who is actually selling \
(`salesperson_orders`), who generates the system activity (`representatives`), gaps \
between work opened and work closed (`workflow`), signs that rows are machine-generated \
rather than human (`automation`), and anything in `data_quality` with severity `high` \
that limits what can be concluded at all.

STATISTICS PACKAGE:
{json_payload(trim_metrics(metrics), "metrics")}

TABLES SHOWN TO THE READER (write one note for each key): {keys}

Return exactly this JSON and nothing else:
{{
  "key_points": [
    {{
      "point": "one sentence, specific, carrying the figure it rests on",
      "why_it_matters": "one sentence on the consequence for the business",
      "confidence": "high|medium|low"
    }}
  ],
  "table_notes": {{
    "<table key>": "2-3 sentences reading that table: the pattern, the exception \
worth noticing, and any reason not to over-read it. No bullet points, no headers."
  }}
}}
Exactly {target} entries in `key_points`, ordered most important first. One note per \
table key listed above, and no keys that were not listed."""


async def prompt_situation_writer(key_facts: str, representatives: list[dict],
                                  data_quality: list[dict] | None = None) -> str:
    """Agent B. Verified fact sheet in; the current-situation narrative out. The
    fact sheet is both its material and its allow-list, so a figure absent from
    it is flagged automatically."""
    blockers = [n for n in (data_quality or []) if n.get("severity") == "high"]
    blocker_text = "\n".join(f"- {n['message']}" for n in blockers) or "- none"
    people = json_payload(_pick(representatives or [], _REPRESENTATIVE_FIELDS,
                                _MAX_REPRESENTATIVES_IN_PROMPT), "representatives")

    return f"""You are writing the analysis half of an activity report for the manager \
of a B2B distribution business. They are not an analyst. They have just read the \
statistics tables; your job is to tell them what the situation is, what it means, and \
what to do -- in that order, in plain language.

VERIFIED FACT SHEET (every figure here is safe to use; nothing else is):
{key_facts}

WHO GENERATES THE ACTIVITY:
{people}

MUST BE STATED SOMEWHERE IN YOUR TEXT (these limit what the numbers can mean):
{blocker_text}

{_RULES}

Write markdown with exactly these five sections, in this order, and nothing else:

**Executive summary** -- 2-3 sentences. The single thing that matters, with its figure. \
If the honest answer is "activity has nearly stopped and the recent numbers are too thin \
to read", say that.

**What the activity shows** -- 3-5 bullets on volume, mix and direction. Each bullet \
carries a figure from the fact sheet. Hedge every low-confidence comparison inside the \
same bullet, not in a footnote.

**Who is doing the work** -- 2-4 bullets on representatives, channels and salespeople. If \
attribution is too weak for per-person conclusions about system activity, lead with that \
and use the order book's salesperson figures instead of ranking channel buckets.

**What looks wrong** -- 2-4 bullets: work opened but not closed, risk signals, \
machine-generated rows, anything that makes a number mean less than it appears to.

**Do this week** -- exactly 3 numbered actions. Each names who or what to check and what a \
good answer would look like. Prefer "verify X before trusting Y" over generic advice such \
as "improve engagement".

Style: no preamble, no closing summary, no tables, no invented context about the \
business. Short sentences. Bold only the five section headers. The statistics tables are \
printed above your text, so do not reproduce them."""
