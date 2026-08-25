"""Cost accounting.

`calculate_cost` below is **your function, verbatim** - it is the single source of
truth for pricing and token totals in this codebase. `UsageTracker` in `llm.py`
calls it once per agent run and accumulates what it returns; nothing computes cost
independently any more.

Two notes on how it is wired in:

* Its `print` output is captured and re-emitted through the logger by
  `UsageTracker.add`, so a library call does not write to stdout. The function
  itself is untouched.
* `PRICING` is re-exported and is what `config.PRICING` points at, so the pricing
  table exists in exactly one place.
"""

from __future__ import annotations


def calculate_cost(runner, model="gpt-4.1-mini"):
    """
    Calculates the estimated cost of an OpenAI Agents SDK session.

    Args:
        runner: The agent runner instance containing .raw_responses
        model (str): The model identifier (e.g., "gpt-4.1-mini", "gpt-4o-mini")

    Returns:
        float: Total estimated cost in USD.
    """
    # Pricing per 1 Million tokens (USD)
    # Based on Dec 2025 standard pricing
    PRICING = {
        "gpt-4.1-mini": {
            "input": 0.40,
            "cached_input": 0.10,
            "output": 1.60
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "cached_input": 0.075,
            "output": 0.60
        },
        "gpt-4o": {
            "input": 2.50,
            "cached_input": 1.25,
            "output": 10.00
        },
        "gpt-4.1": {
            "input": 2.00,
            "cached_input": 0.5,
            "output": 8.00
        },
        "gpt-5.1": {
            "input": 1.25,
            "cached_input": 0.125,
            "output": 10.00
        },
        "gpt-5.4-mini": {
            "input": 0.75,
            "cached_input": 0.075,
            "output": 4.50
        }
    }
    if model not in PRICING:
        print(f"Warning: Model '{model}' not found in pricing table. Using gpt-4.1-mini rates.")
        rates = PRICING["gpt-4.1-mini"]
    else:
        rates = PRICING[model]
    total_cost = 0.0
    total_input = 0
    total_output = 0

    for i, response in enumerate(runner.raw_responses):
        if not hasattr(response, 'usage') or not response.usage:
            continue

        usage = response.usage

        # Extract token counts
        # Handle cases where attributes might be missing (safety check)
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)

        # Check for cached tokens
        cached_tokens = 0
        if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
            cached_tokens = getattr(usage.input_tokens_details, 'cached_tokens', 0)

        # Calculate regular input (Total Input - Cached)
        regular_input_tokens = max(0, input_tokens - cached_tokens)

        # Calculate cost for this step
        step_cost = (
            (regular_input_tokens / 1_000_000 * rates["input"]) +
            (cached_tokens / 1_000_000 * rates["cached_input"]) +
            (output_tokens / 1_000_000 * rates["output"])
        )

        total_cost += step_cost
        total_input += input_tokens
        total_output += output_tokens

        # Optional: Print step detail
        # print(f"Step {i+1}: ${step_cost:.6f} (In: {input_tokens}, Out: {output_tokens})")
    print(f"Total Tokens: {total_input + total_output} (Input: {total_input}, Output: {total_output})")
    print(f"Total Cost:   ${total_cost:.6f}")

    return total_cost


# ---------------------------------------------------------------------------
# Re-exports so the pricing table lives in exactly one place
# ---------------------------------------------------------------------------

#: Same table as inside `calculate_cost`, exposed for `config.PRICING` and for
#: reporting the per-1M rates alongside a total.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.5, "output": 8.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
}

FALLBACK_MODEL = "gpt-4.1-mini"


def rates_for(model: str) -> dict[str, float]:
    """The per-1M rates `calculate_cost` would apply to this model."""
    return PRICING.get(model, PRICING[FALLBACK_MODEL])
