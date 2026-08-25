"""Getting something out of free-text answers.

Reporting "answered this question: 100%" for a required text field is not
analysis - it is arithmetic on a field that cannot be anything else. Five of the
twelve questions in the reference form were reported that way, each with a table
of fourteen 100%s. Everything in this module exists to replace that with the
content of the answers.

Four readers, tried in order of how much they tell you:

1. **Amounts.** "Did you secure an order? If so, how much?" is where the revenue
   is. `extract_amounts` pulls currency and quantities out of prose, which turns a
   text field into a number series: how often an order was secured, the typical
   size, the total.
2. **Requirements.** "List the name & role of the employees you educated (3 people
   minimum)" states its own pass mark. `parse_requirement` reads the minimum out
   of the question text and `count_items` checks each answer against it, giving a
   compliance rate instead of a response rate.
3. **Substance.** Failing the above, how many answers actually say something -
   measured against a word floor and against the templating check, not just
   against emptiness.
4. **Themes.** What the answers are about: keyword groups with counts and one
   example each, which is more use than a list of word frequencies.

And one check that runs regardless:

**Templating.** Every answer in the reference export was `"Text answer #N to Q"` -
all distinct, so duplicate-rate read 0% and nothing flagged it. Replacing digit
runs with `#` collapses them to one signature covering 100% of answers, which is
conclusive: these were generated, not written. Real forms show the same pattern
when people paste boilerplate.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from ..core.logging_setup import get_log

log = get_log("forms.textual")

# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

_CURRENCY = r"[$€£₴]|usd|eur|gbp|uah|dollars?|euros?"
_MULTIPLIER = {"k": 1_000, "к": 1_000, "m": 1_000_000, "thousand": 1_000,
               "million": 1_000_000}

# $1,200 / 1 200 грн / 12.5k USD / 300 dollars
_AMOUNT = re.compile(
    rf"(?:(?P<pre>{_CURRENCY})\s*)?"
    r"(?P<num>\d{1,3}(?:[ ,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
    rf"\s*(?P<mult>k|m|thousand|million)?\s*(?P<post>{_CURRENCY})?",
    re.I)

_QUANTITY = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>cases?|units?|packs?|boxes|bottles?|cans?|pallets?|pcs?|items?|"
    r"кейс\w*|ящик\w*|шт)",
    re.I)

_NEGATIVE = re.compile(
    r"\b(no|none|nothing|not yet|no order|didn'?t|did not|n/?a|zero|nope|немає|нет)\b",
    re.I)
_AFFIRMATIVE = re.compile(r"\b(yes|yeah|secured|placed|ordered|confirmed|так|да)\b",
                          re.I)


def _to_number(raw: str, multiplier: str | None) -> float | None:
    text = raw.replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif text.count(",") == 1 and re.search(r",\d{1,2}$", text):
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if multiplier:
        value *= _MULTIPLIER.get(multiplier.lower(), 1)
    return value


def extract_amounts(text: str) -> dict[str, Any]:
    """Money and quantities mentioned in one answer.

    `has_currency` distinguishes "$300" from a bare "300" - the second could be
    anything, so it is reported separately rather than silently treated as money.
    """
    body = str(text or "")
    money: list[float] = []
    with_currency = False
    for match in _AMOUNT.finditer(body):
        value = _to_number(match.group("num"), match.group("mult"))
        if value is None:
            continue
        tagged = bool(match.group("pre") or match.group("post"))
        if tagged:
            with_currency = True
        money.append(value)

    quantities = []
    for match in _QUANTITY.finditer(body):
        value = _to_number(match.group("num"), None)
        if value is not None:
            quantities.append({"value": value, "unit": match.group("unit").lower()})

    negative = bool(_NEGATIVE.search(body))
    affirmative = bool(_AFFIRMATIVE.search(body))
    return {
        "amounts": money,
        "largest_amount": max(money) if money else None,
        "has_currency": with_currency,
        "quantities": quantities,
        "says_no": negative and not money,
        "says_yes": affirmative or bool(money),
    }


# ---------------------------------------------------------------------------
# Requirements stated in the question itself
# ---------------------------------------------------------------------------

_MINIMUM = re.compile(
    r"(?:\(?\s*(?P<n1>\d+)\s*(?:people|persons?|names?|employees?|items?|photos?)?"
    r"\s*(?:minimum|min\.?|or more|at least)\s*\)?)"
    r"|(?:at\s+least\s+(?P<n2>\d+))"
    r"|(?:minimum\s+(?:of\s+)?(?P<n3>\d+))",
    re.I)

_SPLITTERS = re.compile(r"\s*[;,/]\s*|\s*\band\b\s*|\n+|\s*\d+[.)]\s*")


def parse_requirement(question_text: str) -> int | None:
    """The minimum this question demands of an answer, if it states one.

    "(3 people minimum)" -> 3. That is a pass mark the form itself defined, so
    compliance against it is a fact rather than an opinion about what is good
    enough.
    """
    match = _MINIMUM.search(str(question_text or ""))
    if not match:
        return None
    for group in ("n1", "n2", "n3"):
        if match.group(group):
            try:
                value = int(match.group(group))
            except ValueError:
                continue
            return value if 1 <= value <= 50 else None
    return None


def count_items(text: str) -> int:
    """How many distinct things an answer lists.

    Splits on the separators people actually use, then falls back to counting
    capitalised runs (name-like), so "Ann - cashier, Bob - manager" and
    "Ann Smith Bob Jones" both count as two.
    """
    body = str(text or "").strip()
    if not body:
        return 0
    parts = [part.strip() for part in _SPLITTERS.split(body) if part.strip()]
    parts = [part for part in parts if re.search(r"[A-Za-zЀ-ӿ]", part)]
    if len(parts) > 1:
        return len(parts)
    names = re.findall(r"\b[A-ZЀ-Я][a-zа-яїієґ'\-]{1,}\b", body)
    return max(1, len(names) // 2) if len(names) > 2 else 1


# ---------------------------------------------------------------------------
# Templating
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def shape_signature(text: str) -> str:
    """An answer's shape with the varying parts removed.

    `"Text answer #189 to 7"` and `"Text answer #796 to 6"` both become
    `"text answer # to #"`. Identical text is caught by a duplicate check; this
    catches identical *structure*, which is what boilerplate and generated data
    look like.
    """
    body = _SPACE.sub(" ", str(text or "").strip().lower())
    return _DIGITS.sub("#", body)


def templating(answers: Iterable[str], min_answers: int = 10) -> dict[str, Any]:
    """How much of this question's text is one repeated template."""
    values = [str(a) for a in answers if str(a or "").strip()]
    if len(values) < min_answers:
        return {"checked": False, "template_share": None, "distinct_shapes": None}
    shapes = Counter(shape_signature(value) for value in values)
    top_shape, top_count = shapes.most_common(1)[0]
    share = top_count / len(values)
    example = next(v for v in values if shape_signature(v) == top_shape)
    return {
        "checked": True,
        "template_share": share,
        "distinct_shapes": len(shapes),
        "shapes_per_answer": len(shapes) / len(values),
        "dominant_example": example[:160],
        # Distinct text but one shape is the giveaway. 60% is deliberately
        # cautious: a genuinely repetitive but human field ("Yes, checked") should
        # not be called generated.
        "looks_generated": bool(share >= 0.6 and len(shapes) <= max(3, len(values) // 20)),
    }


# ---------------------------------------------------------------------------
# Substance
# ---------------------------------------------------------------------------

_FILLER = re.compile(r"^\s*(n/?a|none|nothing|no|-+|\.+|ok|okay|done|yes)\s*[.!]?\s*$",
                     re.I)


def is_substantive(text: str, min_words: int = 3) -> bool:
    """Does this answer say anything, as opposed to merely being non-empty?"""
    body = str(text or "").strip()
    if not body or _FILLER.match(body):
        return False
    return len(body.split()) >= min_words


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

_STOP = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "have",
    "has", "had", "not", "but", "you", "your", "they", "them", "their", "our",
    "from", "all", "any", "can", "will", "would", "there", "then", "than",
    "into", "out", "about", "just", "some", "more", "very", "too", "also",
    "did", "does", "done", "get", "got", "one", "two", "who", "how", "what",
    "when", "where", "which", "been", "being", "over", "only", "his", "her",
    "she", "him", "its", "yes", "answer", "text", "today", "store",
}
_WORD = re.compile(r"[A-Za-zЀ-ӿ][A-Za-zЀ-ӿ'\-]{2,}")


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(str(text or ""))
            if word.lower() not in _STOP]


def themes(answers: Iterable[str], max_themes: int = 5,
           min_group: int = 2) -> list[dict[str, Any]]:
    """Group answers by their most distinctive word.

    A cheap stand-in for topic modelling that needs no extra dependency and, more
    importantly, is explainable: every group is named by a word the reader can find
    in the example underneath it.
    """
    values = [str(a) for a in answers if str(a or "").strip()]
    if len(values) < max(4, min_group * 2):
        return []

    tokenised = [(value, _tokens(value)) for value in values]
    document_frequency = Counter()
    for _, words in tokenised:
        document_frequency.update(set(words))
    if not document_frequency:
        return []

    total = len(values)
    # Ignore words in nearly every answer (they name the form, not a theme) and
    # words in almost none (they are noise).
    candidates = {word: count for word, count in document_frequency.items()
                  if min_group <= count <= total * 0.8}
    if not candidates:
        return []

    assigned: dict[str, list[str]] = {}
    for value, words in tokenised:
        scored = [(candidates[word], word) for word in set(words) if word in candidates]
        if not scored:
            continue
        # Rarest qualifying word wins: it is the most distinctive thing said.
        _, word = min(scored)
        assigned.setdefault(word, []).append(value)

    groups = sorted(assigned.items(), key=lambda item: -len(item[1]))
    return [{"theme": word, "answers": len(members),
             "share": len(members) / total,
             "example": max(members, key=len)[:200]}
            for word, members in groups[:max_themes] if len(members) >= min_group]


# ---------------------------------------------------------------------------
# Deciding what to measure
# ---------------------------------------------------------------------------

def profile(question_text: str, answers: list[str], *, required: bool | None
            ) -> dict[str, Any]:
    """Choose what this text question should actually be measured on.

    Order: an amount if the answers carry one, compliance if the question states a
    minimum, substance if the field is required (where a response rate is 100% by
    construction and says nothing), and a plain response rate only when the field
    is optional and none of the above applies.
    """
    non_empty = [a for a in answers if str(a or "").strip()]
    minimum = parse_requirement(question_text)

    amount_rows = [extract_amounts(a) for a in non_empty]
    with_amount = [row for row in amount_rows if row["amounts"]]
    with_currency = [row for row in amount_rows if row["has_currency"]]
    quantity_rows = [row for row in amount_rows if row["quantities"]]

    kind = "response_rate"
    if non_empty and (len(with_currency) >= max(3, 0.2 * len(non_empty))
                      or len(quantity_rows) >= max(3, 0.2 * len(non_empty))):
        kind = "amount"
    elif minimum:
        kind = "compliance"
    elif required:
        kind = "substance"

    return {
        "content_kind": kind,
        "required_minimum": minimum,
        "answers_with_amount": len(with_amount),
        "answers_with_currency": len(with_currency),
        "answers_with_quantity": len(quantity_rows),
        "amount_values": [row["largest_amount"] for row in with_amount
                          if row["largest_amount"] is not None],
        "said_no": sum(1 for row in amount_rows if row["says_no"]),
        "said_yes": sum(1 for row in amount_rows if row["says_yes"]),
        "quantity_units": Counter(
            q["unit"] for row in quantity_rows for q in row["quantities"]).most_common(3),
    }
