"""The statistics themselves, as small pure functions over plain sequences.

Deliberately dependency-light (numpy only, no scipy) so the analyser can run in
a slim worker image, and deliberately separate from the pandas plumbing so each
formula is unit-testable in isolation.

Choices worth defending
-----------------------
* **Mann-Kendall, not just a regression line.** Bucket series are short (4-24
  points), often skewed and occasionally spiky. A least-squares slope on 6 noisy
  points reports a confident trend that is not there. Mann-Kendall is
  rank-based, needs no distributional assumption, and is not dragged around by a
  single outlier bucket. The OLS slope is still reported, because "+1.8 units per
  bucket" is what a human wants to read - but the *claim* that a trend exists
  comes from the non-parametric test.
* **Wilson intervals for rates, never bare percentages.** "67% said yes" from 3
  responses and from 300 are different facts. Wilson behaves at small n and at
  rates near 0 and 1, where the textbook normal interval produces impossible
  bounds.
* **Two outlier rules.** The IQR fence is what people expect; the modified
  z-score (median absolute deviation) is what survives a contaminated sample.
  Both are reported so a disagreement between them is visible.
* **Exact p-values.** Student's t and the normal tail are computed here rather
  than approximated, so a significance claim in the report is a real one.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "describe", "wilson_interval", "mann_kendall", "ols_trend", "entropy",
    "normalised_entropy", "gini", "outliers_iqr", "outliers_mad",
    "pct_change", "volatility", "largest_shift", "moving_average", "safe_div",
    "classify_trend", "dispersion", "value_bands", "distribution_shape",
    "expected_rate_band", "expected_median_band", "expected_count_band",
    "two_proportion_p", "shrink_rates", "shrink_values", "detectable_difference",
    "answers_needed_for",
    "chi_square_p", "rates_homogeneous", "values_homogeneous", "longest_run",
    "band_shares",
]


def safe_div(numerator: float, denominator: float, default: float | None = None
             ) -> float | None:
    return default if not denominator else numerator / denominator


def _clean(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([v for v in values if v is not None], dtype="float64")
    return array[np.isfinite(array)]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def describe(values: Iterable[Any]) -> dict[str, Any]:
    """Location, spread and shape for a numeric sample."""
    array = _clean(values)
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None,
                "max": None, "sum": None, "p10": None, "p25": None, "p75": None,
                "p90": None, "iqr": None, "cv": None, "skew": None, "range": None}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    p10, p25, p50, p75, p90 = (float(x) for x in
                               np.percentile(array, [10, 25, 50, 75, 90]))
    skew = None
    if n > 2 and std > 0:
        skew = float(((array - mean) ** 3).mean() / std ** 3)
    return {
        "n": n, "mean": mean, "median": p50, "std": std,
        "min": float(array.min()), "max": float(array.max()),
        "sum": float(array.sum()), "range": float(array.max() - array.min()),
        "p10": p10, "p25": p25, "p75": p75, "p90": p90, "iqr": p75 - p25,
        "cv": (std / abs(mean)) if mean else None,
        "skew": skew,
    }


def outliers_iqr(values: Iterable[Any], k: float = 1.5) -> dict[str, Any]:
    array = _clean(values)
    if array.size < 4:
        return {"method": "iqr", "count": 0, "lower": None, "upper": None, "values": []}
    p25, p75 = np.percentile(array, [25, 75])
    iqr = p75 - p25
    lower, upper = p25 - k * iqr, p75 + k * iqr
    flagged = array[(array < lower) | (array > upper)]
    return {"method": "iqr", "k": k, "count": int(flagged.size),
            "lower": float(lower), "upper": float(upper),
            "values": sorted({float(v) for v in flagged})[:20]}


def outliers_mad(values: Iterable[Any], threshold: float = 3.5) -> dict[str, Any]:
    """Modified z-score (Iglewicz-Hoaglin). Robust to a contaminated sample."""
    array = _clean(values)
    if array.size < 4:
        return {"method": "mad", "count": 0, "values": []}
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad == 0:
        return {"method": "mad", "count": 0, "values": [], "note": "MAD is zero"}
    scores = 0.6745 * (array - median) / mad
    flagged = array[np.abs(scores) > threshold]
    return {"method": "mad", "threshold": threshold, "median": median, "mad": mad,
            "count": int(flagged.size),
            "values": sorted({float(v) for v in flagged})[:20]}


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.959963985
                    ) -> dict[str, Any]:
    """Wilson score interval for a proportion. Correct at small n and at the
    boundaries, where the normal approximation is not."""
    if total <= 0:
        return {"rate": None, "low": None, "high": None, "n": 0, "successes": 0,
                "margin": None}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / denominator)
    low, high = max(0.0, centre - spread), min(1.0, centre + spread)
    return {"rate": p, "low": low, "high": high, "n": total,
            "successes": int(successes), "margin": (high - low) / 2}


# ---------------------------------------------------------------------------
# Tail probabilities (no scipy)
# ---------------------------------------------------------------------------

def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps, max_iter = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log(1.0 - x) + a * math.log(x)) \
        * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, degrees_of_freedom: int) -> float:
    """Exact two-sided p-value for a t statistic."""
    if degrees_of_freedom <= 0:
        return 1.0
    df = float(degrees_of_freedom)
    return float(_betai(df / 2.0, 0.5, df / (df + t * t)))


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def mann_kendall(values: Sequence[Any]) -> dict[str, Any]:
    """Non-parametric monotonic-trend test with tie correction.

    Returns Kendall's tau, the S statistic, a normal-approximation z and its
    two-sided p-value. n < 4 returns p = None: with three points there is nothing
    to test and pretending otherwise is how reports start lying.
    """
    array = _clean(values)
    n = int(array.size)
    if n < 4:
        return {"n": n, "tau": None, "s": None, "z": None, "p_value": None,
                "significant": False, "note": "need at least 4 buckets"}

    s = 0
    for i in range(n - 1):
        s += int(np.sign(array[i + 1:] - array[i]).sum())

    tie_correction = sum(t * (t - 1) * (2 * t + 5)
                         for t in Counter(array.tolist()).values() if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0
    if variance <= 0:
        return {"n": n, "tau": 0.0, "s": s, "z": 0.0, "p_value": 1.0,
                "significant": False, "note": "no variation between buckets"}

    z = (s - np.sign(s)) / math.sqrt(variance) if s != 0 else 0.0
    p = 2.0 * _norm_sf(abs(z))
    denominator = 0.5 * n * (n - 1)
    return {"n": n, "tau": float(s / denominator) if denominator else None,
            "s": int(s), "z": float(z), "p_value": float(min(p, 1.0)),
            "significant": bool(p < 0.05)}


def ols_trend(values: Sequence[Any]) -> dict[str, Any]:
    """Least-squares slope per bucket, with R^2 and an exact p-value.

    Reported for magnitude ("about +1.8 per month"), not for the existence of a
    trend - that is Mann-Kendall's job.
    """
    array = _clean(values)
    n = int(array.size)
    if n < 3:
        return {"n": n, "slope": None, "intercept": None, "r_squared": None,
                "p_value": None, "significant": False}
    x = np.arange(n, dtype="float64")
    x_centred = x - x.mean()
    y_centred = array - array.mean()
    sxx = float((x_centred ** 2).sum())
    if sxx == 0:
        return {"n": n, "slope": 0.0, "intercept": float(array.mean()),
                "r_squared": None, "p_value": None, "significant": False}
    slope = float((x_centred * y_centred).sum() / sxx)
    intercept = float(array.mean() - slope * x.mean())
    fitted = intercept + slope * x
    residual_ss = float(((array - fitted) ** 2).sum())
    total_ss = float((y_centred ** 2).sum())
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else None

    p_value = None
    if n > 2 and residual_ss > 0:
        standard_error = math.sqrt(residual_ss / (n - 2) / sxx)
        if standard_error > 0:
            p_value = t_two_sided_p(slope / standard_error, n - 2)
    elif residual_ss == 0 and total_ss > 0:
        p_value = 0.0
    return {"n": n, "slope": slope, "intercept": intercept,
            "r_squared": r_squared, "p_value": p_value,
            "significant": bool(p_value is not None and p_value < 0.05)}


def classify_trend(mk: dict, ols: dict, first: float | None, last: float | None
                   ) -> dict[str, Any]:
    """Turn the two tests into one plain-language verdict.

    `direction` is only "rising"/"falling" when the non-parametric test agrees;
    otherwise the movement is called flat or noisy, whatever the slope's sign.
    """
    tau, p = mk.get("tau"), mk.get("p_value")
    slope = ols.get("slope")
    if tau is None or p is None:
        direction, confidence = "insufficient_data", "none"
    elif p < 0.05 and abs(tau) >= 0.3:
        direction = "rising" if tau > 0 else "falling"
        confidence = "high" if p < 0.01 else "moderate"
    elif p < 0.2 and abs(tau) >= 0.2:
        direction = "possibly_rising" if tau > 0 else "possibly_falling"
        confidence = "low"
    else:
        direction, confidence = "flat", "low" if abs(tau or 0) < 0.15 else "none"

    return {"direction": direction, "confidence": confidence,
            "tau": tau, "p_value": p,
            "slope_per_bucket": slope,
            "first_bucket_value": first, "last_bucket_value": last,
            "change_absolute": (None if first is None or last is None
                                else last - first),
            "change_pct": pct_change(first, last),
            "r_squared": ols.get("r_squared")}


def pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first == 0:
        return None
    return (last - first) / abs(first) * 100.0


def volatility(values: Sequence[Any]) -> dict[str, Any]:
    """Coefficient of variation across buckets - how erratic the metric is,
    independent of its level."""
    array = _clean(values)
    if array.size < 2:
        return {"std": None, "cv": None, "label": "unknown"}
    mean, std = float(array.mean()), float(array.std(ddof=1))
    cv = (std / abs(mean)) if mean else None
    label = "unknown"
    if cv is not None:
        label = "stable" if cv < 0.15 else "moderate" if cv < 0.4 else "erratic"
    return {"std": std, "cv": cv, "label": label}


def largest_shift(values: Sequence[Any], labels: Sequence[str] | None = None
                  ) -> dict[str, Any] | None:
    """The biggest bucket-to-bucket jump: the cheapest useful change-point signal
    on a series this short."""
    array = _clean(values)
    if array.size < 2:
        return None
    deltas = np.diff(array)
    if not np.any(deltas):            # a constant series has no "biggest move"
        return None
    index = int(np.argmax(np.abs(deltas)))
    out: dict[str, Any] = {"from_index": index, "to_index": index + 1,
                           "delta": float(deltas[index]),
                           "pct": pct_change(float(array[index]),
                                             float(array[index + 1]))}
    if labels is not None and len(labels) > index + 1:
        out["from_label"], out["to_label"] = labels[index], labels[index + 1]
    return out


def moving_average(values: Sequence[Any], window: int = 3) -> list[float | None]:
    array = _clean(values)
    if array.size < window or window < 2:
        return [None] * len(list(values))
    kernel = np.ones(window) / window
    smoothed = np.convolve(array, kernel, mode="valid")
    pad = [None] * (window - 1)
    return pad + [float(v) for v in smoothed]


# ---------------------------------------------------------------------------
# Is one number a fair summary at all?
# ---------------------------------------------------------------------------
# The averaging problem: answers of 10 and 100 do not mean 55. They mean the
# question is being answered about materially different things, and any single
# central value hides that. So before anything reports a summary value, it asks
# how dispersed the answers are and whether they form one group or several.
#
# The measure is the quartile coefficient of dispersion:
#
#     qcd = (p75 - p25) / (p75 + p25)
#
# scale-free (comparable between a question answered in units and one answered in
# thousands), robust (built from quartiles, so one absurd answer cannot move it),
# and defined for any positive data. The mean and standard deviation are still
# computed, but they are never the headline.

def dispersion(values: Iterable[Any], *, tight: float = 0.15, wide: float = 0.40,
               spread_ratio_wide: float = 10.0) -> dict[str, Any]:
    """How spread out the answers are, and whether a single value may be reported.

    `single_value_representative` False means: report the range and the bands,
    not an average.
    """
    array = np.sort(_clean(values))
    n = int(array.size)
    if n == 0:
        return {"n": 0, "verdict": "no_answers", "qcd": None, "spread_ratio": None,
                "single_value_representative": False, "typical_low": None,
                "typical_high": None, "shape": "no_answers"}
    if n < 4:
        return {"n": n, "verdict": "too_few_answers", "qcd": None,
                "spread_ratio": None,
                "single_value_representative": bool(n and len(set(array.tolist())) == 1),
                "typical_low": float(array.min()), "typical_high": float(array.max()),
                "shape": "too_few_answers"}

    p25, p50, p75, p90 = (float(x) for x in np.percentile(array, [25, 50, 75, 90]))
    total = p75 + p25
    qcd = abs(p75 - p25) / abs(total) if total else (0.0 if p75 == p25 else None)
    spread_ratio = (p75 / p25) if p25 > 0 else None

    shape = distribution_shape(array)
    if qcd is None:
        verdict = "unmeasurable"
    elif qcd < tight:
        verdict = "tight"
    elif qcd < wide:
        verdict = "moderate"
    else:
        verdict = "wide"
    if spread_ratio is not None and spread_ratio >= spread_ratio_wide:
        verdict = "wide"

    return {
        "n": n,
        "verdict": verdict,
        "qcd": qcd,
        "spread_ratio": spread_ratio,
        "typical_low": p25,
        "typical_high": p75,
        "middle": p50,
        "tail_ratio": (p90 / p50) if p50 else None,
        "shape": shape,
        # One value may be quoted only when the answers really do cluster.
        "single_value_representative": bool(
            verdict in ("tight", "moderate") and shape == "one_group"),
    }


def distribution_shape(values: Iterable[Any], *, gap_share: float = 0.35,
                       min_side: float = 0.25) -> str:
    """"one_group", "two_groups" or "long_tail".

    Two groups are detected by the largest gap between consecutive sorted answers:
    if it spans a third of the range and both sides hold a quarter of the answers,
    there are two populations here and no single summary value is honest.
    """
    array = np.sort(_clean(values))
    n = int(array.size)
    if n < 4:
        return "too_few_answers"
    span = float(array[-1] - array[0])
    if span == 0:
        return "one_group"

    gaps = np.diff(array)
    index = int(np.argmax(gaps))
    biggest = float(gaps[index])
    left, right = index + 1, n - index - 1
    if (biggest / span >= gap_share
            and min(left, right) / n >= min_side):
        return "two_groups"

    p50, p90 = (float(x) for x in np.percentile(array, [50, 90]))
    if p50 > 0 and p90 / p50 >= 3.0:
        return "long_tail"
    return "one_group"


def value_bands(values: Iterable[Any], bands: int = 4) -> list[dict[str, Any]]:
    """Quantile bands over the observed range: the cheapest honest replacement for
    an average when the answers do not cluster. Four bands and their counts show
    a 10-and-100 split immediately."""
    array = _clean(values)
    n = int(array.size)
    if n < bands or bands < 2:
        return []
    edges = np.percentile(array, np.linspace(0, 100, bands + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return []
    counts, _ = np.histogram(array, bins=edges)
    return [{"from": float(edges[i]), "to": float(edges[i + 1]),
             "count": int(counts[i]), "share": float(counts[i] / n)}
            for i in range(len(counts))]


# ---------------------------------------------------------------------------
# Is this period actually different, or is it just a small sample?
# ---------------------------------------------------------------------------
# The failure this prevents: 172 forms over 14 weeks is ~12 answers a week. At a
# true yes-rate of 52%, chance alone throws up weekly rates anywhere between 26%
# and 78%. Reporting "highest in mid-February, lowest in early March" out of that
# is reading a pattern into a run of coin flips - and a model handed those numbers
# will build a story on them every time.
#
# So before any period is called high or low, it is compared against the range
# that its own sample size would produce anyway. Only periods outside that range
# are notable. Everything else is explicitly reported as ordinary variation.

def expected_rate_band(overall_rate: float | None, n: int, z: float = 1.959963985
                       ) -> dict[str, Any]:
    """The range a period's rate would fall in by chance, given its `n`.

    Normal approximation to the binomial sampling distribution around the
    period-independent overall rate. `n` is the answers in that period.
    """
    if overall_rate is None or n <= 0:
        return {"low": None, "high": None, "n": n, "width": None}
    p = min(max(float(overall_rate), 0.0), 1.0)
    spread = z * math.sqrt(max(p * (1 - p), 1e-12) / n)
    low, high = max(0.0, p - spread), min(1.0, p + spread)
    return {"low": low, "high": high, "n": int(n), "width": high - low}


def expected_median_band(overall_median: float | None, iqr: float | None, n: int,
                         z: float = 1.959963985) -> dict[str, Any]:
    """The range a period's middle value would fall in by chance.

    Standard error of the median is about `1.2533 * s / sqrt(n)`; `s` is estimated
    robustly from the interquartile range (`IQR / 1.349`) so one wild answer does
    not inflate the band and hide a real difference.
    """
    if overall_median is None or n <= 0 or iqr is None or iqr <= 0:
        return {"low": None, "high": None, "n": n, "width": None}
    sigma = iqr / 1.349
    spread = z * 1.2533 * sigma / math.sqrt(n)
    return {"low": overall_median - spread, "high": overall_median + spread,
            "n": int(n), "width": 2 * spread}


def expected_count_band(mean_per_unit: float | None, units: float,
                        z: float = 1.959963985) -> dict[str, Any]:
    """The range a period's count-per-unit would fall in by chance (Poisson)."""
    if mean_per_unit is None or units <= 0 or mean_per_unit < 0:
        return {"low": None, "high": None, "n": None, "width": None}
    spread = z * math.sqrt(mean_per_unit / units)
    low = max(0.0, mean_per_unit - spread)
    return {"low": low, "high": mean_per_unit + spread, "n": None,
            "width": mean_per_unit + spread - low}


def is_notable(value: float | None, band: dict[str, Any] | None) -> bool:
    """Does this value sit outside what its sample size would produce anyway?"""
    if value is None or not band or band.get("low") is None:
        return False
    return bool(value < band["low"] or value > band["high"])


# ---------------------------------------------------------------------------
# How much data would be needed to see a difference at all
# ---------------------------------------------------------------------------

def answers_needed_for(difference: float, baseline_rate: float = 0.5,
                       power: float = 0.80, alpha: float = 0.05) -> int:
    """Answers per period needed to detect a rate difference of `difference`.

    Used by the interval engine: if the requested split leaves 12 answers per
    period and 174 are needed to see a 15-point move, the split is too fine and
    the engine says so rather than producing 14 uninterpretable numbers.
    """
    if difference <= 0:
        return 0
    z_alpha, z_beta = 1.959963985, 0.8416212336        # two-sided 5%, 80% power
    p = min(max(baseline_rate, 0.01), 0.99)
    return int(math.ceil(2 * ((z_alpha + z_beta) ** 2) * p * (1 - p)
                         / difference ** 2))


def detectable_difference(n: int, baseline_rate: float = 0.5, power: float = 0.80
                          ) -> float | None:
    """The smallest rate difference `n` answers per period could reveal."""
    if n <= 0:
        return None
    z_alpha, z_beta = 1.959963985, 0.8416212336
    p = min(max(baseline_rate, 0.01), 0.99)
    return math.sqrt(2 * ((z_alpha + z_beta) ** 2) * p * (1 - p) / n)


# ---------------------------------------------------------------------------
# Comparing one person against the rest
# ---------------------------------------------------------------------------

def two_proportion_p(successes_a: int, total_a: int, successes_b: int,
                     total_b: int) -> float | None:
    """Two-sided p-value for two proportions differing (pooled normal test).

    The gate on naming a person: one yes out of six answers is not evidence of
    anything, and this is what says so.
    """
    if total_a <= 0 or total_b <= 0:
        return None
    pooled = (successes_a + successes_b) / (total_a + total_b)
    if pooled in (0.0, 1.0):
        return 1.0
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if standard_error == 0:
        return 1.0
    z = (successes_a / total_a - successes_b / total_b) / standard_error
    return float(min(1.0, 2.0 * _norm_sf(abs(z))))


def shrink_rates(observations: Sequence[tuple[Any, int, int]]
                 ) -> list[dict[str, Any]]:
    """Empirical-Bayes adjustment of per-person rates towards the group.

    `observations` is (key, successes, total). A person with 1 of 6 answers is
    mostly noise; a person with 99 is mostly signal. A beta prior fitted from the
    group by moment matching weights each person accordingly, so the ranking stops
    being "whoever had the smallest sample".

    Returns raw and adjusted rates plus the prior strength, so the report can show
    both and say which one the ranking used.
    """
    rows = [(key, int(s), int(t)) for key, s, t in observations if t and t > 0]
    if not rows:
        return []

    total_successes = sum(s for _, s, _ in rows)
    total_answers = sum(t for _, _, t in rows)
    group_rate = total_successes / total_answers

    rates = [s / t for _, s, t in rows]
    # Between-person variance, with the within-person sampling variance removed;
    # what is left is the real spread the prior should reflect.
    if len(rates) > 1:
        observed_var = sum((r - group_rate) ** 2 for r in rates) / (len(rates) - 1)
        within = sum(group_rate * (1 - group_rate) / t for _, _, t in rows) / len(rows)
        between = max(observed_var - within, 1e-6)
    else:
        between = 1e-6
    strength = max(1.0, group_rate * (1 - group_rate) / between - 1)
    alpha, beta = group_rate * strength, (1 - group_rate) * strength

    out = []
    for key, successes, total in rows:
        adjusted = (successes + alpha) / (total + alpha + beta)
        low, high = wilson_interval(successes, total)["low"], \
            wilson_interval(successes, total)["high"]
        out.append({
            "key": key, "successes": successes, "answers": total,
            "raw_rate": successes / total,
            "adjusted_rate": adjusted,
            "low": low, "high": high,
            "shrunk_by": adjusted - successes / total,
            "p_vs_rest": two_proportion_p(successes, total,
                                          total_successes - successes,
                                          total_answers - total),
        })
    for row in out:
        row["group_rate"] = group_rate
        row["prior_strength"] = strength
    return out


def shrink_values(observations: Sequence[tuple[Any, float | None, int]],
                  prior_answers: float = 10.0) -> list[dict[str, Any]]:
    """The same idea for a measured value: pull each person's figure towards the
    group figure in proportion to how little data it rests on.

    `prior_answers` is how many answers the group figure is worth - the weight a
    person needs before their own number is trusted outright.
    """
    rows = [(key, float(value), int(n)) for key, value, n in observations
            if value is not None and n and n > 0]
    if not rows:
        return []
    weighted = sum(value * n for _, value, n in rows) / sum(n for _, _, n in rows)
    out = []
    for key, value, n in rows:
        weight = n / (n + prior_answers)
        out.append({"key": key, "answers": n, "raw_value": value,
                    "adjusted_value": weight * value + (1 - weight) * weighted,
                    "group_value": weighted, "weight": weight})
    return out


# ---------------------------------------------------------------------------
# Categorical spread
# ---------------------------------------------------------------------------

def entropy(counts: Iterable[float], base: float = 2.0) -> float | None:
    values = [float(c) for c in counts if c and c > 0]
    total = sum(values)
    if total <= 0 or len(values) < 2:
        return 0.0 if total > 0 else None
    return -sum((c / total) * math.log(c / total, base) for c in values)


def normalised_entropy(counts: Iterable[float], base: float = 2.0) -> float | None:
    """Entropy scaled to 0-1: 0 = everyone picked the same option, 1 = perfectly
    even spread. Comparable between questions with different option counts."""
    values = [float(c) for c in counts if c and c > 0]
    if len(values) < 2:
        return 0.0 if values else None
    raw = entropy(values, base)
    ceiling = math.log(len(values), base)
    return None if not ceiling else (raw or 0.0) / ceiling


def gini(counts: Iterable[float]) -> float | None:
    """Concentration of a category distribution (0 = even, ->1 = dominated)."""
    values = sorted(float(c) for c in counts if c and c > 0)
    n = len(values)
    total = sum(values)
    if n < 2 or total <= 0:
        return 0.0 if total > 0 else None
    cumulative = sum((i + 1) * value for i, value in enumerate(values))
    return (2 * cumulative) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------------------
# Is the figure steady, or does it genuinely move?
# ---------------------------------------------------------------------------
# A manager needs these apart. A check that sits at 52% every single fortnight is
# a stable process running at 52%: to change it you change the process. A check
# that swings 39%-59% between fortnights is an unstable one: something differs
# between those fortnights and finding out what it is comes first. Both currently
# read as "no real change here", which is true about the *trend* and useless as
# guidance.
#
# Testing each period against its own band cannot separate them, because it asks
# a question about one period at a time. Pooling every period into a single
# homogeneity test is far more powerful: seven fortnights each too thin to flag
# alone can jointly show that the figure is not sitting still.

def _gammln(x: float) -> float:
    """Log of the gamma function (Lanczos)."""
    coefficients = (76.18009172947146, -86.50532032941677, 24.01409824083091,
                    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    series = 1.000000000190015
    for coefficient in coefficients:
        y += 1.0
        series += coefficient / y
    return -tmp + math.log(2.5066282746310005 * series / x)


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x).

    Series expansion below the crossover, continued fraction above it, which is
    where each converges quickly.
    """
    if x < 0.0 or a <= 0.0:
        return float("nan")
    if x == 0.0:
        return 1.0
    log_gamma = _gammln(a)
    if x < a + 1.0:                                  # series for P, then complement
        total = term = 1.0 / a
        for n in range(1, 300):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - log_gamma)
    # Continued fraction for Q (modified Lentz).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - log_gamma)


def chi_square_p(chi_square: float, degrees_of_freedom: int) -> float | None:
    """Upper-tail probability of a chi-square statistic."""
    if degrees_of_freedom <= 0 or chi_square is None or chi_square < 0:
        return None
    value = _gamma_q(degrees_of_freedom / 2.0, chi_square / 2.0)
    if value != value:
        return None
    return float(min(max(value, 0.0), 1.0))


def rates_homogeneous(successes: Sequence[Any], totals: Sequence[Any],
                      alpha: float = 0.05) -> dict[str, Any]:
    """Are these period rates all the same rate, or do they really differ?

    Chi-square test of homogeneity on the yes/no counts. `steady` True means the
    spread between periods is what one rate sampled repeatedly would produce.
    """
    pairs = [(int(s), int(n)) for s, n in zip(successes, totals)
             if n is not None and s is not None and int(n) > 0]
    out: dict[str, Any] = {"periods": len(pairs), "chi_square": None, "df": None,
                           "p": None, "steady": None, "pooled_rate": None}
    if len(pairs) < 2:
        return out
    total_n = sum(n for _, n in pairs)
    total_s = sum(s for s, _ in pairs)
    pooled = total_s / total_n
    out["pooled_rate"] = float(pooled)
    if pooled <= 0.0 or pooled >= 1.0:
        # Every answer went the same way: nothing varies, so nothing to test.
        out.update({"chi_square": 0.0, "df": len(pairs) - 1, "p": 1.0, "steady": True})
        return out
    chi = 0.0
    for successes_i, n_i in pairs:
        for observed, expected in ((successes_i, n_i * pooled),
                                   (n_i - successes_i, n_i * (1 - pooled))):
            if expected > 0:
                chi += (observed - expected) ** 2 / expected
    df = len(pairs) - 1
    p = chi_square_p(chi, df)
    out.update({"chi_square": float(chi), "df": df, "p": p,
                "steady": None if p is None else bool(p >= alpha)})
    return out


def values_homogeneous(groups: Sequence[Sequence[Any]], alpha: float = 0.05
                       ) -> dict[str, Any]:
    """Kruskal-Wallis: do these periods share one distribution of values?

    Rank-based, so a bimodal or heavily skewed question - where a mean would be
    meaningless - is still testable. The rate-based twin of this is
    `rates_homogeneous`.
    """
    cleaned = [_clean(group) for group in groups]
    cleaned = [group for group in cleaned if group.size > 0]
    out: dict[str, Any] = {"periods": len(cleaned), "chi_square": None, "df": None,
                           "p": None, "steady": None}
    if len(cleaned) < 2:
        return out
    combined = np.concatenate(cleaned)
    total = combined.size
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(total, dtype=float)
    ranks[order] = np.arange(1, total + 1, dtype=float)
    # Average ranks within ties, or the statistic is inflated. A form where every
    # answer is the same number is entirely ties, which must give chi-square 0.
    sorted_values = combined[order]
    start = 0
    for index in range(1, total + 1):
        if index == total or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    offset = 0
    statistic = 0.0
    for group in cleaned:
        size = group.size
        group_ranks = ranks[offset:offset + size]
        statistic += (group_ranks.sum() ** 2) / size
        offset += size
    chi = 12.0 / (total * (total + 1)) * statistic - 3.0 * (total + 1)

    # Tie correction.
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    if total > 1 and tie_term and (total ** 3 - total) != tie_term:
        chi /= 1.0 - tie_term / (total ** 3 - total)
    chi = max(float(chi), 0.0)

    df = len(cleaned) - 1
    p = chi_square_p(chi, df)
    out.update({"chi_square": chi, "df": df, "p": p,
                "steady": None if p is None else bool(p >= alpha)})
    return out


def longest_run(flags: Sequence[Any]) -> dict[str, Any]:
    """The longest unbroken stretch of True, and where it starts.

    "Three fortnights in a row below target" is a sentence a manager acts on;
    "the mean is below target" is not.
    """
    best = current = 0
    best_end = current_start = 0
    best_start = 0
    for index, flag in enumerate(flags):
        if flag:
            if current == 0:
                current_start = index
            current += 1
            if current > best:
                best, best_start, best_end = current, current_start, index
        else:
            current = 0
    return {"length": int(best), "start_index": int(best_start) if best else None,
            "end_index": int(best_end) if best else None,
            "of": len(list(flags))}


def band_shares(values: Iterable[Any], bands: Sequence[dict[str, Any]]
                ) -> list[dict[str, Any]]:
    """How this slice of answers falls across bands defined on the whole period.

    The point of fixing the edges once, on the full window, is that the per-period
    shares are then comparable: a rising share in the top band means something,
    where a per-period quartile would silently move the goalposts each time.
    """
    array = _clean(values)
    total = int(array.size)
    out: list[dict[str, Any]] = []
    for index, band in enumerate(bands):
        low, high = band.get("from"), band.get("to")
        if low is None or high is None:
            continue
        last = index == len(bands) - 1
        inside = ((array >= low) & (array <= high)) if last else \
                 ((array >= low) & (array < high))
        count = int(np.count_nonzero(inside))
        out.append({"from": float(low), "to": float(high), "count": count,
                    "share": (count / total) if total else None})
    return out
