from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class NumericConstraint:
    field: str   # "weight_grams" | "fov" | "kare_hizi"
    op: str      # "<" | "<=" | ">" | ">=" | "="
    value: float


_NUM_RE = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)")

# Pattern: number + optional space + unit
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(gram|gr\b|g\b|kilogram|kg\b|kilo\b)",
    re.IGNORECASE,
)


def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _word(t: str, word: str) -> bool:
    """Match word with word-boundary (avoids 'ağır' matching inside 'ağırlığı')."""
    return bool(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", t))


def _any_phrase(t: str, *phrases: str) -> bool:
    return any(p in t for p in phrases)


def _any_word(t: str, *words: str) -> bool:
    return any(_word(t, w) for w in words)


def parse_weight_grams_from_text(raw: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    Parse a weight string from DB into (min_grams, max_grams).
    Handles: "470 gram", "2.5 kg", "2,5 kg", "< 500g", "≤ 18 kg", "<5 kg"
    """
    t = (raw or "").lower().strip()
    if not t:
        return None

    # Detect unit
    if "kg" in t:
        unit = "kg"
    elif "gram" in t or re.search(r"\bg\b", t):
        unit = "g"
    else:
        # No unit → can't reliably interpret
        return None

    m = _NUM_RE.search(t)
    if not m:
        return None
    num = _to_float(m.group("num"))
    if num is None:
        return None

    grams = num * 1000.0 if unit == "kg" else num

    if "<=" in t or "≤" in t:
        return (None, grams)
    if "<" in t:
        return (None, grams)   # strict < treated same as <= for intervals
    if ">=" in t or "≥" in t:
        return (grams, None)
    if ">" in t:
        return (grams, None)
    # Exact value
    return (grams, grams)


def parse_weight_constraint_from_question(q: str) -> Optional[NumericConstraint]:
    """
    Parse the user's numeric weight constraint from a natural language question.
    Returns NumericConstraint or None if no weight filter detected.
    """
    t = (q or "").lower()

    # Must mention weight concept
    if not any([
        "ağırl" in t, "agirl" in t, "kilo" in t, "kg" in t, "gram" in t,
    ]):
        return None

    # Extract number + unit
    unit = None
    if "kg" in t or "kilo" in t:
        unit = "kg"
    elif "gram" in t or re.search(r"\bg\b", t):
        unit = "g"
    if unit is None:
        return None

    m = _NUM_RE.search(t)
    if not m:
        return None
    num = _to_float(m.group("num"))
    if num is None:
        return None
    grams = num * 1000.0 if unit == "kg" else num

    # Explicit operators (highest priority)
    if "<=" in t or "≤" in t:
        op: Optional[str] = "<="
    elif "<" in t:
        op = "<="   # treat strict-less as <=
    elif ">=" in t or "≥" in t:
        op = ">="
    elif ">" in t:
        op = ">="
    else:
        op = None

    # Turkish phrasing (only when no explicit operator)
    if op is None:
        less_phrases = [
            "daha az", "daha düşük", "daha dusuk",
            "veya daha az", "ya da daha az",
            "eşit ve az", "esit ve az", "eşit veya az", "esit veya az",
            "en fazla", "maks", "maksimum",
        ]
        less_words = [
            "az", "alt", "altında", "altinda",
            "düşük", "dusuk", "hafif",
        ]
        more_phrases = [
            "daha fazla", "daha yüksek", "daha yuksek",
            "en az", "minimum",
        ]
        more_words = [
            "fazla", "üst", "üstünde", "ustunde",
            "üzeri", "uzeri", "üstü", "ustu",
        ]

        if _any_phrase(t, *less_phrases) or _any_word(t, *less_words):
            op = "<="
        elif _any_phrase(t, *more_phrases) or _any_word(t, *more_words):
            op = ">="
        else:
            op = "="

    return NumericConstraint(field="weight_grams", op=op, value=grams)


def parse_weight_range_from_question(
    q: str,
) -> Optional[Tuple[NumericConstraint, NumericConstraint]]:
    """
    Detect two-sided weight range like "500 gram ile 10 kilo arasında".
    Returns (min_constraint, max_constraint) or None if no range found.
    """
    t = (q or "").lower()

    # Must mention weight AND range concept
    has_weight = any(x in t for x in ["ağırl", "agirl", "kilo", "kg", "gram"])
    has_range = any(x in t for x in ["arasında", "arasinda", "ile", "between", "-"])
    if not (has_weight and has_range):
        return None

    # Find all (number, unit) pairs in the text
    matches = _NUM_UNIT_RE.findall(t)
    if len(matches) < 2:
        return None

    def _to_grams(num_str: str, unit_str: str) -> float:
        val = _to_float(num_str) or 0.0
        u = unit_str.lower().strip()
        if "k" in u:  # kg, kilo, kilogram
            return val * 1000.0
        return val  # gram, gr, g

    g_values = [_to_grams(n, u) for n, u in matches]
    lo = min(g_values)
    hi = max(g_values)

    return (
        NumericConstraint(field="weight_grams", op=">=", value=lo),
        NumericConstraint(field="weight_grams", op="<=", value=hi),
    )


def interval_satisfies_constraint(
    *, interval: Tuple[Optional[float], Optional[float]], constraint: NumericConstraint
) -> bool:
    """
    Check if a product's weight interval satisfies the user's constraint.

    Products often have weights stored as "< 5 kg" → (None, 5000) or
    "≤ 18 kg" → (None, 18000). For >= / > queries we use the upper bound
    as a proxy: if a product can weigh UP TO 18 kg, it can certainly
    satisfy a ">= 5 kg" requirement.
    """
    min_v, max_v = interval
    v = constraint.value
    op = constraint.op

    if op in ("=", "=="):
        if min_v is not None and max_v is not None:
            return min_v <= v <= max_v
        # Open interval – include when the bound touches the target
        if min_v is None and max_v is not None:
            return max_v >= v
        if max_v is None and min_v is not None:
            return min_v <= v
        return False
    if op == "<":
        # Product's known upper bound must be strictly less than v
        return max_v is not None and max_v < v
    if op == "<=":
        return max_v is not None and max_v <= v
    if op == ">":
        # If min is known use it; else use max as a conservative proxy
        if min_v is not None:
            return min_v > v
        return max_v is not None and max_v > v
    if op == ">=":
        if min_v is not None:
            return min_v >= v
        return max_v is not None and max_v >= v
    return False


def parse_generic_constraint_from_question(q: str) -> Optional[NumericConstraint]:
    """Generic numeric constraint for FOV or frame-rate fields."""
    t = (q or "").lower()

    if "fov" in t:
        field = "fov"
    elif "kare h" in t or "fps" in t or "hz" in t:
        field = "kare_hizi"
    else:
        return None

    if "<=" in t or "≤" in t:
        op: Optional[str] = "<="
    elif "<" in t:
        op = "<="
    elif ">=" in t or "≥" in t:
        op = ">="
    elif ">" in t:
        op = ">="
    elif _any_phrase(t, "daha az", "daha düşük", "en fazla") or _any_word(t, "az", "alt", "düşük"):
        op = "<="
    elif _any_phrase(t, "daha fazla", "daha yüksek", "en az") or _any_word(t, "fazla", "üst", "üzeri"):
        op = ">="
    else:
        op = "="

    m = _NUM_RE.search(t)
    if not m:
        return None
    num = _to_float(m.group("num"))
    if num is None:
        return None

    return NumericConstraint(field=field, op=op, value=float(num))
