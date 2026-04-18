"""Analyze module — on-demand deep-dive orchestration.

Reads user target, classifies as symbol/thesis/freeform, loads prior memos
from archive, builds context, calls Gemini Pro, streams memo to terminal,
writes result back to archive.
"""

import re


def _watchlist_symbols() -> list[str]:
    """Return current watchlist symbols. Isolated for easy monkeypatching."""
    import data
    return data.get_all_symbols()


def _thesis_keys() -> list[str]:
    """Return configured thesis bucket keys (lowercased)."""
    import config
    return [k.lower() for k in (config.THESIS_BUCKETS or {}).keys()]


_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")


def classify_target(target: str) -> tuple[str, str]:
    """Classify a raw target string.

    Returns (kind, normalized_target) where kind is "symbol", "thesis",
    or "freeform".

    Rules (in order):
    1. If target is in watchlist (case-insensitive) → symbol
    2. If target is in thesis keys (case-insensitive) → thesis
    3. If target is 1-5 letters with no spaces → symbol (even if unknown)
    4. Otherwise → freeform (normalized unchanged)
    """
    stripped = target.strip()
    upper = stripped.upper()
    lower = stripped.lower()

    if upper in _watchlist_symbols():
        return ("symbol", upper)
    if lower in _thesis_keys():
        return ("thesis", lower)
    if _TICKER_RE.match(stripped):
        return ("symbol", upper)
    return ("freeform", stripped)
