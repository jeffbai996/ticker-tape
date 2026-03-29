"""Trade journal — persistent timestamped log of decisions, observations, and rationale.

Distinct from memories (AI context) and history (chat transcript).
Journal entries are the user's own thinking, importable into chat on demand.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

log = logging.getLogger(__name__)

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "journal.json")
MAX_ENTRIES = 500


def load_entries() -> list[dict]:
    """Load journal entries from disk.

    Returns list of {"id": int, "text": str, "ts": str, "symbols": list[str]}.
    """
    if not os.path.exists(JOURNAL_FILE):
        return []
    try:
        with open(JOURNAL_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load journal: %s", e)
        return []


def _save_entries(entries: list[dict]) -> None:
    """Write journal entries to disk."""
    os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except OSError as e:
        log.warning("Failed to save journal: %s", e)


def _next_id(entries: list[dict]) -> int:
    if not entries:
        return 1
    return max(e.get("id", 0) for e in entries) + 1


_SYM_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _extract_symbols(text: str) -> list[str]:
    """Extract likely ticker symbols from text (1-5 uppercase letters)."""
    # Filter common English words that look like tickers
    stopwords = {"I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
                 "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR",
                 "SO", "TO", "UP", "US", "WE", "THE", "AND", "BUT", "FOR",
                 "NOT", "ALL", "CAN", "HAD", "HAS", "HER", "HIM", "HIS",
                 "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "OUR", "OUT",
                 "OWN", "SAY", "SHE", "TOO", "USE", "WAY", "WHO", "BOY",
                 "DID", "GET", "HIT", "LET", "PUT", "RUN", "SET", "TOP",
                 "BUY", "SELL", "SOLD", "TRIM", "HOLD", "LONG", "SHORT",
                 "FROM", "JUST", "OVER", "SUCH", "TAKE", "THAN", "THEM",
                 "VERY", "WHEN", "COME", "MAKE", "LIKE", "INTO", "YEAR",
                 "BACK", "ALSO", "BEEN", "CALL", "EACH", "EVEN", "FIND",
                 "WANT", "WILL", "WITH", "WHAT", "THIS", "THAT", "HAVE",
                 "KEEP", "NEED", "GOOD", "HIGH", "LAST", "MOST", "SOME",
                 "THEN", "WENT", "WERE", "WELL"}
    matches = _SYM_RE.findall(text)
    return list(dict.fromkeys(m for m in matches if m not in stopwords))


def add_entry(text: str) -> dict:
    """Add a journal entry. Returns the new entry dict."""
    entries = load_entries()
    entry = {
        "id": _next_id(entries),
        "text": text.strip(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbols": _extract_symbols(text),
    }
    entries.append(entry)
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]
    _save_entries(entries)
    return entry


def remove_entry(entry_id: int) -> bool:
    """Remove an entry by ID. Returns True if found and removed."""
    entries = load_entries()
    before = len(entries)
    entries = [e for e in entries if e.get("id") != entry_id]
    if len(entries) == before:
        return False
    _save_entries(entries)
    return True


def search_entries(term: str) -> list[dict]:
    """Search entries by text content (case-insensitive)."""
    entries = load_entries()
    term_lower = term.lower()
    return [e for e in entries if term_lower in e["text"].lower()]
