"""Trade journal — persistent timestamped log of decisions, observations, and rationale.

Distinct from memories (AI context) and history (chat transcript).
Journal entries are the user's own thinking, importable into chat on demand.
"""

import os
import re

from json_store import JsonStore

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "journal.json")
MAX_ENTRIES = 500

_store = JsonStore(JOURNAL_FILE, MAX_ENTRIES)

_SYM_RE = re.compile(r"\b[A-Z]{1,5}\b")

_STOPWORDS = {"I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO",
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


def _extract_symbols(text: str) -> list[str]:
    """Extract likely ticker symbols from text (1-5 uppercase letters)."""
    matches = _SYM_RE.findall(text)
    return list(dict.fromkeys(m for m in matches if m not in _STOPWORDS))


def load_entries() -> list[dict]:
    """Load journal entries from disk."""
    return _store.load()


def add_entry(text: str) -> dict:
    """Add a journal entry. Returns the new entry dict."""
    return _store.add({"text": text.strip(), "symbols": _extract_symbols(text)})


def remove_entry(entry_id: int) -> bool:
    """Remove an entry by ID. Returns True if found and removed."""
    return _store.remove(entry_id)


def search_entries(term: str) -> list[dict]:
    """Search entries by text content (case-insensitive)."""
    entries = load_entries()
    term_lower = term.lower()
    return [e for e in entries if term_lower in e["text"].lower()]
