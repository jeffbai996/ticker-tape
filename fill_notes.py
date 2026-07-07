"""Fill annotations — a one-line thesis note per fill, stored beside the ledger.

The fills ledger (data/fills.csv) is the immutable record of what executed. A
note is the mutable "why" — "trimmed into strength", "starter position", "tax
loss harvest". Six months on, the position-detail view can answer "what was I
thinking" instead of just showing P&L.

Storage is a SIDECAR (data/fill_notes.json), never a new column on fills.csv:
the ledger stays a clean import target, and a note survives a re-import as long
as the fill's identity (date/symbol/side/qty/price) is unchanged.

    { "<fill_key>": "note text", ... }

The key is a short stable hash of the fill's five immutable fields, so the same
fill always maps to the same note regardless of row order or re-import. Money
values are normalized through Decimal→str so 100 and 100.0 don't split the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal

DEFAULT_NOTES_PATH = os.path.join("data", "fill_notes.json")


def fill_key(date, symbol: str, side: str, qty, price) -> str:
    """Stable 12-char key for a fill from its immutable fields.

    date may be a date or ISO string; qty/price anything Decimal-coercible.
    Normalized so equivalent values (100 vs 100.0, ' aapl ' vs 'AAPL') collide
    to the same key intentionally.
    """
    d = date.isoformat() if hasattr(date, "isoformat") else str(date).strip()
    sym = str(symbol).strip().upper()
    sd = str(side).strip().upper()
    q = _norm_num(qty)
    p = _norm_num(price)
    raw = f"{d}|{sym}|{sd}|{q}|{p}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _norm_num(value) -> str:
    """Normalize a numeric to a canonical string (Decimal-normalized), so
    100, 100.0, and Decimal('100') all produce the same token."""
    try:
        d = Decimal(str(value)).normalize()
        # normalize() can yield exponent form (1E+2) for 100 — undo that.
        return format(d, "f")
    except Exception:
        return str(value).strip()


def load_notes(path: str = DEFAULT_NOTES_PATH) -> dict[str, str]:
    """Load the note sidecar. Missing/corrupt file → empty dict (never raises)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return {}


def save_notes(notes: dict[str, str], path: str = DEFAULT_NOTES_PATH) -> None:
    """Atomically write the note sidecar (temp file + rename)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(notes, fh, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def set_note(key: str, text: str, path: str = DEFAULT_NOTES_PATH) -> None:
    """Add or replace a note for a fill key. Empty text removes it."""
    notes = load_notes(path)
    if text and text.strip():
        notes[key] = text.strip()
    else:
        notes.pop(key, None)
    save_notes(notes, path)


def remove_note(key: str, path: str = DEFAULT_NOTES_PATH) -> bool:
    """Delete a note. Returns True if one existed."""
    notes = load_notes(path)
    if key in notes:
        del notes[key]
        save_notes(notes, path)
        return True
    return False


def note_for_fill(fill, notes: dict[str, str]) -> str | None:
    """Look up the note for a Fill-like object (has date/symbol/side/qty/price)."""
    key = fill_key(fill.date, fill.symbol, fill.side, fill.qty, fill.price)
    return notes.get(key)


def annotate_fills(fills: list, path: str = DEFAULT_NOTES_PATH) -> list[dict]:
    """Return [{fill, key, note}] for a list of Fill-like objects, loading the
    sidecar once. Used by listing/detail views to render notes inline."""
    notes = load_notes(path)
    out = []
    for f in fills:
        key = fill_key(f.date, f.symbol, f.side, f.qty, f.price)
        out.append({"fill": f, "key": key, "note": notes.get(key)})
    return out
