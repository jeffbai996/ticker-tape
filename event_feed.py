"""Event feed — pure ring buffer of recent trading events.

Feeds the bottom event-tape strip (widgets/event_tape.py). Kept free of any
Textual / I/O imports so it unit-tests as a plain data structure: callers push
events, the widget pulls a rendered segment list.

An event is a small dict — kind, text, severity, ts — and the feed keeps at
most CAPACITY of them (oldest evicted first). Identical (kind, text) pairs
pushed within DEDUPE_WINDOW_SEC collapse to one entry so a 15s watchlist poll
that keeps seeing the same mover doesn't spam the tape.
"""

from decimal import Decimal
import time as _time
from typing import Callable

# Ring buffer size — how many recent events the tape can rotate through.
CAPACITY = 50

# Suppress a re-add of the same (kind, text) within this many seconds. A mover
# poll fires every 15s; 60s means a symbol that stays over-threshold lands on
# the tape once per minute rather than four times.
DEDUPE_WINDOW_SEC = 60

# Event kinds accepted by add(). Not enforced (add() takes any string) but
# documents the vocabulary and drives the default color map below.
KINDS = ("alert", "breaker", "calendar", "mover", "fill", "info")

# Severity → Rich color. Severities are the small controlled set the callers
# pass; anything unknown falls back to the neutral info color.
SEVERITY_COLORS: dict[str, str] = {
    "critical": "#ff3232",   # red — alert fired / breaker FIRED
    "warning": "#ffc800",    # amber — cushion / caution
    "positive": "green",     # green — up mover
    "negative": "#ff3232",   # red — down mover
    "info": "#00c8ff",       # cyan — calendar / neutral notices
    "neutral": "#00c8ff",
}

# Default severity per kind when a caller doesn't specify one.
KIND_DEFAULT_SEVERITY: dict[str, str] = {
    "alert": "critical",
    "breaker": "warning",
    "calendar": "info",
    "mover": "info",
    "fill": "info",
    "info": "info",
}


def severity_color(severity: str) -> str:
    """Rich color string for a severity, neutral cyan for anything unknown."""
    return SEVERITY_COLORS.get(severity, SEVERITY_COLORS["info"])


def mover_severity(pct: Decimal) -> str:
    """Severity for a mover given its signed daily %.

    Positive/zero → green, negative → red. Decimal in, so callers keep money
    math off floats end-to-end.
    """
    return "positive" if pct >= 0 else "negative"


def exceeds_threshold(pct: Decimal, threshold_pct: Decimal) -> bool:
    """True when |pct| meets or exceeds a positive threshold (Decimal math).

    Pure comparison so the sidebar hook stays a one-liner and the logic is
    unit-tested here rather than inside a Textual widget.
    """
    return abs(pct) >= abs(threshold_pct)


class EventFeed:
    """Fixed-capacity ring buffer of recent events with short-window dedupe.

    `time_fn` is injectable so tests drive the dedupe window deterministically
    instead of sleeping.
    """

    def __init__(self, capacity: int = CAPACITY,
                 dedupe_window: float = DEDUPE_WINDOW_SEC,
                 time_fn: Callable[[], float] = _time.monotonic) -> None:
        self._capacity = capacity
        self._dedupe_window = dedupe_window
        self._time = time_fn
        self._events: list[dict] = []
        # (kind, text) → last-added monotonic ts, for dedupe. Kept separate
        # from _events so an evicted event still suppresses a quick re-add.
        self._last_seen: dict[tuple[str, str], float] = {}

    def add(self, kind: str, text: str, severity: str | None = None) -> bool:
        """Append an event. Returns True if added, False if deduped.

        `severity` defaults to the kind's default when omitted.
        """
        now = self._time()
        key = (kind, text)
        last = self._last_seen.get(key)
        if last is not None and (now - last) < self._dedupe_window:
            return False  # identical event still inside the dedupe window
        self._last_seen[key] = now
        if severity is None:
            severity = KIND_DEFAULT_SEVERITY.get(kind, "info")
        self._events.append({
            "kind": kind,
            "text": text,
            "severity": severity,
            "ts": now,
        })
        if len(self._events) > self._capacity:
            self._events = self._events[-self._capacity:]
        return True

    def clear(self) -> None:
        """Drop all events and dedupe history (used when demo-seeding)."""
        self._events = []
        self._last_seen = {}

    @property
    def events(self) -> list[dict]:
        """A copy of the current events, oldest first."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def segments(self) -> list[tuple[str, str]]:
        """Return (text, color) tuples, newest first.

        The widget joins these with a separator and colorizes each. Newest-first
        so the freshest event leads the tape.
        """
        out: list[tuple[str, str]] = []
        for ev in reversed(self._events):
            out.append((ev["text"], severity_color(ev["severity"])))
        return out

    def render_line(self, width: int) -> str:
        """Plain-text (no markup) one-line render, truncated to `width`.

        Newest event first, events joined by "  ·  ". Empty feed → "". Used for
        tests and any non-Rich consumer; the widget uses segments() for color.
        """
        if width <= 0 or not self._events:
            return ""
        parts = [ev["text"] for ev in reversed(self._events)]
        line = "  ·  ".join(parts)
        if len(line) > width:
            if width <= 1:
                return line[:width]
            line = line[:width - 1] + "…"
        return line
