"""Event tape — persistent one-line bottom strip of recent trading events.

Distinct from widgets/ticker_tape.py (the full-screen animated quote tape).
This is a single always-visible row docked just above the command bar that
scrolls the newest events (alert fires, breaker flips, today's calendar,
big watchlist movers, fills) from the app's EventFeed.

The widget owns only presentation: it pulls segments from EventFeed, joins
them into a Rich-markup line, and rotates that line one column per tick for a
marquee scroll. All event vocabulary/dedupe/colour lives in the pure
event_feed module.
"""

from textual.widgets import Static

from formatters import ACC, DIM_HEX
from i18n import t

# Marquee scroll cadence (seconds per one-column shift).
_SCROLL_INTERVAL = 0.4
# Gap between the tail and head of the looped line so the scroll reads as a
# continuous belt rather than an abrupt wrap.
_LOOP_GAP = "     "
# Separator drawn between events.
_SEP = "  ·  "


class EventTape(Static):
    """One-line docked marquee of recent events from EventFeed.

    Rebuilds its source string each tick from the live feed (so newly added
    events appear on the next frame) and scrolls it by rotating the render
    offset. Pauses ingestion/scroll under time travel, showing a frozen note.

    Subclasses Static (not bare Widget) so the tape IS its own renderable
    surface — a wrapper Widget that only composes a child produced a null
    own-visual under `dock: bottom` and crashed the compositor.
    """

    DEFAULT_CSS = """
    EventTape {
        dock: bottom;
        height: 1;
        background: #12121a;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(self, feed, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._feed = feed
        self._offset = 0
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(_SCROLL_INTERVAL, self._tick)
        self._repaint()

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()

    # ── time-travel gate ──

    def _time_traveling(self) -> bool:
        """True when the app is replaying an as-of date (freeze live events)."""
        return getattr(self.app, "_tt_state", None) is not None

    def _tick(self) -> None:
        """Advance the marquee one column (skipped while time traveling)."""
        if self._time_traveling():
            self._repaint()
            return
        self._offset += 1
        self._repaint()

    def _plain_line(self) -> str:
        """The undecorated event string (newest first), or empty."""
        segs = self._feed.segments()
        if not segs:
            return ""
        return _SEP.join(text for text, _ in segs)

    def _markup_segments(self) -> list[tuple[str, str]]:
        """(text, color) list including separators, for colored rendering."""
        segs = self._feed.segments()
        out: list[tuple[str, str]] = []
        for i, (text, color) in enumerate(segs):
            if i:
                out.append((_SEP, DIM_HEX))
            out.append((text, color))
        return out

    def _repaint(self) -> None:
        if self._time_traveling():
            self.update(f"[dim]⏸ {t('eventtape.frozen')}[/]")
            return

        plain = self._plain_line()
        if not plain:
            self.update(f"[dim]{t('eventtape.idle')}[/]")
            return

        width = max(self.size.width - 2, 10)
        # Short line that fits: static, colored per-segment, no scroll.
        if len(plain) <= width:
            markup = "".join(
                f"[{color}]{_esc(text)}[/]" for text, color in self._markup_segments()
            )
            self.update(markup)
            return

        # Long line: build a looped belt and window it at the current offset.
        belt = plain + _LOOP_GAP + plain
        period = len(plain) + len(_LOOP_GAP)
        start = self._offset % period
        window = belt[start:start + width]
        self.update(f"[{ACC}]{_esc(window)}[/]")


def _esc(text: str) -> str:
    """Escape Rich markup brackets so event text renders literally."""
    return text.replace("[", "\\[")
