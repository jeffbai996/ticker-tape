"""Tests for event_feed.py — pure event ring buffer logic (not the widget)."""

from decimal import Decimal

import pytest

from event_feed import (
    EventFeed,
    CAPACITY,
    DEDUPE_WINDOW_SEC,
    SEVERITY_COLORS,
    KIND_DEFAULT_SEVERITY,
    KINDS,
    severity_color,
    mover_severity,
    exceeds_threshold,
)


class _Clock:
    """Injectable monotonic clock — tests advance it explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


class TestRingBuffer:
    def test_add_appends_and_reports_length(self):
        feed = EventFeed()
        assert feed.add("info", "hello") is True
        assert len(feed) == 1
        assert feed.events[0]["text"] == "hello"

    def test_capacity_evicts_oldest(self):
        feed = EventFeed(capacity=CAPACITY)
        for i in range(CAPACITY + 10):
            feed.add("info", f"event {i}")
        assert len(feed) == CAPACITY
        texts = [e["text"] for e in feed.events]
        # Oldest 10 evicted — first surviving is event 10.
        assert texts[0] == "event 10"
        assert texts[-1] == f"event {CAPACITY + 9}"

    def test_custom_capacity(self):
        feed = EventFeed(capacity=3)
        for i in range(5):
            feed.add("info", f"e{i}")
        assert len(feed) == 3
        assert [e["text"] for e in feed.events] == ["e2", "e3", "e4"]

    def test_events_returns_copy(self):
        feed = EventFeed()
        feed.add("info", "x")
        snap = feed.events
        snap.append({"junk": True})
        assert len(feed) == 1  # mutation of returned list doesn't leak

    def test_clear_resets_events_and_dedupe(self):
        clock = _Clock()
        feed = EventFeed(time_fn=clock)
        feed.add("info", "x")
        feed.clear()
        assert len(feed) == 0
        # After clear, an identical add is not suppressed by stale dedupe state.
        assert feed.add("info", "x") is True


class TestDedupe:
    def test_identical_within_window_is_deduped(self):
        clock = _Clock()
        feed = EventFeed(time_fn=clock)
        assert feed.add("mover", "AAPL +3%") is True
        clock.advance(DEDUPE_WINDOW_SEC - 1)
        assert feed.add("mover", "AAPL +3%") is False
        assert len(feed) == 1

    def test_identical_after_window_is_readded(self):
        clock = _Clock()
        feed = EventFeed(time_fn=clock)
        assert feed.add("mover", "AAPL +3%") is True
        clock.advance(DEDUPE_WINDOW_SEC + 1)
        assert feed.add("mover", "AAPL +3%") is True
        assert len(feed) == 2

    def test_different_text_not_deduped(self):
        clock = _Clock()
        feed = EventFeed(time_fn=clock)
        feed.add("mover", "AAPL +3%")
        feed.add("mover", "MSFT +3%")
        assert len(feed) == 2

    def test_different_kind_same_text_not_deduped(self):
        clock = _Clock()
        feed = EventFeed(time_fn=clock)
        feed.add("alert", "TSLA")
        feed.add("info", "TSLA")
        assert len(feed) == 2

    def test_dedupe_survives_eviction(self):
        """An evicted event still suppresses a quick re-add within the window."""
        clock = _Clock()
        feed = EventFeed(capacity=2, time_fn=clock)
        feed.add("info", "keep-out")
        feed.add("info", "a")
        feed.add("info", "b")  # evicts "keep-out" from _events
        assert "keep-out" not in [e["text"] for e in feed.events]
        # Still within window → re-add of the evicted text is suppressed.
        assert feed.add("info", "keep-out") is False


class TestSeverityColor:
    def test_each_kind_default_severity_maps_to_a_color(self):
        for kind in KINDS:
            sev = KIND_DEFAULT_SEVERITY[kind]
            assert sev in SEVERITY_COLORS
            assert severity_color(sev)

    def test_add_uses_kind_default_severity(self):
        feed = EventFeed()
        feed.add("alert", "x")
        assert feed.events[0]["severity"] == KIND_DEFAULT_SEVERITY["alert"]

    def test_explicit_severity_overrides_default(self):
        feed = EventFeed()
        feed.add("alert", "x", "warning")
        assert feed.events[0]["severity"] == "warning"

    def test_unknown_severity_falls_back_to_info(self):
        assert severity_color("nonsense") == SEVERITY_COLORS["info"]

    def test_critical_is_red_positive_is_green(self):
        assert severity_color("critical") == "#ff3232"
        assert severity_color("positive") == "green"
        assert severity_color("negative") == "#ff3232"

    def test_segments_carry_color_newest_first(self):
        feed = EventFeed()
        feed.add("mover", "up", "positive")
        feed.add("mover", "down", "negative")
        segs = feed.segments()
        assert segs[0] == ("down", "#ff3232")
        assert segs[1] == ("up", "green")


class TestMoverLogic:
    def test_mover_severity_positive_and_zero_are_green(self):
        assert mover_severity(Decimal("2.5")) == "positive"
        assert mover_severity(Decimal("0")) == "positive"

    def test_mover_severity_negative_is_red(self):
        assert mover_severity(Decimal("-2.5")) == "negative"

    def test_exceeds_threshold_true_when_over(self):
        assert exceeds_threshold(Decimal("3.0"), Decimal("2.0")) is True
        assert exceeds_threshold(Decimal("-3.0"), Decimal("2.0")) is True

    def test_exceeds_threshold_boundary_is_inclusive(self):
        assert exceeds_threshold(Decimal("2.0"), Decimal("2.0")) is True

    def test_exceeds_threshold_false_when_under(self):
        assert exceeds_threshold(Decimal("1.9"), Decimal("2.0")) is False
        assert exceeds_threshold(Decimal("-1.9"), Decimal("2.0")) is False


class TestRenderLine:
    def test_empty_feed_renders_empty(self):
        feed = EventFeed()
        assert feed.render_line(80) == ""

    def test_newest_first_order(self):
        feed = EventFeed()
        feed.add("info", "first")
        feed.add("info", "second")
        line = feed.render_line(80)
        assert line.index("second") < line.index("first")

    def test_narrow_width_truncates_with_ellipsis(self):
        feed = EventFeed()
        feed.add("info", "a very long event description that overflows")
        line = feed.render_line(10)
        assert len(line) == 10
        assert line.endswith("…")

    def test_zero_width_renders_empty(self):
        feed = EventFeed()
        feed.add("info", "x")
        assert feed.render_line(0) == ""

    def test_width_one_does_not_crash(self):
        feed = EventFeed()
        feed.add("info", "hello")
        line = feed.render_line(1)
        assert len(line) <= 1

    def test_fits_within_width_unchanged(self):
        feed = EventFeed()
        feed.add("info", "hi")
        assert feed.render_line(80) == "hi"
