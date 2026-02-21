"""Tests for views.py display helpers and chart rendering."""

import re

from views import (
    _fmt_cap, _fmt_ratio, _fmt_pct, _fmt_num,
    _sparkline, _line_chart, _bar, _rsi_color,
    _safe_pct, _off_high,
)
from config import GREEN, RED, RESET, DIM, WHITE


# ── Formatters ─────────────────────────────────────────────

class TestFmtCap:
    def test_trillions(self):
        assert _fmt_cap(1.5e12) == "$1.50T"

    def test_billions(self):
        assert _fmt_cap(42.3e9) == "$42.30B"

    def test_millions(self):
        assert _fmt_cap(750e6) == "$750.00M"

    def test_small_value(self):
        assert _fmt_cap(50000) == "$50,000"

    def test_none_returns_dash(self):
        assert _fmt_cap(None) == "—"

    def test_negative_billions(self):
        assert _fmt_cap(-5e9) == "-$5.00B"

    def test_negative_millions(self):
        assert _fmt_cap(-120e6) == "-$120.00M"

    def test_zero(self):
        assert _fmt_cap(0) == "$0"


class TestFmtRatio:
    def test_normal(self):
        assert _fmt_ratio(25.123) == "25.12"

    def test_none(self):
        assert _fmt_ratio(None) == "—"


class TestFmtPct:
    def test_normal(self):
        assert _fmt_pct(12.567) == "12.57%"

    def test_none(self):
        assert _fmt_pct(None) == "—"


class TestFmtNum:
    def test_large_number(self):
        assert _fmt_num(1234567) == "1,234,567"

    def test_none(self):
        assert _fmt_num(None) == "—"


# ── Chart helpers ──────────────────────────────────────────

class TestSparkline:
    def test_empty_returns_empty(self):
        assert _sparkline([]) == ""

    def test_single_point_returns_empty(self):
        assert _sparkline([100.0]) == ""

    def test_uptrend_is_green(self):
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert GREEN in result

    def test_downtrend_is_red(self):
        result = _sparkline([5.0, 4.0, 3.0, 2.0, 1.0])
        assert RED in result

    def test_respects_width(self):
        prices = list(range(100))
        result = _sparkline(prices, width=20)
        # Strip ANSI codes to count visible chars
        visible = re.sub(r"\033\[[^m]*m", "", result)
        assert len(visible) == 20


class TestLineChart:
    def test_empty_returns_empty(self):
        assert _line_chart([]) == []

    def test_single_point_returns_empty(self):
        assert _line_chart([100.0]) == []

    def test_returns_correct_height(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
        lines = _line_chart(prices, width=8, height=6)
        assert len(lines) == 6

    def test_flat_line_renders(self):
        """Flat prices should produce a line at one row."""
        prices = [100.0] * 10
        lines = _line_chart(prices, width=10, height=4)
        assert len(lines) == 4
        # All blocks should be on one row (the bottom row for flat line)
        visible_rows = []
        for line in lines:
            stripped = re.sub(r"\033\[[^m]*m", "", line)
            if "█" in stripped:
                visible_rows.append(stripped)
        assert len(visible_rows) >= 1

    def test_uptrend_is_green(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        lines = _line_chart(prices, width=5, height=4)
        assert any(GREEN in line for line in lines)

    def test_downtrend_is_red(self):
        prices = [5.0, 4.0, 3.0, 2.0, 1.0]
        lines = _line_chart(prices, width=5, height=4)
        assert any(RED in line for line in lines)

    def test_no_fill_below_line(self):
        """Line chart should NOT fill below the price — only line points."""
        prices = [1.0, 5.0, 3.0, 5.0, 1.0]
        lines = _line_chart(prices, width=5, height=5)
        # Count total █ chars across all rows — should be much less than filled area
        total_blocks = 0
        for line in lines:
            stripped = re.sub(r"\033\[[^m]*m", "", line)
            total_blocks += stripped.count("█")
        # A filled chart would have ~15 blocks (5 cols * ~3 avg rows), a line chart ~5 + connectors
        assert total_blocks <= 10

    def test_vertical_connectors_on_large_jumps(self):
        """Adjacent columns with large price jumps should have │ connectors."""
        prices = [1.0, 10.0]  # Huge jump
        lines = _line_chart(prices, width=2, height=8)
        all_text = "".join(lines)
        assert "│" in all_text


class TestBar:
    def test_positive_pct_is_green(self):
        result = _bar(2.0, width=10)
        assert GREEN in result

    def test_negative_pct_is_red(self):
        result = _bar(-2.0, width=10)
        assert RED in result


class TestRsiColor:
    def test_overbought_is_red(self):
        assert _rsi_color(75.0) == RED

    def test_oversold_is_green(self):
        assert _rsi_color(25.0) == GREEN

    def test_neutral_is_white(self):
        assert _rsi_color(50.0) == WHITE


# ── Screen helpers ─────────────────────────────────────────

class TestSafePct:
    def test_converts_ratio(self):
        assert _safe_pct(0.25) == 25.0

    def test_none(self):
        assert _safe_pct(None) is None


class TestOffHigh:
    def test_at_high(self):
        info = {"regularMarketPrice": 100, "fiftyTwoWeekHigh": 100}
        assert _off_high(info) == 0.0

    def test_below_high(self):
        info = {"regularMarketPrice": 80, "fiftyTwoWeekHigh": 100}
        result = _off_high(info)
        assert result == -20.0

    def test_missing_data(self):
        assert _off_high({}) is None

    def test_zero_high(self):
        info = {"regularMarketPrice": 50, "fiftyTwoWeekHigh": 0}
        assert _off_high(info) is None
