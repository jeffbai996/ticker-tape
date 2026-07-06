"""Tests for backtest.py — realized equity-curve reconstruction vs benchmark.

The contract, test-first. The money math is the part that must not lie:
a wrong equity curve misreports how the thesis actually performed.
"""

from datetime import date
from decimal import Decimal

import pytest

import backtest
from backtest import Fill, assemble_backtest


# ── fixtures: a tiny, hand-verifiable book ──────────────────────────────
#
# One name, one entry, one exit, three trading days. Every number below is
# computed by hand so the test pins the arithmetic, not the implementation.
#
#   day 1 (2026-01-02): BUY 10 FOO @ $100  → hold 10, book = 10*100 = 1000
#   day 2 (2026-01-05): FOO closes $110     → hold 10, book = 10*110 = 1100
#   day 3 (2026-01-06): SELL 10 FOO @ $120  → hold 0,  book = proceeds 1200
#
# Benchmark BAR closes: 50, 55, 60 → buy-and-hold return +20% over window.

@pytest.fixture
def simple_fills():
    return [
        Fill(date(2026, 1, 2), "FOO", "BUY", Decimal("10"), Decimal("100")),
        Fill(date(2026, 1, 6), "FOO", "SELL", Decimal("10"), Decimal("120")),
    ]


@pytest.fixture
def simple_bars():
    return {
        "FOO": {
            date(2026, 1, 2): Decimal("100"),
            date(2026, 1, 5): Decimal("110"),
            date(2026, 1, 6): Decimal("120"),
        }
    }


@pytest.fixture
def simple_bench():
    return {
        date(2026, 1, 2): Decimal("50"),
        date(2026, 1, 5): Decimal("55"),
        date(2026, 1, 6): Decimal("60"),
    }


# ── equity-curve reconstruction ─────────────────────────────────────────

def test_book_curve_marks_to_market_held_positions(simple_fills, simple_bars, simple_bench):
    r = assemble_backtest(simple_fills, simple_bars, simple_bench)
    # book value each day: entry-day cost, mid-day MTM, exit-day realized
    assert r.book_curve == [Decimal("1000"), Decimal("1100"), Decimal("1200")]


def test_dates_are_aligned_and_sorted(simple_fills, simple_bars, simple_bench):
    r = assemble_backtest(simple_fills, simple_bars, simple_bench)
    assert r.dates == [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]


def test_no_lookahead_later_fill_does_not_affect_earlier_days(simple_bars, simple_bench):
    # The replay starts at the first fill (day 1). A SECOND buy on day 3 must
    # NOT retroactively inflate the book on days 1-2 — no look-ahead.
    fills = [
        Fill(date(2026, 1, 2), "FOO", "BUY", Decimal("10"), Decimal("100")),
        Fill(date(2026, 1, 6), "FOO", "BUY", Decimal("5"), Decimal("120")),
    ]
    r = assemble_backtest(fills, simple_bars, simple_bench)
    # day 1: 10@100 → 1000; day 2: 10 held @110 → 1100 (2nd buy not yet seen);
    # day 3: 15 held, MTM: 10@120 basis100 + 5@120 basis120 = 1200 + 600 = 1800
    assert r.book_curve[0] == Decimal("1000")
    assert r.book_curve[1] == Decimal("1100")
    assert r.book_curve[2] == Decimal("1800")


def test_entry_and_exit_marks_land_on_correct_dates(simple_fills, simple_bars, simple_bench):
    r = assemble_backtest(simple_fills, simple_bars, simple_bench)
    entries = [m for m in r.marks if m.side == "BUY"]
    exits = [m for m in r.marks if m.side == "SELL"]
    assert len(entries) == 1 and entries[0].date == date(2026, 1, 2) and entries[0].symbol == "FOO"
    assert len(exits) == 1 and exits[0].date == date(2026, 1, 6)


# ── benchmark comparison ────────────────────────────────────────────────

def test_benchmark_curve_normalized_to_book_start(simple_fills, simple_bars, simple_bench):
    # Benchmark is buy-and-hold of the SAME starting book value (1000), so it
    # can be compared apples-to-apples on one chart.
    r = assemble_backtest(simple_fills, simple_bars, simple_bench)
    # bench +0%, +10%, +20% off day-1 → 1000, 1100, 1200
    assert r.benchmark_curve == [Decimal("1000"), Decimal("1100"), Decimal("1200")]


def test_stats_total_return_and_alpha(simple_fills, simple_bars, simple_bench):
    r = assemble_backtest(simple_fills, simple_bars, simple_bench)
    # book +20% (1000→1200), bench +20% → alpha 0
    assert r.stats.book_return_pct == pytest.approx(20.0)
    assert r.stats.benchmark_return_pct == pytest.approx(20.0)
    assert r.stats.alpha_pct == pytest.approx(0.0)


def test_alpha_positive_when_book_beats_benchmark(simple_fills, simple_bars):
    # Flat benchmark → book's +20% is pure alpha.
    flat_bench = {d: Decimal("50") for d in (date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6))}
    r = assemble_backtest(simple_fills, simple_bars, flat_bench)
    assert r.stats.benchmark_return_pct == pytest.approx(0.0)
    assert r.stats.alpha_pct == pytest.approx(20.0)


def test_max_drawdown_computed_on_book_curve(simple_bars, simple_bench):
    # Book that peaks then falls: buy 10@100, mark 110 (peak), mark 90.
    bars = {"FOO": {
        date(2026, 1, 2): Decimal("100"),
        date(2026, 1, 5): Decimal("110"),
        date(2026, 1, 6): Decimal("90"),
    }}
    fills = [Fill(date(2026, 1, 2), "FOO", "BUY", Decimal("10"), Decimal("100"))]
    r = assemble_backtest(fills, bars, simple_bench)
    # peak 1100, trough 900 → max DD = (900-1100)/1100 = -18.18%
    assert r.stats.max_drawdown_pct == pytest.approx(-18.1818, abs=1e-3)


# ── horizon honesty + degenerate inputs (never fabricate) ───────────────

def test_empty_fills_yields_empty_result_not_crash(simple_bars, simple_bench):
    r = assemble_backtest([], simple_bars, simple_bench)
    assert r.book_curve == []
    assert r.dates == []
    assert r.stats is None


def test_missing_benchmark_bars_still_produces_book_curve(simple_fills, simple_bars):
    r = assemble_backtest(simple_fills, simple_bars, {})
    assert r.book_curve == [Decimal("1000"), Decimal("1100"), Decimal("1200")]
    # benchmark absent → curve empty, benchmark_return None, NOT a fake 0
    assert r.benchmark_curve == []
    assert r.stats.benchmark_return_pct is None
    assert r.stats.alpha_pct is None


def test_horizon_starts_at_first_fill_not_first_bar(simple_bench):
    # Bars reach back to Jan 2, but the first fill is Jan 5. The replay must
    # start at the first fill (there was no position before it), and report it.
    bars = {"FOO": {
        date(2026, 1, 2): Decimal("100"),
        date(2026, 1, 5): Decimal("110"),
        date(2026, 1, 6): Decimal("120"),
    }}
    fills = [Fill(date(2026, 1, 5), "FOO", "BUY", Decimal("10"), Decimal("110"))]
    r = assemble_backtest(fills, bars, simple_bench)
    assert r.dates[0] == date(2026, 1, 5)
    assert r.horizon_start == date(2026, 1, 5)


def test_missing_price_on_a_day_carries_last_known_not_zero():
    # If a held symbol has no bar on a trading day (holiday gap in one feed),
    # the position is marked at the last known price, never dropped to 0.
    bars = {"FOO": {
        date(2026, 1, 2): Decimal("100"),
        # 2026-01-05 missing for FOO
        date(2026, 1, 6): Decimal("120"),
    }}
    bench = {
        date(2026, 1, 2): Decimal("50"),
        date(2026, 1, 5): Decimal("55"),
        date(2026, 1, 6): Decimal("60"),
    }
    fills = [Fill(date(2026, 1, 2), "FOO", "BUY", Decimal("10"), Decimal("100"))]
    r = assemble_backtest(fills, bars, bench)
    # day 2 (2026-01-05): no FOO price → carry $100 → book stays 1000, not 0
    assert r.book_curve[1] == Decimal("1000")
