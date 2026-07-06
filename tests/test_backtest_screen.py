"""Tests for screens/backtest.py — the CLI render of a BacktestResult.

Render tests are light (per house rules), but the contract worth pinning: it
shows both curves, labels the alpha, renders entry/exit marks, and degrades
honestly on empty/degenerate results (no fabricated numbers, no crash).
"""

from datetime import date
from decimal import Decimal

from backtest import BacktestResult, BacktestStats, Mark
from screens.backtest import format_backtest


def _result(**over):
    base = dict(
        dates=[date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)],
        book_curve=[Decimal("1000"), Decimal("1100"), Decimal("1200")],
        benchmark_curve=[Decimal("1000"), Decimal("1050"), Decimal("1100")],
        marks=[
            Mark(date(2026, 1, 2), "FOO", "BUY", Decimal("10"), Decimal("100")),
            Mark(date(2026, 1, 6), "FOO", "SELL", Decimal("10"), Decimal("120")),
        ],
        stats=BacktestStats(20.0, 10.0, 10.0, -0.0),
        horizon_start=date(2026, 1, 2),
    )
    base.update(over)
    return BacktestResult(**base)


def test_renders_book_and_benchmark_returns():
    out = format_backtest(_result(), benchmark_label="SOXX")
    assert "20.0%" in out          # book return
    assert "10.0%" in out          # benchmark return / alpha
    assert "SOXX" in out


def test_renders_alpha():
    out = format_backtest(_result(), benchmark_label="SOXX")
    # alpha is book - benchmark = +10%; label present
    assert "alpha" in out.lower() or "α" in out


def test_shows_horizon_start_date():
    out = format_backtest(_result(), benchmark_label="SOXX")
    assert "2026-01-02" in out


def test_empty_result_is_graceful():
    empty = BacktestResult([], [], [], [], None, None)
    out = format_backtest(empty, benchmark_label="SOXX")
    assert out                      # returns something, doesn't crash
    assert "%" not in out or "no" in out.lower()   # no fabricated return numbers


def test_missing_benchmark_shows_book_only_not_fake_alpha():
    r = _result(
        benchmark_curve=[],
        stats=BacktestStats(20.0, None, None, -0.0),
    )
    out = format_backtest(r, benchmark_label="SOXX")
    assert "20.0%" in out                      # book return still shown
    # no fabricated benchmark/alpha number
    assert "n/a" in out.lower() or "—" in out or "unavailable" in out.lower()
