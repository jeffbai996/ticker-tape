"""Tests for the time-travel as-of reconstruction (timetravel.py).

Coverage per the feature spec:
  - as-of state mid-ledger, before the first fill, after the last fill
  - P&L against a known dated close
  - matched-sell / avg-cost semantics MATCH the backtest engine (shared code)
  - cache behavior (no recompute on repeat, no refetch)
  - weekend / holiday carry-back (nearest close at-or-before the date)
  - watchlist quotes as-of
  - graceful "no data for this date"

Generic tickers only (AAPL / MSFT / GOOGL) — never real portfolio symbols.
"""

from datetime import date
from decimal import Decimal

import pytest

import timetravel
from timetravel import (
    AsOfBook,
    TimeTravelState,
    price_asof,
    reconstruct_as_of,
    book_asof,
    watch_quotes_asof,
    clamp_date,
    banner_text,
)
from backtest import Fill


D = Decimal


# ── a small deterministic ledger ────────────────────────────────────────
#   2024-01-08  BUY  50 AAPL @ 100   → basis 5000
#   2024-03-12  BUY  20 MSFT @ 400   → basis 8000  (cumulative 13000)
#   2024-09-04  SELL 20 AAPL @ 150   → realize 20*(150-100)=1000; 30 AAPL left
#   2025-01-15  BUY  30 GOOGL @ 200  → basis 6000  (cumulative 19000)

def _ledger():
    return [
        Fill(date(2024, 1, 8), "AAPL", "BUY", D("50"), D("100")),
        Fill(date(2024, 3, 12), "MSFT", "BUY", D("20"), D("400")),
        Fill(date(2024, 9, 4), "AAPL", "SELL", D("20"), D("150")),
        Fill(date(2025, 1, 15), "GOOGL", "BUY", D("30"), D("200")),
    ]


def _bars():
    return {
        "AAPL": {date(2024, 1, 8): D("100"), date(2024, 3, 12): D("110"),
                 date(2024, 9, 4): D("150"), date(2025, 1, 15): D("180")},
        "MSFT": {date(2024, 3, 12): D("400"), date(2024, 9, 4): D("450"),
                 date(2025, 1, 15): D("500")},
        "GOOGL": {date(2025, 1, 15): D("200")},
    }


# ── price_asof: carry-back ──────────────────────────────────────────────

def test_price_asof_exact_date_hit():
    series = {date(2024, 1, 8): D("100"), date(2024, 1, 9): D("105")}
    assert price_asof(series, date(2024, 1, 9)) == D("105")


def test_price_asof_carries_back_over_weekend():
    # Friday close carried through Sat/Sun to next trading query.
    series = {date(2025, 1, 10): D("180")}  # a Friday
    assert price_asof(series, date(2025, 1, 11)) == D("180")  # Saturday
    assert price_asof(series, date(2025, 1, 12)) == D("180")  # Sunday


def test_price_asof_before_first_close_is_none():
    series = {date(2025, 1, 10): D("180")}
    assert price_asof(series, date(2025, 1, 1)) is None


def test_price_asof_empty_series_is_none():
    assert price_asof({}, date(2025, 1, 1)) is None


# ── reconstruct_as_of: before first fill ────────────────────────────────

def test_before_first_fill_is_empty_book():
    b = reconstruct_as_of(_ledger(), date(2023, 12, 31), _bars())
    assert b.positions == []
    assert b.realized_pnl == D("0")
    assert b.cost_basis == D("0")


# ── reconstruct_as_of: mid-ledger ───────────────────────────────────────

def test_midledger_after_two_buys_before_sell():
    # 2024-06-01: both buys applied, sell not yet. AAPL 50 @100, MSFT 20 @400.
    b = reconstruct_as_of(_ledger(), date(2024, 6, 1), _bars())
    syms = {p.symbol: p for p in b.positions}
    assert syms["AAPL"].qty == D("50")
    assert syms["AAPL"].avg_cost == D("100")
    # nearest close at-or-before 2024-06-01 for AAPL is the 2024-03-12 close 110
    assert syms["AAPL"].close == D("110")
    assert syms["AAPL"].unrealized == D("50") * (D("110") - D("100"))  # 500
    assert b.realized_pnl == D("0")
    assert b.cost_basis == D("13000")


def test_midledger_after_partial_sell():
    # 2024-12-01: sell applied. 30 AAPL left, realized 1000. GOOGL not yet.
    b = reconstruct_as_of(_ledger(), date(2024, 12, 1), _bars())
    syms = {p.symbol: p for p in b.positions}
    assert "GOOGL" not in syms
    assert syms["AAPL"].qty == D("30")
    assert b.realized_pnl == D("1000")
    assert b.cost_basis == D("13000")


# ── reconstruct_as_of: after last fill ──────────────────────────────────

def test_after_last_fill_full_book():
    b = reconstruct_as_of(_ledger(), date(2025, 6, 1), _bars())
    syms = {p.symbol: p for p in b.positions}
    assert set(syms) == {"AAPL", "MSFT", "GOOGL"}
    assert syms["GOOGL"].qty == D("30")
    assert b.realized_pnl == D("1000")
    assert b.cost_basis == D("19000")


# ── P&L against a known dated close ─────────────────────────────────────

def test_pnl_against_known_close():
    # On 2025-01-15: AAPL 30 @100 close 180 → unreal 30*80=2400
    #                MSFT 20 @400 close 500 → unreal 20*100=2000
    #                GOOGL 30 @200 close 200 → unreal 0
    # total unrealized 4400; realized 1000; total P&L 5400
    b = reconstruct_as_of(_ledger(), date(2025, 1, 15), _bars())
    assert b.unrealized_pnl == D("4400")
    assert b.realized_pnl == D("1000")
    assert b.total_pnl == D("5400")
    assert b.market_value == (D("30") * D("180") + D("20") * D("500") + D("30") * D("200"))


# ── matched-sell semantics MUST match the backtest engine ───────────────

def test_asof_matches_backtest_equity_on_same_day():
    """The as-of book's cost_basis+realized+unrealized must equal the backtest
    equity curve's value on that day — they share PositionBook, so any drift is
    a real bug."""
    from backtest import assemble_backtest
    fills = _ledger()
    bars = _bars()
    result = assemble_backtest(fills, bars, {})
    target = date(2025, 1, 15)
    idx = result.dates.index(target)
    equity = result.book_curve[idx]

    b = reconstruct_as_of(fills, target, bars)
    asof_equity = b.cost_basis + b.realized_pnl + b.unrealized_pnl
    assert asof_equity == equity


def test_unmatched_sell_carries_warning_and_clamps():
    fills = [Fill(date(2026, 1, 2), "AAPL", "SELL", D("100"), D("50"))]
    bars = {"AAPL": {date(2026, 1, 2): D("50")}}
    b = reconstruct_as_of(fills, date(2026, 1, 2), bars)
    assert b.positions == []  # qty clamped to 0
    assert b.realized_pnl == D("0")
    assert any("AAPL" in w and "100" in w for w in b.warnings)


# ── unknown mark → graceful None, not a fabricated 0 ────────────────────

def test_position_with_no_close_yields_none_not_zero():
    fills = [Fill(date(2024, 1, 8), "AAPL", "BUY", D("10"), D("100"))]
    bars = {"AAPL": {}}  # no closes at all
    b = reconstruct_as_of(fills, date(2024, 6, 1), bars)
    p = b.positions[0]
    assert p.close is None
    assert p.market_value is None
    assert p.unrealized is None
    assert b.market_value is None
    assert b.unrealized_pnl is None
    assert b.total_pnl is None
    # realized is still known even when marks are missing
    assert b.realized_pnl == D("0")


# ── cache behavior ──────────────────────────────────────────────────────

def _state():
    return TimeTravelState(
        report_ccy="USD",
        fills=_ledger(),
        bars=_bars(),
        watch_bars=_bars(),
        min_date=date(2024, 1, 8),
        max_date=date(2025, 6, 1),
    )


def test_book_asof_caches_per_date(monkeypatch):
    state = _state()
    calls = {"n": 0}
    real = timetravel.reconstruct_as_of

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(timetravel, "reconstruct_as_of", counting)

    b1 = book_asof(state, date(2024, 6, 1))
    b2 = book_asof(state, date(2024, 6, 1))
    assert b1 is b2                 # same cached object
    assert calls["n"] == 1          # only reconstructed once

    book_asof(state, date(2024, 12, 1))
    assert calls["n"] == 2          # a new date does recompute


def test_watch_quotes_asof():
    state = _state()
    q = watch_quotes_asof(state, date(2024, 6, 1))
    # AAPL nearest close at-or-before 2024-06-01 is the 2024-03-12 close 110
    assert q["AAPL"] == D("110")
    # GOOGL didn't trade until 2025 → None
    assert q["GOOGL"] is None


# ── clamp / banner ──────────────────────────────────────────────────────

def test_clamp_date_bounds():
    state = _state()
    assert clamp_date(state, date(2020, 1, 1)) == state.min_date
    assert clamp_date(state, date(2030, 1, 1)) == state.max_date
    assert clamp_date(state, date(2024, 6, 1)) == date(2024, 6, 1)


def test_banner_text_mentions_date_and_home():
    txt = banner_text(date(2026, 3, 30))
    assert "2026-03-30" in txt
    assert "Home" in txt
    assert "AS OF" in txt


# ── build_dataset in demo mode (no network) ─────────────────────────────

def test_build_dataset_demo_mode(monkeypatch):
    monkeypatch.setattr("config.DEMO_MODE", True)
    state = timetravel.build_dataset(["AAPL", "MSFT"], end=date(2025, 6, 1))
    assert state is not None
    assert state.report_ccy == "USD"
    assert state.fills                       # demo ledger loaded
    assert state.min_date <= state.max_date
    # a mid-range date reconstructs without network
    b = book_asof(state, date(2025, 3, 3))
    assert isinstance(b, AsOfBook)


def test_build_dataset_returns_none_on_empty_ledger(monkeypatch):
    monkeypatch.setattr("config.DEMO_MODE", False)
    monkeypatch.setattr("backtest_data.load_fills_ledger", lambda p: [])
    assert timetravel.build_dataset(["AAPL"]) is None
