"""Tests for the backtest data layer — fills ledger + dated price bars.

The IBKR execution API (`ibkr_trades`) only reaches back ~7 days, so it cannot
reconstruct a multi-year book. The PRIMARY fill source is therefore a local
ledger the user seeds from IBKR account statements / Flex Queries (which do go
back years); IBKR fills are a live top-up, out of scope for the historical
replay. These tests pin the ledger parser and the dated-bar shape.
"""

from datetime import date
from decimal import Decimal

import pytest

import backtest_data
from backtest import Fill


# ── fills ledger (CSV) ──────────────────────────────────────────────────

def test_load_fills_ledger_parses_rows(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_text(
        "date,symbol,side,qty,price\n"
        "2022-03-15,NVDA,BUY,100,25.50\n"
        "2024-06-01,NVDA,SELL,40,120.00\n"
    )
    fills = backtest_data.load_fills_ledger(str(p))
    assert fills == [
        Fill(date(2022, 3, 15), "NVDA", "BUY", Decimal("100"), Decimal("25.50")),
        Fill(date(2024, 6, 1), "NVDA", "SELL", Decimal("40"), Decimal("120.00")),
    ]


def test_load_fills_ledger_normalizes_side_and_symbol_case(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_text("date,symbol,side,qty,price\n2023-01-01,nvda,buy,10,50\n")
    fills = backtest_data.load_fills_ledger(str(p))
    assert fills[0].symbol == "NVDA"
    assert fills[0].side == "BUY"


def test_load_fills_ledger_missing_file_returns_empty(tmp_path):
    assert backtest_data.load_fills_ledger(str(tmp_path / "nope.csv")) == []


def test_load_fills_ledger_skips_malformed_rows_without_crashing(tmp_path):
    p = tmp_path / "fills.csv"
    p.write_text(
        "date,symbol,side,qty,price\n"
        "2022-03-15,NVDA,BUY,100,25.50\n"
        "garbage,row,here\n"                       # too few cols
        "2022-bad-date,NVDA,BUY,1,1\n"             # unparseable date
        "2024-06-01,NVDA,SELL,40,120.00\n"
    )
    fills = backtest_data.load_fills_ledger(str(p))
    # the two valid rows survive; the two bad ones are skipped, not fatal
    assert len(fills) == 2
    assert fills[0].symbol == "NVDA" and fills[1].side == "SELL"


def test_symbols_from_fills_is_sorted_unique():
    fills = [
        Fill(date(2022, 1, 1), "MU", "BUY", Decimal("1"), Decimal("1")),
        Fill(date(2022, 1, 2), "AVGO", "BUY", Decimal("1"), Decimal("1")),
        Fill(date(2022, 1, 3), "MU", "SELL", Decimal("1"), Decimal("1")),
    ]
    assert backtest_data.symbols_from_fills(fills) == ["AVGO", "MU"]


# ── dated price bars (yfinance-backed, but the parse is what we test) ────

def test_frame_to_dated_closes_shape():
    # A minimal stand-in for the yfinance MultiIndex Close frame: the function
    # under test converts {symbol: [(date, close), ...]} rows into the nested
    # {symbol: {date: Decimal}} the core consumes.
    raw = {
        "NVDA": [(date(2022, 3, 15), 25.5), (date(2022, 3, 16), 26.0)],
        "MU":   [(date(2022, 3, 15), 80.0)],
    }
    out = backtest_data.rows_to_dated_closes(raw)
    assert out["NVDA"][date(2022, 3, 15)] == Decimal("25.5")
    assert out["MU"][date(2022, 3, 15)] == Decimal("80.0")
    # all values are Decimal (money-math contract), not float
    assert all(isinstance(v, Decimal)
               for series in out.values() for v in series.values())


def test_rows_to_dated_closes_drops_nan_prices():
    raw = {"NVDA": [(date(2022, 3, 15), 25.5), (date(2022, 3, 16), float("nan"))]}
    out = backtest_data.rows_to_dated_closes(raw)
    assert date(2022, 3, 16) not in out["NVDA"]
    assert out["NVDA"][date(2022, 3, 15)] == Decimal("25.5")
