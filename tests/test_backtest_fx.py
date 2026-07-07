"""Tests for backtest FX normalization — mixed CAD/USD books.

The core (`backtest.py`) is deliberately single-currency; a book that mixes
USD tickers and CAD listings (CDRs, TSX ETFs) must be normalized to one
report currency BEFORE the core sees it, or the equity sum is silently wrong.
Conversion happens at each date's FX rate (never today's), gaps carry the
last-known rate, and missing FX data degrades honestly rather than assuming
parity.
"""

from datetime import date
from decimal import Decimal

import pytest

import backtest_fx
from backtest import Fill, assemble_backtest
from backtest_data import LedgerFill


D = Decimal
# usdcad: CAD per 1 USD
FX = {
    date(2022, 1, 3): D("1.25"),
    date(2022, 1, 4): D("1.30"),
    # 1/5 missing — a gap
    date(2022, 1, 6): D("1.40"),
}


def lf(d, sym, side, qty, price, ccy):
    return LedgerFill(Fill(d, sym, side, D(qty), D(price)), ccy)


# ── fill conversion ─────────────────────────────────────────────────────

def test_cad_fills_converted_at_fill_date_fx():
    ledger = [lf(date(2022, 1, 3), "AAPL.NE", "BUY", "100", "25.00", "CAD")]
    fills = backtest_fx.convert_fills(ledger, "USD", FX)
    # 25.00 CAD / 1.25 = 20.00 USD — the 1/3 rate, not a later one
    assert fills[0].price == D("20.00")
    assert fills[0].qty == D("100")


def test_usd_fills_converted_to_cad():
    ledger = [lf(date(2022, 1, 4), "MSFT", "BUY", "10", "100.00", "USD")]
    fills = backtest_fx.convert_fills(ledger, "CAD", FX)
    assert fills[0].price == D("130.000")  # 100 × 1.30


def test_same_currency_fill_untouched():
    ledger = [lf(date(2022, 1, 3), "MSFT", "BUY", "10", "100.00", "USD")]
    fills = backtest_fx.convert_fills(ledger, "USD", FX)
    assert fills[0].price == D("100.00")


def test_missing_fx_day_carries_last_known():
    ledger = [lf(date(2022, 1, 5), "AAPL.NE", "BUY", "10", "26.00", "CAD")]
    fills = backtest_fx.convert_fills(ledger, "USD", FX)
    # 1/5 has no rate; the 1/4 rate (1.30) carries forward
    assert fills[0].price == D("20.00")  # 26.00 / 1.30


def test_fx_leading_gap_uses_first_available_rate():
    ledger = [lf(date(2022, 1, 1), "AAPL.NE", "BUY", "10", "25.00", "CAD")]
    fills = backtest_fx.convert_fills(ledger, "USD", FX)
    # 1/1 predates the series; first available (1/3 → 1.25) is used, honestly
    assert fills[0].price == D("20.00")


def test_conversion_needed_with_empty_fx_raises():
    ledger = [lf(date(2022, 1, 3), "AAPL.NE", "BUY", "10", "25.00", "CAD")]
    with pytest.raises(ValueError):
        backtest_fx.convert_fills(ledger, "USD", {})


# ── bar conversion ──────────────────────────────────────────────────────

def test_cad_bars_converted_daily():
    bars = {"AAPL.NE": {date(2022, 1, 3): D("25.00"), date(2022, 1, 4): D("26.00")}}
    out = backtest_fx.convert_bars(bars, {"AAPL.NE": "CAD"}, "USD", FX)
    assert out["AAPL.NE"][date(2022, 1, 3)] == D("20.00")   # /1.25
    assert out["AAPL.NE"][date(2022, 1, 4)] == D("20.00")   # 26/1.30
    # each day uses ITS OWN rate, not one frozen rate


def test_bars_in_report_currency_untouched():
    bars = {"MSFT": {date(2022, 1, 3): D("100")}}
    out = backtest_fx.convert_bars(bars, {"MSFT": "USD"}, "USD", FX)
    assert out["MSFT"][date(2022, 1, 3)] == D("100")


# ── currency inference helpers ──────────────────────────────────────────

def test_symbol_currency_by_suffix():
    assert backtest_fx.symbol_currency("AAPL.NE") == "CAD"
    assert backtest_fx.symbol_currency("VFV.TO") == "CAD"
    assert backtest_fx.symbol_currency("XYZ.V") == "CAD"
    assert backtest_fx.symbol_currency("MSFT") == "USD"
    assert backtest_fx.symbol_currency("QQQ") == "USD"


def test_currency_by_symbol_from_ledger():
    ledger = [
        lf(date(2022, 1, 3), "AAPL.NE", "BUY", "1", "1", "CAD"),
        lf(date(2022, 1, 3), "MSFT", "BUY", "1", "1", "USD"),
    ]
    assert backtest_fx.currency_by_symbol(ledger) == {"AAPL.NE": "CAD", "MSFT": "USD"}


def test_needs_fx_only_when_currencies_mix():
    usd_only = [lf(date(2022, 1, 3), "MSFT", "BUY", "1", "1", "USD")]
    mixed = usd_only + [lf(date(2022, 1, 3), "AAPL.NE", "BUY", "1", "1", "CAD")]
    assert backtest_fx.needs_fx(usd_only, "USD", "USD") is False
    assert backtest_fx.needs_fx(usd_only, "CAD", "USD") is True
    assert backtest_fx.needs_fx(mixed, "USD", "USD") is True


# ── end-to-end: mixed-currency book through the core ────────────────────

def test_mixed_currency_book_equity_matches_hand_computed():
    """USD book + one CAD position, reported in USD.

    Hand math (report USD):
      1/3: BUY 10 MSFT @ 100 USD  → basis 1000
           BUY 100 CDR @ 25 CAD   → 25/1.25 = 20 USD → basis 2000
           closes: MSFT 100, CDR 25 CAD (=20 USD) → unrealized 0
           equity = 3000
      1/4: closes: MSFT 110, CDR 26 CAD (26/1.30 = 20 USD)
           unrealized = 10×(110-100) + 100×(20-20) = 100 → equity 3100
    """
    ledger = [
        lf(date(2022, 1, 3), "MSFT", "BUY", "10", "100", "USD"),
        lf(date(2022, 1, 3), "CDR.NE", "BUY", "100", "25", "CAD"),
    ]
    bars_native = {
        "MSFT": {date(2022, 1, 3): D("100"), date(2022, 1, 4): D("110")},
        "CDR.NE": {date(2022, 1, 3): D("25"), date(2022, 1, 4): D("26")},
    }
    ccy = backtest_fx.currency_by_symbol(ledger)
    fills = backtest_fx.convert_fills(ledger, "USD", FX)
    bars = backtest_fx.convert_bars(bars_native, ccy, "USD", FX)
    benchmark = {date(2022, 1, 3): D("100"), date(2022, 1, 4): D("101")}  # USD index

    result = assemble_backtest(fills, bars, benchmark)
    assert result.book_curve[0] == D("3000")
    assert result.book_curve[1] == D("3100")
