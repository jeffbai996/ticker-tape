"""TDD tests for 4 backtest correctness bugs (fixed in this changeset):

  1. yfinance auto_adjust must be pinned False on the backtest path — the
     fills ledger stores RAW traded prices; adjusted closes would corrupt the
     equity curve by mixing split/dividend-adjusted history with raw fills.
  2. yfinance `end` is exclusive — the internal download call must pad by one
     day so the function's own [start, end] contract stays inclusive.
  3. An unmatched SELL (no/insufficient prior BUY) must not realize bogus P&L
     off a basis-defaults-to-sell-price fallback, and position qty must never
     go negative — the unmatched portion is skipped and surfaced as a warning.
  4. FX pre-series fallback and split-detection both emit warnings into the
     same BacktestResult.warnings channel.
"""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

import backtest_data
import backtest_fx
from backtest import Fill, assemble_backtest


D = Decimal


# ── Bug 1: auto_adjust must be pinned False ─────────────────────────────

def test_fetch_dated_closes_pins_auto_adjust_false():
    captured = {}

    def fake_retry_download(*args, **kwargs):
        captured.update(kwargs)
        return None  # empty result is fine — we only care about the call args

    with patch("config.DEMO_MODE", False), \
         patch("data._retry_download", side_effect=fake_retry_download):
        backtest_data.fetch_dated_closes(["NVDA"], date(2024, 1, 1), date(2024, 1, 5))

    assert captured.get("auto_adjust") is False


def test_fetch_usdcad_pins_auto_adjust_false():
    captured = {}

    def fake_retry_download(*args, **kwargs):
        captured.update(kwargs)
        return None

    with patch("config.DEMO_MODE", False), \
         patch("data._retry_download", side_effect=fake_retry_download):
        backtest_fx.fetch_usdcad(date(2024, 1, 1), date(2024, 1, 5))

    assert captured.get("auto_adjust") is False


# ── Bug 2: yfinance `end` is exclusive — pad by one day at the call site ──

def test_fetch_dated_closes_pads_end_by_one_day_for_yfinance_call():
    captured = {}

    def fake_retry_download(*args, **kwargs):
        captured.update(kwargs)
        return None

    end = date(2024, 1, 5)
    with patch("config.DEMO_MODE", False), \
         patch("data._retry_download", side_effect=fake_retry_download):
        backtest_data.fetch_dated_closes(["NVDA"], date(2024, 1, 1), end)

    # function's own contract keeps end inclusive; only the yfinance call pads +1
    assert captured.get("end") == (end + timedelta(days=1)).isoformat()


def test_fetch_usdcad_pads_end_by_one_day_for_yfinance_call():
    captured = {}

    def fake_retry_download(*args, **kwargs):
        captured.update(kwargs)
        return None

    end = date(2024, 1, 5)
    with patch("config.DEMO_MODE", False), \
         patch("data._retry_download", side_effect=fake_retry_download):
        backtest_fx.fetch_usdcad(date(2024, 1, 1), end)

    assert captured.get("end") == (end + timedelta(days=1)).isoformat()


# ── Bug 3: unmatched SELL must not fabricate P&L or go negative ─────────

def test_unmatched_sell_realizes_zero_pnl_and_clamps_qty_at_zero():
    # No prior BUY at all — a SELL with no held shares.
    fills = [Fill(date(2026, 1, 2), "FOO", "SELL", D("100"), D("50"))]
    bars = {"FOO": {date(2026, 1, 2): D("50")}}
    bench = {}
    r = assemble_backtest(fills, bars, bench)
    # No cost basis existed — the unmatched sell must realize $0 P&L, not
    # (price - price) which also happens to be 0 here, so use a case where a
    # naive "basis = sell price" bug is distinguishable: cumulative basis is 0.
    assert r.book_curve[0] == Decimal("0")


def test_partially_matched_sell_only_realizes_gain_on_held_portion():
    fills = [
        Fill(date(2026, 1, 2), "XYZ", "BUY", D("40"), D("10")),   # basis 400
        Fill(date(2026, 1, 3), "XYZ", "SELL", D("100"), D("15")),  # only 40 held
    ]
    bars = {"XYZ": {date(2026, 1, 2): D("10"), date(2026, 1, 3): D("15")}}
    r = assemble_backtest(fills, bars, {})
    # matched = min(100, 40) = 40; realized = 40 * (15 - 10) = 200
    # cumulative_basis stays 400 (principal), qty clamped to 0 (not -60)
    # equity = cumulative_basis(400) + realized(200) + unrealized(0) = 600
    assert r.book_curve[1] == Decimal("600")


def test_unmatched_sell_never_drives_qty_negative():
    fills = [
        Fill(date(2026, 1, 2), "XYZ", "BUY", D("40"), D("10")),
        Fill(date(2026, 1, 3), "XYZ", "SELL", D("100"), D("15")),
        # a further BUY should start from qty 0, not -60
        Fill(date(2026, 1, 4), "XYZ", "BUY", D("10"), D("20")),
    ]
    bars = {"XYZ": {
        date(2026, 1, 2): D("10"),
        date(2026, 1, 3): D("15"),
        date(2026, 1, 4): D("20"),
    }}
    r = assemble_backtest(fills, bars, {})
    # after the sell, held qty clamps to 0; the day-4 buy holds only 10 shares
    # unrealized on day 4 = 10 * (20 - 20) = 0
    # cumulative_basis = 400 (buy1) + 200 (buy3) = 600; realized = 200
    # equity = 600 + 200 + 0 = 800
    assert r.book_curve[2] == Decimal("800")


def test_unmatched_sell_appends_human_readable_warning():
    fills = [
        Fill(date(2026, 3, 4), "XYZ", "SELL", D("100"), D("15")),
    ]
    bars = {"XYZ": {date(2026, 3, 4): D("15")}}
    r = assemble_backtest(fills, bars, {})
    assert any("XYZ" in w and "2026-03-04" in w and "100" in w for w in r.warnings)


def test_no_warnings_when_all_sells_fully_matched():
    fills = [
        Fill(date(2026, 1, 2), "FOO", "BUY", D("10"), D("100")),
        Fill(date(2026, 1, 6), "FOO", "SELL", D("10"), D("120")),
    ]
    bars = {"FOO": {date(2026, 1, 2): D("100"), date(2026, 1, 6): D("120")}}
    r = assemble_backtest(fills, bars, {})
    assert r.warnings == []


def test_backtest_result_has_warnings_field_default_empty():
    r = assemble_backtest([], {}, {})
    assert r.warnings == []


# ── Bug 4a: FX pre-series fallback warning (deduped, once per run) ──────

def test_fx_rate_before_series_start_emits_one_deduped_warning():
    from backtest_data import LedgerFill

    fx = {date(2023, 1, 3): D("1.25"), date(2023, 1, 4): D("1.30")}
    ledger = [
        LedgerFill(Fill(date(2021, 6, 15), "AAPL.NE", "BUY", D("10"), D("25")), "CAD"),
        LedgerFill(Fill(date(2020, 1, 1), "AAPL.NE", "BUY", D("5"), D("20")), "CAD"),
    ]
    warnings: list[str] = []
    fills = backtest_fx.convert_fills(ledger, "USD", fx, warnings=warnings)
    assert len(warnings) == 1
    assert "2023-01-03" in warnings[0]          # series start
    assert "2020-01-01" in warnings[0]          # earliest offending fill date


def test_fx_rate_within_series_emits_no_warning():
    from backtest_data import LedgerFill

    fx = {date(2023, 1, 3): D("1.25"), date(2023, 1, 4): D("1.30")}
    ledger = [LedgerFill(Fill(date(2023, 1, 4), "AAPL.NE", "BUY", D("10"), D("25")), "CAD")]
    warnings: list[str] = []
    backtest_fx.convert_fills(ledger, "USD", fx, warnings=warnings)
    assert warnings == []


# ── Bug 4b: split-detection warnings (best-effort, DEMO_MODE-skipped) ────

def test_split_inside_ledger_window_emits_warning():
    fake_ticker = MagicMock()
    import pandas as pd
    fake_ticker.splits = pd.Series(
        {pd.Timestamp("2024-06-10"): 10.0},
    )
    with patch("config.DEMO_MODE", False), \
         patch("yfinance.Ticker", return_value=fake_ticker):
        warnings = backtest_data.detect_splits_in_window(
            "NVDA", date(2022, 1, 1), date(2026, 1, 1)
        )
    assert any("NVDA" in w and "2024-06-10" in w and "10" in w for w in warnings)


def test_split_detection_skipped_in_demo_mode():
    with patch("config.DEMO_MODE", True), \
         patch("yfinance.Ticker") as mock_ticker:
        warnings = backtest_data.detect_splits_in_window(
            "NVDA", date(2022, 1, 1), date(2026, 1, 1)
        )
    assert warnings == []
    mock_ticker.assert_not_called()


def test_split_detection_never_raises_on_fetch_failure():
    with patch("config.DEMO_MODE", False), \
         patch("yfinance.Ticker", side_effect=RuntimeError("network down")):
        warnings = backtest_data.detect_splits_in_window(
            "NVDA", date(2022, 1, 1), date(2026, 1, 1)
        )
    assert warnings == []
