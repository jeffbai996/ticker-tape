"""Tests for shadows — counterfactual replay and delta vs the real book."""

import os
from datetime import date
from decimal import Decimal

import pytest

import shadows
from backtest import Fill


def _bars(symbol, series):
    return {symbol: {d: Decimal(str(p)) for d, p in series.items()}}


# ── exact hand-computed delta ──────────────────────────────────────────────

def test_shadow_delta_is_exact():
    # Real book: BUY 100 AAPL @ 10 on day1, SELL 50 @ 12 on day2.
    # Shadow "never sold": BUY 100 AAPL @ 10 on day1, holds.
    # Bars: AAPL 10 on day1, 12 on day2. Benchmark irrelevant to book_curve.
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    bars = _bars("AAPL", {d1: 10, d2: 12})
    bench = {d1: Decimal("100"), d2: Decimal("110")}

    real_fills = [
        Fill(d1, "AAPL", "BUY", Decimal("100"), Decimal("10")),
        Fill(d2, "AAPL", "SELL", Decimal("50"), Decimal("12")),
    ]
    shadow_fills = [Fill(d1, "AAPL", "BUY", Decimal("100"), Decimal("10"))]

    # Equity model: cumulative_basis + realized + unrealized (mark to close).
    # Real, end of day2: basis=1000, realized=50*(12-10)=100,
    #   held=50 @ avg 10 marked to 12 → unrealized=50*(12-10)=100. Total=1200.
    real_final = shadows.replay_shadow(real_fills, bars, bench)
    assert real_final == Decimal("1200")

    # Shadow, end of day2: basis=1000, realized=0,
    #   held=100 @ 10 marked to 12 → unrealized=100*2=200. Total=1200.
    shadow_final = shadows.replay_shadow(shadow_fills, bars, bench)
    assert shadow_final == Decimal("1200")

    results = shadows.compare_shadows(
        real_final, [("never sold", shadow_fills)], bars, bench)
    assert len(results) == 1
    # At 12, holding 50 more @ cost 10 exactly offsets the realized gain →
    # delta is exactly 0 at this price. (The paths diverge above/below 12.)
    assert results[0].delta == Decimal("0")


def test_shadow_delta_nonzero_when_price_moves_past_sale():
    # Same setup but mark day2 at 20: the shadow that held 100 beats the real
    # book that sold 50 at 12.
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    bars = _bars("AAPL", {d1: 10, d2: 20})
    bench = {d1: Decimal("100"), d2: Decimal("100")}
    real_fills = [
        Fill(d1, "AAPL", "BUY", Decimal("100"), Decimal("10")),
        Fill(d2, "AAPL", "SELL", Decimal("50"), Decimal("12")),
    ]
    shadow_fills = [Fill(d1, "AAPL", "BUY", Decimal("100"), Decimal("10"))]
    # Real: basis 1000 + realized 50*(12-10)=100 + unreal 50*(20-10)=500 = 1600.
    # Shadow: basis 1000 + unreal 100*(20-10)=1000 = 2000. Delta +400.
    real_final = shadows.replay_shadow(real_fills, bars, bench)
    results = shadows.compare_shadows(
        real_final, [("never sold", shadow_fills)], bars, bench)
    assert results[0].delta == Decimal("400")
    assert results[0].delta_pct == pytest.approx(25.0)


# ── robustness ─────────────────────────────────────────────────────────────

def test_missing_shadows_dir_returns_empty(tmp_path):
    missing = os.path.join(str(tmp_path), "nope")
    assert shadows.list_shadow_files(missing) == []


def test_malformed_shadow_does_not_crash_comparison():
    d1 = date(2026, 1, 5)
    bars = _bars("AAPL", {d1: 10})
    bench = {d1: Decimal("100")}

    def _boom(*a, **k):
        raise ValueError("bad csv")

    # A shadow whose fills object explodes on iteration in assemble_backtest.
    class BadFills(list):
        def __iter__(self):
            raise ValueError("corrupt")

    results = shadows.compare_shadows(
        Decimal("1000"), [("broken", BadFills())], bars, bench)
    assert results[0].error is not None
    assert results[0].delta is None


def test_empty_shadow_reports_no_data():
    d1 = date(2026, 1, 5)
    bars = _bars("AAPL", {d1: 10})
    results = shadows.compare_shadows(
        Decimal("1000"), [("empty", [])], bars, {d1: Decimal("100")})
    assert results[0].final_value is None


def test_shadow_name_from_filename():
    assert shadows.shadow_name("data/shadows/never_trimmed.csv") == "never trimmed"


def test_load_shadow_fills_roundtrip(tmp_path):
    p = os.path.join(str(tmp_path), "s.csv")
    with open(p, "w") as fh:
        fh.write("date,symbol,side,qty,price,currency\n")
        fh.write("2026-01-05,AAPL,BUY,100,10.00,USD\n")
    fills = shadows.load_shadow_fills(p)
    assert len(fills) == 1
    assert fills[0].symbol == "AAPL"
    assert fills[0].qty == Decimal("100")


def test_decision_cost_line_framing():
    from shadows import ShadowResult, decision_cost_line
    up = ShadowResult("kept it", Decimal("2000"), Decimal("1600"),
                      Decimal("400"), 25.0)
    assert "would have done better" in decision_cost_line(up)
    down = ShadowResult("sold early", Decimal("1400"), Decimal("1600"),
                        Decimal("-200"), -12.5)
    assert "better call" in decision_cost_line(down)


def test_list_shadow_files_sorted(tmp_path):
    sd = os.path.join(str(tmp_path), "shadows")
    os.makedirs(sd)
    for name in ("b.csv", "a.csv", "ignore.txt"):
        open(os.path.join(sd, name), "w").close()
    files = shadows.list_shadow_files(sd)
    assert [os.path.basename(f) for f in files] == ["a.csv", "b.csv"]
