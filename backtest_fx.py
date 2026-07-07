"""FX normalization for the backtest — mixed CAD/USD books.

The pure core (`backtest.py`) is single-currency by design: summing CAD and
USD amounts in one equity number would be silently wrong. This module sits in
the data layer and converts fills, price bars, and the benchmark to ONE report
currency before the core runs.

Conversion rules (same honesty contract as the rest of the backtest):
  - each amount converts at ITS OWN date's rate, never today's;
  - a missing FX day carries the last-known rate forward;
  - a date before the series starts uses the first available rate
    (mirrors the benchmark's first-at-or-after normalization);
  - no FX data at all when conversion is needed → ValueError, never a
    fabricated parity rate.

The rate series is USDCAD (CAD per 1 USD) from yfinance ("CAD=X").
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal

from backtest import Fill
from backtest_data import LedgerFill

log = logging.getLogger(__name__)

_CAD_SUFFIXES = (".NE", ".TO", ".V")
USDCAD_TICKER = "CAD=X"


# ── currency inference ──────────────────────────────────────────────────

def symbol_currency(symbol: str) -> str:
    """Trading currency implied by the yfinance suffix (benchmarks and other
    symbols that don't come through the ledger). Canadian listings carry
    .NE/.TO/.V; everything else is treated as USD."""
    return "CAD" if symbol.upper().endswith(_CAD_SUFFIXES) else "USD"


def currency_by_symbol(ledger: list[LedgerFill]) -> dict[str, str]:
    """{symbol: currency} from the ledger — a symbol trades in one currency."""
    return {lf.fill.symbol: lf.currency for lf in ledger}


def needs_fx(ledger: list[LedgerFill], report_ccy: str, bench_ccy: str) -> bool:
    """True if any fill or the benchmark is not already in the report currency."""
    if bench_ccy != report_ccy:
        return True
    return any(lf.currency != report_ccy for lf in ledger)


# ── rate lookup ─────────────────────────────────────────────────────────

def _rate_for(
    usdcad: dict[date, Decimal], day: date, warnings: list[date] | None = None
) -> Decimal:
    """USDCAD rate for `day`: exact, else last-known before, else first
    available (day predates the series — record it in `warnings` so the
    caller can emit ONE deduped notice per run instead of one per fill)."""
    if not usdcad:
        raise ValueError("FX conversion needed but no USDCAD data available")
    if day in usdcad:
        return usdcad[day]
    before = [d for d in usdcad if d < day]
    if before:
        return usdcad[max(before)]
    if warnings is not None:
        warnings.append(day)
    return usdcad[min(usdcad)]


def _convert(amount: Decimal, ccy: str, report_ccy: str,
             day: date, usdcad: dict[date, Decimal],
             pre_series_days: list[date] | None = None) -> Decimal:
    if ccy == report_ccy:
        return amount
    rate = _rate_for(usdcad, day, pre_series_days)
    if ccy == "USD" and report_ccy == "CAD":
        return amount * rate
    if ccy == "CAD" and report_ccy == "USD":
        return amount / rate
    raise ValueError(f"unsupported currency pair {ccy}->{report_ccy}")


# ── conversion passes ───────────────────────────────────────────────────

def convert_fills(ledger: list[LedgerFill], report_ccy: str,
                  usdcad: dict[date, Decimal],
                  warnings: list[str] | None = None) -> list[Fill]:
    """Ledger → plain Fills in the report currency, each at its fill-date rate.

    If `warnings` is passed, appends ONE deduped notice when any fill predates
    the FX series (rather than one warning per fill, which would flood a
    long-running book that started years before FX coverage).
    """
    pre_series_days: list[date] = []
    fills = [
        Fill(
            date=lf.fill.date,
            symbol=lf.fill.symbol,
            side=lf.fill.side,
            qty=lf.fill.qty,
            price=_convert(lf.fill.price, lf.currency, report_ccy,
                           lf.fill.date, usdcad, pre_series_days),
        )
        for lf in ledger
    ]
    if warnings is not None and pre_series_days and usdcad:
        earliest_fill = min(pre_series_days)
        series_start = min(usdcad)
        warnings.append(
            f"FX rate unavailable before {series_start.isoformat()} (series start) — "
            f"fill(s) as early as {earliest_fill.isoformat()} used earliest available rate"
        )
    return fills


def convert_bars(
    bars: dict[str, dict[date, Decimal]],
    ccy_by_symbol: dict[str, str],
    report_ccy: str,
    usdcad: dict[date, Decimal],
) -> dict[str, dict[date, Decimal]]:
    """Convert each symbol's daily closes to the report currency, day by day
    (each close at its own date's rate — an FX move is part of the return)."""
    out: dict[str, dict[date, Decimal]] = {}
    for sym, series in bars.items():
        ccy = ccy_by_symbol.get(sym, symbol_currency(sym))
        if ccy == report_ccy:
            out[sym] = series
            continue
        out[sym] = {d: _convert(px, ccy, report_ccy, d, usdcad)
                    for d, px in series.items()}
    return out


# ── rate series fetch ───────────────────────────────────────────────────

def fetch_usdcad(start: date, end: date | None = None) -> dict[date, Decimal]:
    """Daily USDCAD closes over [start, end] from yfinance ("CAD=X").

    Demo mode returns a flat deterministic series (no network) so `bt ... ccy`
    still renders offline.
    """
    import config
    end = end or date.today()
    if config.DEMO_MODE:
        rate = Decimal("1.35")
        days = (end - start).days + 1
        return {start + timedelta(days=i): rate for i in range(days)}

    import backtest_data
    # Routes through backtest_data.fetch_dated_closes, which already pins
    # auto_adjust=False and pads `end` +1 day at the yfinance call boundary
    # (FX has no splits, but pinning keeps the call explicit/deterministic).
    series = backtest_data.fetch_dated_closes([USDCAD_TICKER], start, end)
    return series.get(USDCAD_TICKER, {})
