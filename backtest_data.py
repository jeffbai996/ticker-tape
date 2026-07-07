"""Data layer for the backtest / thesis-replay view.

Two sources, cleanly separated from the pure core (`backtest.py`):

  1. Fills — a LOCAL LEDGER (CSV) the user seeds from IBKR account statements
     or Flex Queries. This is primary: IBKR's live execution API
     (`ibkr_trades`) only reaches back ~7 days, so it cannot reconstruct a
     multi-year book. The ledger is the historical record; IBKR fills are a
     recent-days top-up, out of scope here.

  2. Price bars — daily closes from yfinance (reaches back years for prices,
     unlike the fills feed), returned dated: {symbol: {date: Decimal}}.

Parsing is pure and unit-tested; the network fetch is a thin wrapper over the
retry machinery already in data.py.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from backtest import Fill

log = logging.getLogger(__name__)


# ── fills ledger ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LedgerFill:
    """A ledger row: the fill plus the currency it traded in.

    The pure core is single-currency by design — currency lives out here in
    the data layer, where backtest_fx normalizes everything to the report
    currency before the core ever sees a number.
    """
    fill: Fill
    currency: str  # "USD", "CAD", ...


def load_fills_ledger(path: str) -> list[LedgerFill]:
    """Parse a CSV fills ledger into LedgerFill records.

    Columns: date (ISO), symbol, side (BUY/SELL), qty, price, and an optional
    currency (defaults to USD so pre-currency ledgers keep working). Malformed
    rows are skipped with a warning, never fatal — a seeded ledger is
    hand-edited and one bad line shouldn't sink the whole replay.
    Missing file → [].
    """
    if not os.path.exists(path):
        log.info("fills ledger not found: %s", path)
        return []

    ledger: list[LedgerFill] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=2):  # line 1 is the header
            try:
                fill = Fill(
                    date=datetime.strptime(row["date"].strip(), "%Y-%m-%d").date(),
                    symbol=row["symbol"].strip().upper(),
                    side=row["side"].strip().upper(),
                    qty=Decimal(row["qty"].strip()),
                    price=Decimal(row["price"].strip()),
                )
                currency = (row.get("currency") or "USD").strip().upper() or "USD"
                ledger.append(LedgerFill(fill, currency))
            except (KeyError, ValueError, TypeError, InvalidOperation, AttributeError) as e:
                log.warning("skipping malformed ledger row %d: %s", i, e)
                continue
    ledger.sort(key=lambda lf: (lf.fill.date, lf.fill.symbol))
    return ledger


def symbols_from_fills(fills: list[Fill]) -> list[str]:
    """Sorted unique symbols touched by the ledger — what we need bars for."""
    return sorted({f.symbol for f in fills})


# ── dated price bars ────────────────────────────────────────────────────

def rows_to_dated_closes(
    raw: dict[str, list[tuple[date, float]]]
) -> dict[str, dict[date, Decimal]]:
    """Convert {symbol: [(date, close), ...]} rows into the nested
    {symbol: {date: Decimal}} the core consumes. NaN closes are dropped (a
    yfinance gap), never coerced to 0 — the core carries last-known instead.
    """
    out: dict[str, dict[date, Decimal]] = {}
    for sym, rows in raw.items():
        series: dict[date, Decimal] = {}
        for d, close in rows:
            if close is None or (isinstance(close, float) and math.isnan(close)):
                continue
            series[d] = Decimal(str(close))
        out[sym] = series
    return out


def fetch_dated_closes(
    symbols: list[str], start: date, end: date | None = None
) -> dict[str, dict[date, Decimal]]:
    """Daily closes for `symbols` over [start, end], dated.

    Reuses data.py's retry-wrapped yfinance download. Unlike
    data.fetch_comparison_data (which drops dates and returns bare lists), this
    preserves the DatetimeIndex so the equity curve can align fills to bars.
    """
    import config
    if config.DEMO_MODE:
        return _demo_dated_closes(symbols, start, end)

    import data
    import pandas as pd

    end = end or date.today()
    try:
        df = data._retry_download(
            symbols, start=start.isoformat(), end=end.isoformat(),
            interval="1d", progress=False,
        )
    except Exception as e:
        log.warning("fetch_dated_closes download failed: %s", e)
        return {}
    if df is None or df.empty:
        return {}

    close = df["Close"]
    raw: dict[str, list[tuple[date, float]]] = {}
    if isinstance(close, pd.Series):
        # single-symbol edge case: Close is a Series indexed by date
        raw[symbols[0]] = [(idx.date(), float(v)) for idx, v in close.items()]
    else:
        for sym in symbols:
            if sym in close.columns:
                raw[sym] = [(idx.date(), float(v)) for idx, v in close[sym].items()]
    return rows_to_dated_closes(raw)


def _demo_dated_closes(symbols, start, end):
    """Deterministic fake bars for --demo (no network)."""
    import demo_data
    if hasattr(demo_data, "fetch_dated_closes"):
        return demo_data.fetch_dated_closes(symbols, start, end)
    return {}
