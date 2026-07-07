"""IBKR Flex-Query importer — Flex CSV → the backtest fills ledger.

A Flex Query (Account Management → Reports → Flex Queries, Trades section) is
the only IBKR export that reaches back years; the live execution API stops at
~7 days. This importer turns that export into `data/fills.csv` in one command:

    python flex_import.py my_flex_export.csv
    python flex_import.py my_flex_export.csv -o data/fills.csv --force

Column names vary between Flex configurations, so known aliases are mapped
(TradeDate / Date/Time, TradePrice / Price, CurrencyPrimary / Currency, ...).
When there's no Buy/Sell column, side is inferred from the sign of Quantity.
Canadian listings (currency CAD) get their yfinance suffix from the listing
exchange (NEO → .NE, TSX → .TO) so the price fetch works unmodified.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from backtest import Fill
from backtest_data import LedgerFill

log = logging.getLogger(__name__)

# Column aliases, first match wins. Keys are the ledger's concepts.
_ALIASES = {
    "date": ("TradeDate", "Date/Time", "Date"),
    "symbol": ("Symbol",),
    "side": ("Buy/Sell", "Side"),
    "qty": ("Quantity",),
    "price": ("TradePrice", "Price", "Trade Price"),
    "currency": ("CurrencyPrimary", "Currency"),
    "exchange": ("ListingExchange", "Listing Exchange", "Exchange"),
}

# Listing exchange → yfinance suffix for CAD-denominated symbols.
_EXCHANGE_SUFFIX = {
    "AEQLIT": ".NE", "NEO": ".NE", "AEQ": ".NE",
    "TSE": ".TO", "TSX": ".TO",
    "VENTURE": ".V", "TSXV": ".V",
}

_DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d")


def _pick(row: dict, concept: str) -> str | None:
    for alias in _ALIASES[concept]:
        v = row.get(alias)
        if v is not None and v.strip():
            return v.strip()
    return None


def _parse_date(raw: str) -> date:
    # "Date/Time" cells carry a time part after ',' or ';' — drop it.
    for sep in (";", ","):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {raw!r}")


def _yf_symbol(symbol: str, currency: str, exchange: str | None) -> str:
    """Map a CAD listing to its yfinance ticker via the exchange suffix.
    Symbols that already carry a suffix, or whose exchange we don't know,
    pass through unchanged (a warning beats a wrong guess)."""
    if currency != "CAD" or "." in symbol:
        return symbol
    suffix = _EXCHANGE_SUFFIX.get((exchange or "").upper())
    if suffix is None:
        log.warning("CAD symbol %s has unknown listing exchange %r — "
                    "left unsuffixed; map it by hand if yfinance misses it",
                    symbol, exchange)
        return symbol
    return symbol + suffix


def parse_flex_csv(text: str) -> list[LedgerFill]:
    """Parse a Flex trades CSV into LedgerFill rows.

    Non-trade rows (repeated section headers, subtotals, blanks, zero-qty)
    are skipped with a debug log, never fatal — Flex exports carry noise.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows: list[LedgerFill] = []
    for i, row in enumerate(reader, start=2):
        try:
            raw_date = _pick(row, "date")
            symbol = _pick(row, "symbol")
            raw_qty = _pick(row, "qty")
            raw_price = _pick(row, "price")
            if not (raw_date and symbol and raw_qty and raw_price):
                log.debug("skipping non-trade row %d", i)
                continue
            if raw_date in _ALIASES["date"]:  # repeated section header
                continue

            qty = Decimal(raw_qty.replace(",", ""))
            if qty == 0:
                continue
            side_raw = _pick(row, "side")
            if side_raw:
                side = side_raw.upper()
                if side not in ("BUY", "SELL"):
                    raise ValueError(f"bad side {side_raw!r}")
            else:
                side = "BUY" if qty > 0 else "SELL"
            currency = (_pick(row, "currency") or "USD").upper()
            symbol = _yf_symbol(symbol.upper(), currency, _pick(row, "exchange"))

            rows.append(LedgerFill(
                Fill(
                    date=_parse_date(raw_date),
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    price=Decimal(raw_price.replace(",", "")),
                ),
                currency,
            ))
        except (ValueError, InvalidOperation, KeyError) as e:
            log.debug("skipping row %d: %s", i, e)
            continue
    rows.sort(key=lambda lf: (lf.fill.date, lf.fill.symbol))
    return rows


def write_ledger(rows: list[LedgerFill], path: str, force: bool = False) -> None:
    """Write LedgerFill rows as the backtest's fills.csv (with currency).

    Refuses to overwrite an existing ledger unless `force` — the ledger is the
    book's history of record and a bad import shouldn't silently destroy it.
    """
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists — pass --force to overwrite")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "symbol", "side", "qty", "price", "currency"])
        for lf in rows:
            f = lf.fill
            w.writerow([f.date.isoformat(), f.symbol, f.side, f.qty, f.price,
                        lf.currency])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Import an IBKR Flex-Query trades CSV into the backtest fills ledger.")
    ap.add_argument("flex_csv", help="path to the Flex export CSV")
    ap.add_argument("-o", "--out", default=os.path.join("data", "fills.csv"),
                    help="ledger destination (default: data/fills.csv)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing ledger")
    args = ap.parse_args(argv)

    with open(args.flex_csv, newline="") as fh:
        rows = parse_flex_csv(fh.read())
    if not rows:
        log.error("no trades found in %s — is it a Trades-section Flex export?",
                  args.flex_csv)
        return 1

    write_ledger(rows, args.out, force=args.force)
    first, last = rows[0].fill.date, rows[-1].fill.date
    ccys = sorted({lf.currency for lf in rows})
    log.info("wrote %d fills (%s → %s, %s) to %s",
             len(rows), first, last, "/".join(ccys), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
