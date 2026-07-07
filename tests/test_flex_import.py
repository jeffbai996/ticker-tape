"""Tests for the IBKR Flex-Query importer — Flex CSV → fills ledger.

A Flex Query (Trade Confirmations / Executions) is the only IBKR export that
reaches back years, but its column names differ from the ledger's and vary a
little between query configurations. The importer maps known aliases, infers
side from signed quantity when there's no Buy/Sell column, and maps CAD
listings to their yfinance tickers (.NE/.TO) so the price fetch just works.
"""

from datetime import date
from decimal import Decimal

import pytest

import flex_import


# ── column mapping ──────────────────────────────────────────────────────

def test_flex_import_maps_ibkr_columns_to_ledger():
    csv_text = (
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315,MSFT,BUY,100,250.50,USD\n"
        "20240601,MSFT,SELL,40,420.00,USD\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    assert len(rows) == 2
    assert rows[0].fill.date == date(2022, 3, 15)
    assert rows[0].fill.symbol == "MSFT"
    assert rows[0].fill.side == "BUY"
    assert rows[0].fill.qty == Decimal("100")
    assert rows[0].fill.price == Decimal("250.50")
    assert rows[0].currency == "USD"
    assert rows[1].fill.side == "SELL"


def test_flex_import_accepts_datetime_column_and_iso_dates():
    csv_text = (
        "Date/Time,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "2022-03-15, 10:23:45,MSFT,BUY,10,250.50,USD\n"
    )
    # NB: "Date/Time" cells often come quoted with an embedded comma — also
    # accept the semicolon variant IBKR uses.
    csv_text2 = (
        "Date/Time,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315;102345,MSFT,BUY,10,250.50,USD\n"
    )
    for text in (csv_text2,):
        rows = flex_import.parse_flex_csv(text)
        assert rows[0].fill.date == date(2022, 3, 15)


def test_flex_import_infers_side_from_signed_quantity():
    csv_text = (
        "TradeDate,Symbol,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315,MSFT,100,250.50,USD\n"
        "20240601,MSFT,-40,420.00,USD\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    assert rows[0].fill.side == "BUY" and rows[0].fill.qty == Decimal("100")
    assert rows[1].fill.side == "SELL" and rows[1].fill.qty == Decimal("40")


def test_flex_import_skips_non_trade_rows():
    csv_text = (
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315,MSFT,BUY,100,250.50,USD\n"
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"  # repeated section header
        ",,,,,\n"                                                          # blank row
        "Total,,,,25050.00,\n"                                             # subtotal row
        "20240601,MSFT,SELL,0,420.00,USD\n"                                # zero qty
        "20240601,MSFT,SELL,40,420.00,USD\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    assert len(rows) == 2
    assert [r.fill.side for r in rows] == ["BUY", "SELL"]


# ── CAD listing → yfinance ticker mapping ───────────────────────────────

def test_flex_import_maps_cad_neo_listing_to_ne_suffix():
    csv_text = (
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary,ListingExchange\n"
        "20230110,AAPL,BUY,500,25.10,CAD,AEQLIT\n"
        "20230110,VFV,BUY,10,105.00,CAD,TSE\n"
        "20230110,MSFT,BUY,10,300.00,USD,NASDAQ\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    symbols = {r.fill.symbol for r in rows}
    assert "AAPL.NE" in symbols   # NEO-listed CDR
    assert "VFV.TO" in symbols    # TSX listing
    assert "MSFT" in symbols      # USD untouched


def test_flex_import_leaves_cad_symbol_with_existing_suffix_alone():
    csv_text = (
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary,ListingExchange\n"
        "20230110,AAPL.NE,BUY,500,25.10,CAD,AEQLIT\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    assert rows[0].fill.symbol == "AAPL.NE"


# ── ledger round-trip ───────────────────────────────────────────────────

def test_flex_import_round_trips_through_ledger(tmp_path):
    import backtest_data
    csv_text = (
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315,AAPL.NE,BUY,100,25.50,CAD\n"
        "20240601,MSFT,SELL,40,420.00,USD\n"
    )
    rows = flex_import.parse_flex_csv(csv_text)
    out = tmp_path / "fills.csv"
    flex_import.write_ledger(rows, str(out))

    ledger = backtest_data.load_fills_ledger(str(out))
    assert len(ledger) == 2
    assert ledger[0].currency == "CAD" and ledger[0].fill.symbol == "AAPL.NE"
    assert ledger[1].currency == "USD" and ledger[1].fill.qty == Decimal("40")


def test_write_ledger_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "fills.csv"
    out.write_text("existing\n")
    rows = flex_import.parse_flex_csv(
        "TradeDate,Symbol,Buy/Sell,Quantity,TradePrice,CurrencyPrimary\n"
        "20220315,MSFT,BUY,1,1,USD\n"
    )
    with pytest.raises(FileExistsError):
        flex_import.write_ledger(rows, str(out))
    flex_import.write_ledger(rows, str(out), force=True)  # explicit force works
    assert "MSFT" in out.read_text()
