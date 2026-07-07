"""Tests for the time-travel screen formatter (screens/timetravel.py)."""

from datetime import date
from decimal import Decimal

from screens.timetravel import format_asof_book
from timetravel import AsOfBook, PositionSnapshot


D = Decimal


def _book():
    positions = [
        PositionSnapshot("AAPL", D("30"), D("100"), D("180"), D("5400"), D("2400")),
        PositionSnapshot("MSFT", D("20"), D("400"), D("500"), D("10000"), D("2000")),
    ]
    return AsOfBook(
        as_of=date(2025, 1, 15),
        positions=positions,
        realized_pnl=D("1000"),
        cost_basis=D("11000"),
        market_value=D("15400"),
        unrealized_pnl=D("4400"),
        warnings=[],
    )


def test_format_shows_positions_and_totals():
    out = format_asof_book(_book(), ccy="USD")
    assert "2025-01-15" in out
    assert "AAPL" in out and "MSFT" in out
    assert "+2,400" in out          # AAPL unrealized
    assert "+4,400" in out          # total unrealized
    assert "+1,000" in out          # realized
    assert "USD" in out


def test_format_empty_book_shows_no_positions_message():
    book = AsOfBook(
        as_of=date(2023, 12, 31),
        positions=[],
        realized_pnl=D("0"),
        cost_basis=D("0"),
        market_value=None,
        unrealized_pnl=None,
    )
    out = format_asof_book(book)
    assert "2023-12-31" in out
    # total P&L is N/A when unrealized is unknown
    assert "N/A" in out


def test_format_surfaces_warnings():
    book = AsOfBook(
        as_of=date(2026, 1, 2),
        positions=[],
        realized_pnl=D("0"),
        cost_basis=D("0"),
        market_value=None,
        unrealized_pnl=None,
        warnings=["SELL 100 AAPL on 2026-01-02: only 0 held — 100 unmatched, skipped"],
    )
    out = format_asof_book(book)
    assert "unmatched" in out


def test_format_unknown_close_renders_dash_not_zero():
    book = AsOfBook(
        as_of=date(2024, 6, 1),
        positions=[PositionSnapshot("AAPL", D("10"), D("100"), None, None, None)],
        realized_pnl=D("0"),
        cost_basis=D("1000"),
        market_value=None,
        unrealized_pnl=None,
    )
    out = format_asof_book(book)
    assert "AAPL" in out
    assert "—" in out  # dash for the missing close/mv/unreal
