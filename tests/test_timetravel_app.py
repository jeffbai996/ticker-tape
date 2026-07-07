"""TUI integration tests for time-travel mode (Pilot-driven).

Reuses the mocked-data fixtures from test_app so no network is touched, and
stubs timetravel.build_dataset with a deterministic in-memory state so the
enter → scrub → exit lifecycle is exercised end-to-end without yfinance.
"""

from datetime import date
from decimal import Decimal

import pytest

from tests.test_app import app, patched_data  # noqa: F401 (fixtures)

import timetravel
from timetravel import TimeTravelState
from backtest import Fill
from widgets.status_bar import StatusBar
from widgets.sidebar import Sidebar


D = Decimal


def _fake_state():
    fills = [
        Fill(date(2024, 1, 8), "AAPL", "BUY", D("50"), D("100")),
        Fill(date(2024, 9, 4), "AAPL", "SELL", D("20"), D("150")),
        Fill(date(2025, 1, 15), "MSFT", "BUY", D("20"), D("400")),
    ]
    bars = {
        "AAPL": {date(2024, 1, 8): D("100"), date(2024, 9, 4): D("150"),
                 date(2025, 1, 15): D("180")},
        "MSFT": {date(2025, 1, 15): D("400")},
    }
    return TimeTravelState(
        report_ccy="USD",
        fills=fills,
        bars=bars,
        watch_bars=bars,
        min_date=date(2024, 1, 8),
        max_date=date(2025, 1, 15),
    )


@pytest.mark.asyncio
async def test_enter_time_travel_sets_mode_and_banner(app, monkeypatch):  # noqa: F811
    monkeypatch.setattr(timetravel, "build_dataset", lambda syms, **k: _fake_state())
    async with app.run_test() as pilot:
        app._enter_time_travel([])
        await pilot.pause()
        # worker fetches then activates; give it a beat
        for _ in range(5):
            if app._tt_state is not None:
                break
            await pilot.pause(0.05)
        assert app._tt_state is not None
        assert app._tt_date == date(2025, 1, 15)  # opens at latest
        # StatusBar + sidebar both flipped into TT mode
        assert app.query_one(StatusBar)._timetravel_date == date(2025, 1, 15)
        assert app.query_one(Sidebar)._timetravel_date == date(2025, 1, 15)


@pytest.mark.asyncio
async def test_scrub_back_and_forward_and_clamps(app, monkeypatch):  # noqa: F811
    monkeypatch.setattr(timetravel, "build_dataset", lambda syms, **k: _fake_state())
    async with app.run_test() as pilot:
        app._enter_time_travel([])
        for _ in range(5):
            if app._tt_state is not None:
                break
            await pilot.pause(0.05)
        assert app._tt_date == date(2025, 1, 15)

        # one day back
        app._scrub(-1)
        await pilot.pause()
        assert app._tt_date == date(2025, 1, 14)

        # forward returns to max and clamps there (can't exceed max_date)
        app._scrub(1)
        app._scrub(1)
        await pilot.pause()
        assert app._tt_date == date(2025, 1, 15)  # clamped at max


@pytest.mark.asyncio
async def test_exit_time_travel_restores_live(app, monkeypatch):  # noqa: F811
    monkeypatch.setattr(timetravel, "build_dataset", lambda syms, **k: _fake_state())
    async with app.run_test() as pilot:
        app._enter_time_travel([])
        for _ in range(5):
            if app._tt_state is not None:
                break
            await pilot.pause(0.05)
        assert app._tt_state is not None

        app._exit_time_travel()
        await pilot.pause()
        assert app._tt_state is None
        assert app._tt_date is None
        assert app.query_one(StatusBar)._timetravel_date is None
        assert app.query_one(Sidebar)._timetravel_date is None


@pytest.mark.asyncio
async def test_enter_via_command_alias(app, monkeypatch):  # noqa: F811
    monkeypatch.setattr(timetravel, "build_dataset", lambda syms, **k: _fake_state())
    from widgets.command_bar import CommandSubmitted
    async with app.run_test() as pilot:
        await app.on_command_submitted(CommandSubmitted("timetravel", []))
        for _ in range(5):
            if app._tt_state is not None:
                break
            await pilot.pause(0.05)
        assert app._tt_state is not None


@pytest.mark.asyncio
async def test_invalid_date_arg_does_not_enter(app, monkeypatch):  # noqa: F811
    monkeypatch.setattr(timetravel, "build_dataset", lambda syms, **k: _fake_state())
    async with app.run_test() as pilot:
        app._enter_time_travel(["not-a-date"])
        await pilot.pause()
        assert app._tt_state is None
