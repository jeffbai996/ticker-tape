"""Tests for data.py — alert CRUD, watchlist CRUD, pure logic."""

import json
import os

import pytest

from data import (
    load_alerts, save_alerts, add_alert, remove_alert, evaluate_alerts,
    load_watchlist, add_to_watchlist, remove_from_watchlist,
    get_all_names, _load_watchlist_data, _save_watchlist_data,
    _atomic_write_json, market_state,
)
from config import NAMES


# ── Atomic write ───────────────────────────────────────────

class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = str(tmp_path / "test.json")
        _atomic_write_json(path, {"key": "value"})
        with open(path) as f:
            assert json.load(f) == {"key": "value"}

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "test.json")
        _atomic_write_json(path, {"old": True})
        _atomic_write_json(path, {"new": True})
        with open(path) as f:
            assert json.load(f) == {"new": True}

    def test_no_temp_files_left(self, tmp_path):
        path = str(tmp_path / "test.json")
        _atomic_write_json(path, [1, 2, 3])
        files = os.listdir(tmp_path)
        assert files == ["test.json"]


# ── Alert CRUD ─────────────────────────────────────────────

class TestAlertCRUD:
    def test_load_empty_when_no_file(self, alerts_file):
        assert load_alerts() == []

    def test_add_alert_creates_file(self, alerts_file):
        alert = add_alert("NVDA", ">", 150.0)
        assert alert["id"] == 1
        assert alert["symbol"] == "NVDA"
        assert alert["operator"] == ">"
        assert alert["value"] == 150.0
        assert os.path.exists(alerts_file)

    def test_add_multiple_alerts_increments_id(self, alerts_file):
        a1 = add_alert("NVDA", ">", 150.0)
        a2 = add_alert("MU", "<", 80.0)
        assert a1["id"] == 1
        assert a2["id"] == 2

    def test_load_returns_saved_alerts(self, alerts_file):
        add_alert("NVDA", ">", 150.0)
        add_alert("MU", "<", 80.0)
        alerts = load_alerts()
        assert len(alerts) == 2
        assert alerts[0]["symbol"] == "NVDA"
        assert alerts[1]["symbol"] == "MU"

    def test_remove_alert_by_id(self, alerts_file):
        add_alert("NVDA", ">", 150.0)
        add_alert("MU", "<", 80.0)
        assert remove_alert(1) is True
        alerts = load_alerts()
        assert len(alerts) == 1
        assert alerts[0]["symbol"] == "MU"

    def test_remove_nonexistent_returns_false(self, alerts_file):
        add_alert("NVDA", ">", 150.0)
        assert remove_alert(999) is False

    def test_save_and_load_roundtrip(self, alerts_file):
        data = [{"id": 1, "symbol": "GOOG", "operator": "<", "value": 100.0, "created": "now"}]
        save_alerts(data)
        assert load_alerts() == data


class TestEvaluateAlerts:
    def test_greater_than_triggered(self, alerts_file):
        add_alert("NVDA", ">", 100.0)
        quotes = [{"symbol": "NVDA", "price": 150.0}]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 1
        assert triggered[0]["current_price"] == 150.0

    def test_greater_than_not_triggered(self, alerts_file):
        add_alert("NVDA", ">", 200.0)
        quotes = [{"symbol": "NVDA", "price": 150.0}]
        assert evaluate_alerts(quotes) == []

    def test_less_than_triggered(self, alerts_file):
        add_alert("MU", "<", 100.0)
        quotes = [{"symbol": "MU", "price": 80.0}]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 1

    def test_less_than_not_triggered(self, alerts_file):
        add_alert("MU", "<", 50.0)
        quotes = [{"symbol": "MU", "price": 80.0}]
        assert evaluate_alerts(quotes) == []

    def test_missing_symbol_skipped(self, alerts_file):
        add_alert("NVDA", ">", 100.0)
        quotes = [{"symbol": "MU", "price": 80.0}]
        assert evaluate_alerts(quotes) == []

    def test_zero_price_not_skipped(self, alerts_file):
        """Price 0.0 should still be evaluated (not dropped by falsy check)."""
        add_alert("BAD", "<", 5.0)
        quotes = [{"symbol": "BAD", "price": 0.0}]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 1

    def test_no_alerts_returns_empty(self, alerts_file):
        quotes = [{"symbol": "NVDA", "price": 150.0}]
        assert evaluate_alerts(quotes) == []

    def test_multiple_alerts_multiple_triggers(self, alerts_file):
        add_alert("NVDA", ">", 100.0)
        add_alert("MU", "<", 90.0)
        add_alert("GOOG", ">", 200.0)
        quotes = [
            {"symbol": "NVDA", "price": 150.0},
            {"symbol": "MU", "price": 80.0},
            {"symbol": "GOOG", "price": 180.0},
        ]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 2
        syms = {t["symbol"] for t in triggered}
        assert syms == {"NVDA", "MU"}


# ── Watchlist CRUD ─────────────────────────────────────────

class TestWatchlistCRUD:
    def test_load_empty_when_no_file(self, watchlist_file):
        assert load_watchlist() == []

    def test_add_to_watchlist(self, watchlist_file, monkeypatch):
        # Mock yfinance to avoid network calls
        class FakeInfo:
            info = {"shortName": "Test Corp"}
        class FakeTicker:
            def __init__(self, sym): pass
            @property
            def info(self):
                return {"shortName": "Test Corp"}

        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", FakeTicker)

        assert add_to_watchlist("TEST") is True
        assert "TEST" in load_watchlist()

    def test_add_duplicate_returns_false(self, watchlist_file, monkeypatch):
        class FakeTicker:
            def __init__(self, sym): pass
            @property
            def info(self):
                return {"shortName": "Test Corp"}

        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", FakeTicker)

        add_to_watchlist("TEST")
        assert add_to_watchlist("TEST") is False

    def test_add_portfolio_symbol_returns_false(self, watchlist_file):
        """Can't add a symbol that's already in the portfolio."""
        assert add_to_watchlist("NVDA") is False

    def test_remove_from_watchlist(self, watchlist_file, monkeypatch):
        class FakeTicker:
            def __init__(self, sym): pass
            @property
            def info(self):
                return {"shortName": "Test Corp"}

        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", FakeTicker)

        add_to_watchlist("TEST")
        assert remove_from_watchlist("TEST") is True
        assert "TEST" not in load_watchlist()

    def test_remove_nonexistent_returns_false(self, watchlist_file):
        assert remove_from_watchlist("NOPE") is False

    def test_get_all_names_includes_portfolio(self, watchlist_file):
        names = get_all_names()
        for sym, name in NAMES.items():
            assert names[sym] == name

    def test_get_all_names_includes_watchlist(self, watchlist_file, monkeypatch):
        class FakeTicker:
            def __init__(self, sym): pass
            @property
            def info(self):
                return {"shortName": "Watchlist Corp"}

        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", FakeTicker)

        add_to_watchlist("WL1")
        names = get_all_names()
        assert names["WL1"] == "Watchlist Corp"


# ── Market state ───────────────────────────────────────────

class TestMarketState:
    def test_returns_valid_state(self):
        state = market_state()
        assert state in ("pre", "open", "post", "closed")
