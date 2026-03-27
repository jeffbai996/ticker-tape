"""Tests for new screen formatters — options, correlation, impact upgrade."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screens.options import format_options
from screens.correlation import format_correlation
from screens.impact import format_impact


# ── Options chain screen ──────────────────────────────────

class TestFormatOptions:
    def test_none_returns_error_with_symbol(self):
        result = format_options(None, "AAPL")
        assert "AAPL" in result
        assert "ff3232" in result

    def test_basic_calls_puts_render(self):
        data = {
            "symbol": "AAPL",
            "current_price": 130.0,
            "expirations": ["2026-04-03", "2026-04-10"],
            "selected_expiration": "2026-04-03",
            "calls": [
                {"strike": 125.0, "bid": 6.0, "ask": 6.5, "last": 6.2,
                 "volume": 1000, "open_interest": 5000, "iv": 0.45,
                 "itm": True, "moneyness": -3.85},
                {"strike": 130.0, "bid": 3.0, "ask": 3.5, "last": 3.2,
                 "volume": 2000, "open_interest": 8000, "iv": 0.50,
                 "itm": False, "moneyness": 0.0},
            ],
            "puts": [
                {"strike": 125.0, "bid": 1.5, "ask": 2.0, "last": 1.7,
                 "volume": 500, "open_interest": 3000, "iv": 0.42,
                 "itm": False, "moneyness": -3.85},
            ],
        }
        result = format_options(data, "AAPL")
        assert "130.00" in result
        assert "125.00" in result
        assert "ATM" in result  # 130 strike with 0% moneyness
        assert "HEDGE" in result  # OTM put

    def test_empty_legs(self):
        data = {
            "symbol": "TEST",
            "current_price": 100.0,
            "expirations": ["2026-04-03"],
            "selected_expiration": "2026-04-03",
            "calls": [],
            "puts": [],
        }
        result = format_options(data, "TEST")
        assert "TEST" in result

    def test_zero_bid_ask_shows_dash(self):
        data = {
            "symbol": "TEST",
            "current_price": 100.0,
            "expirations": ["2026-04-03"],
            "selected_expiration": "2026-04-03",
            "calls": [
                {"strike": 100.0, "bid": 0.0, "ask": 0.0, "last": 0.0,
                 "volume": 0, "open_interest": 0, "iv": None,
                 "itm": False, "moneyness": 0.0},
            ],
            "puts": [],
        }
        result = format_options(data, "TEST")
        assert "—" in result

    def test_expiration_hint_shown(self):
        data = {
            "symbol": "AAPL",
            "current_price": 250.0,
            "expirations": ["2026-04-03", "2026-04-10", "2026-04-17"],
            "selected_expiration": "2026-04-03",
            "calls": [],
            "puts": [],
        }
        result = format_options(data, "AAPL")
        assert "2026-04-10" in result  # other expirations shown


# ── Correlation matrix screen ─────────────────────────────

class TestFormatCorrelation:
    def test_none_returns_message(self):
        result = format_correlation(None)
        assert result  # non-empty string

    def test_basic_matrix_render(self):
        data = {
            "symbols": ["AAPL", "AAPL", "MSFT"],
            "matrix": [
                [1.0, 0.85, 0.72],
                [0.85, 1.0, 0.68],
                [0.72, 0.68, 1.0],
            ],
            "period": "3mo",
        }
        result = format_correlation(data)
        assert "AAPL" in result
        assert "AAPL" in result
        assert "MSFT" in result
        assert "1.00" in result  # diagonal
        assert "0.85" in result

    def test_truncation_at_12(self):
        syms = [f"S{i:02d}" for i in range(15)]
        matrix = [[1.0 if i == j else 0.5 for j in range(15)] for i in range(15)]
        data = {"symbols": syms, "matrix": matrix, "period": "3mo"}
        result = format_correlation(data)
        assert "S00" in result
        assert "S11" in result  # 12th symbol (index 11)
        assert "S12" not in result  # truncated
        assert "15" in result  # total count mentioned

    def test_avg_pairwise_calculation(self):
        data = {
            "symbols": ["A", "B"],
            "matrix": [[1.0, 0.60], [0.60, 1.0]],
            "period": "3mo",
        }
        result = format_correlation(data)
        assert "0.60" in result  # avg pairwise = 0.60


# ── Impact screen upgrade ─────────────────────────────────

class TestFormatImpact:
    def test_none_returns_error(self):
        result = format_impact(None, "AAPL")
        assert "AAPL" in result
        assert "ff3232" in result

    def test_backward_compat_no_summary(self):
        """Old data shape without summary key still renders events."""
        impact = {
            "symbol": "AAPL",
            "events": [
                {"date": "2025-07-15", "eps_est": 1.8, "eps_actual": 2.0,
                 "surprise_pct": 11.0, "price_move": 3.5, "peers": []},
            ],
        }
        result = format_impact(impact, "AAPL")
        assert "2025-07-15" in result
        assert "$2.00" in result

    def test_beat_indicator(self):
        impact = {
            "symbol": "AAPL",
            "events": [
                {"date": "2025-07-15", "eps_est": 1.8, "eps_actual": 2.0,
                 "surprise_pct": 11.0, "price_move": 3.5, "peers": []},
            ],
            "summary": {"beat_streak": 1, "beat_rate": 1.0, "beats": 1,
                         "total": 1, "avg_surprise": 11.0, "avg_move": 3.5},
        }
        result = format_impact(impact, "AAPL")
        assert "BEAT" in result

    def test_miss_indicator(self):
        impact = {
            "symbol": "AAPL",
            "events": [
                {"date": "2025-07-15", "eps_est": 2.0, "eps_actual": 1.5,
                 "surprise_pct": -25.0, "price_move": -5.0, "peers": []},
            ],
            "summary": {"beat_streak": 0, "beat_rate": 0.0, "beats": 0,
                         "total": 1, "avg_surprise": -25.0, "avg_move": -5.0},
        }
        result = format_impact(impact, "AAPL")
        assert "MISS" in result

    def test_summary_header_content(self):
        impact = {
            "symbol": "AAPL",
            "events": [
                {"date": "2025-07-15", "eps_est": 1.0, "eps_actual": 1.2,
                 "surprise_pct": 20.0, "price_move": 5.0, "peers": []},
                {"date": "2025-04-15", "eps_est": 1.0, "eps_actual": 0.8,
                 "surprise_pct": -20.0, "price_move": -3.0, "peers": []},
            ],
            "summary": {"beat_streak": 1, "beat_rate": 0.5, "beats": 1,
                         "total": 2, "avg_surprise": 0.0, "avg_move": 1.0},
        }
        result = format_impact(impact, "AAPL")
        assert "1/2" in result  # beat rate
        assert "50.0%" in result  # beat rate percentage
        assert "Streak" in result

    def test_beat_rate_one_decimal(self):
        """Beat rate should show 1 decimal, not rounded to integer."""
        impact = {
            "symbol": "X",
            "events": [
                {"date": "2025-07-15", "eps_est": 1.0, "eps_actual": 1.1,
                 "surprise_pct": 10.0, "price_move": 2.0, "peers": []},
                {"date": "2025-04-15", "eps_est": 1.0, "eps_actual": 1.1,
                 "surprise_pct": 10.0, "price_move": 2.0, "peers": []},
                {"date": "2025-01-15", "eps_est": 1.0, "eps_actual": 0.9,
                 "surprise_pct": -10.0, "price_move": -2.0, "peers": []},
            ],
            "summary": {"beat_streak": 2, "beat_rate": 2 / 3, "beats": 2,
                         "total": 3, "avg_surprise": 3.33, "avg_move": 0.67},
        }
        result = format_impact(impact, "X")
        assert "66.7%" in result  # 1 decimal, not 67%
