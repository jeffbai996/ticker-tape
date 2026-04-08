"""Tests for commodities screen formatter."""

import pytest

from screens.commodities import _fmt_price, format_commodities


class TestFmtPrice:
    def test_large_price_no_decimals(self):
        assert _fmt_price(84_000) == "84,000"

    def test_mid_range_two_dp(self):
        assert _fmt_price(500.5) == "500.50"

    def test_below_100_two_dp(self):
        assert _fmt_price(75.5) == "75.50"

    def test_fx_small_four_dp(self):
        assert _fmt_price(1.0850) == "1.0850"

    def test_boundary_10000(self):
        assert _fmt_price(10_000) == "10,000"

    def test_boundary_100(self):
        assert _fmt_price(100.0) == "100.00"


class TestFormatCommodities:
    def test_empty_returns_no_data(self):
        result = format_commodities({})
        assert "No commodity" in result
        assert "dim" in result

    def test_group_header_present(self):
        data = {
            "Energy": [
                {"symbol": "CL=F", "name": "WTI Crude", "price": 75.5, "pct": 1.2, "unit": "$/bbl", "stale": False}
            ]
        }
        result = format_commodities(data)
        assert "Energy" in result
        assert "CL=F" in result

    def test_positive_arrow_green(self):
        data = {
            "Energy": [
                {"symbol": "CL=F", "name": "WTI Crude", "price": 75.5, "pct": 1.2, "unit": "$/bbl", "stale": False}
            ]
        }
        result = format_commodities(data)
        assert "▲" in result
        assert "green" in result

    def test_negative_arrow_red(self):
        data = {
            "Energy": [
                {"symbol": "NG=F", "name": "Nat Gas", "price": 2.75, "pct": -0.5, "unit": "$/MMBtu", "stale": False}
            ]
        }
        result = format_commodities(data)
        assert "▼" in result
        assert "#ff3232" in result

    def test_stale_shows_na(self):
        data = {
            "Metals": [
                {"symbol": "GC=F", "name": "Gold", "price": 0, "pct": 0, "unit": "$/oz", "stale": True}
            ]
        }
        result = format_commodities(data)
        assert "N/A" in result

    def test_unit_shown(self):
        data = {
            "Grains": [
                {"symbol": "ZW=F", "name": "Wheat", "price": 560.0, "pct": 0.3, "unit": "¢/bu", "stale": False}
            ]
        }
        result = format_commodities(data)
        assert "¢/bu" in result
