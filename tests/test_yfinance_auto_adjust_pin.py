"""Pin auto_adjust=True explicitly at every yfinance download/history call site
in data.py.

yfinance's default for auto_adjust has changed across versions (pandas-reader
style vs. modern-adjusted-by-default). Charts, SMA calcs, and cross-symbol
comparisons in this app were built assuming split/dividend-adjusted closes.
Relying on the library default means a future yfinance upgrade could silently
flip the shape of every price series drawn on screen. These tests assert the
kwarg is passed explicitly at each call site — a pin, not a behavior change.
"""

import pandas as pd
import pytest
import yfinance

import data as _data


@pytest.fixture
def capture_download(monkeypatch):
    """Spy on yfinance.download; returns list of captured kwargs dicts."""
    calls = []

    def _fake(*args, **kwargs):
        calls.append(kwargs)
        symbols = args[0].split() if args and isinstance(args[0], str) else (
            kwargs.get("tickers", "") if isinstance(kwargs.get("tickers"), str)
            else " ".join(kwargs.get("tickers", []))
        ).split()
        if len(symbols) <= 1:
            return pd.DataFrame({"Close": [100.0, 101.0]})
        cols = {}
        for sym in symbols:
            cols[("Close", sym)] = [100.0, 101.0]
        return pd.DataFrame(cols)

    monkeypatch.setattr(yfinance, "download", _fake)
    return calls


@pytest.fixture
def capture_history(monkeypatch):
    """Spy on Ticker(...).history(...); returns list of captured kwargs dicts."""
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append(kwargs)
            idx = pd.date_range("2025-01-01", periods=25, freq="D")
            return pd.DataFrame({
                "Close": [100.0 + i for i in range(25)],
                "Volume": [1000] * 25,
                "High": [101.0 + i for i in range(25)],
                "Low": [99.0 + i for i in range(25)],
            }, index=idx)

        def get_earnings_dates(self, limit=20):
            return None

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    return calls


class TestDownloadCallSitesPinAutoAdjust:
    def test_bulk_prices_passes_auto_adjust_true(self, capture_download):
        _data.bulk_prices(["AAPL"])
        assert capture_download, "yf.download was not called"
        assert capture_download[0].get("auto_adjust") is True

    def test_fetch_comparison_data_passes_auto_adjust_true(self, capture_download):
        _data.fetch_comparison_data(["AAPL", "MSFT"], period="1mo")
        assert capture_download, "yf.download was not called"
        assert capture_download[-1].get("auto_adjust") is True

    def test_fetch_correlation_passes_auto_adjust_true(self, capture_download):
        _data.fetch_correlation(["AAPL", "MSFT"], period="3mo")
        assert capture_download, "yf.download was not called"
        assert capture_download[-1].get("auto_adjust") is True


class TestHistoryCallSitesPinAutoAdjust:
    @pytest.fixture(autouse=True)
    def _clear_module_caches(self):
        """fetch_technicals / _get_bench_hist memoize results in module-level
        dicts keyed by symbol — clear them so a cache hit from an earlier
        test in the suite can't skip the yfinance call this test is spying on."""
        _data._ta_cache.clear()
        _data._bench_cache.clear()

    def test_fetch_technicals_passes_auto_adjust_true(self, capture_history):
        _data.fetch_technicals("ZZZQ_TEST_TECH")
        assert capture_history, ".history() was not called"
        assert capture_history[-1].get("auto_adjust") is True

    def test_fetch_chart_data_passes_auto_adjust_true(self, capture_history):
        _data.fetch_chart_data("AAPL", period="1mo", interval="1d")
        assert capture_history, ".history() was not called"
        assert capture_history[-1].get("auto_adjust") is True

    def test_fetch_intraday_data_passes_auto_adjust_true(self, capture_history):
        _data.fetch_intraday_data("AAPL")
        assert capture_history, ".history() was not called"
        assert capture_history[-1].get("auto_adjust") is True

    def test_get_bench_hist_passes_auto_adjust_true(self, capture_history):
        _data._get_bench_hist("ZZZQ_TEST_BENCH", period="1mo", interval="1d")
        assert capture_history, ".history() was not called"
        assert capture_history[-1].get("auto_adjust") is True


class TestFetchEarningsImpactHistoryCallSites:
    """fetch_earnings_impact has two .history() call sites (primary symbol +
    peer reaction loop) that only execute when get_earnings_dates() returns
    rows. Stub that path explicitly to exercise both."""

    def test_both_history_call_sites_pin_auto_adjust(self, monkeypatch):
        calls = []

        class FakeTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def get_earnings_dates(self, limit=20):
                import datetime as dt
                idx = pd.DatetimeIndex([pd.Timestamp("2025-01-10", tz="UTC")])
                return pd.DataFrame(
                    {"Reported EPS": [1.5], "EPS Estimate": [1.4], "Surprise(%)": [7.1]},
                    index=idx,
                )

            def history(self, **kwargs):
                calls.append(kwargs)
                idx = pd.date_range("2025-01-05", periods=10, freq="D")
                return pd.DataFrame({"Close": [100.0 + i for i in range(10)]}, index=idx)

        monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
        _data.fetch_earnings_impact("AAPL")
        assert calls, ".history() was not called"
        assert all(c.get("auto_adjust") is True for c in calls)
