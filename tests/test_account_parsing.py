"""Account-summary / daily-PnL parsing inside assemble_briefing().

The IBKR MCP tool (ibkr_get_account_summary / ibkr_get_account_pnl) emits
Markdown lines like:

    **Net Liquidation Value**: $1,234.56 CAD
    **Daily P&L**: +$1,234.56 CAD (+0.12% of NLV)

for a CAD-base account, or plain USD for a USD-base account. The parser in
data.assemble_briefing() must not assume USD, must not grab the first
whitespace-split token blindly (which breaks on "$1,234.56 CAD" ordering
variants and on a leading currency code like "CAD 1,234.56"), and must
surface the parsed currency in result["portfolio"]["currency"].
"""

import pytest

import data as _data


def _stub_briefing_sources(monkeypatch):
    monkeypatch.setattr(_data, "get_all_symbols", lambda: [])
    monkeypatch.setattr(_data, "fetch_quotes", lambda syms: ([], "x"))
    monkeypatch.setattr(_data, "fetch_earnings", lambda syms: [])
    monkeypatch.setattr(_data, "fetch_news", lambda sym, count=3: [])
    monkeypatch.setattr(_data, "fetch_sector_performance", lambda: [])
    monkeypatch.setattr("db.get_nlv_peak", lambda days=90: None)


def _fake_ibkr(monkeypatch, account_text, pnl_text):
    calls = []

    def _fake(tool_name, arguments=None, url=None, account=""):
        calls.append(tool_name)
        if tool_name == "ibkr_get_account_summary":
            return account_text
        if tool_name == "ibkr_get_account_pnl":
            return pnl_text
        return None

    monkeypatch.setattr("ibkr_client.call_ibkr_tool", _fake)
    return calls


class TestCadBaseAccountParsing:
    """Real ibkr-mcp Markdown output for a CAD-base account (fmt_price /
    fmt_pnl always render "$<amount> <CCY>", currency trailing)."""

    def test_nlv_parses_with_cad_currency_suffix(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "# Account Summary: U12345678\n\n"
            "**Net Liquidation Value**: $500,000.00 CAD\n"
            "**Cushion**: 45.00%\n"
            "**Leverage**: 1.80x\n",
            "**Daily P&L**: +$1,234.56 CAD (+0.25% of NLV)\n",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["nlv"] == pytest.approx(500000.00)
        assert result["portfolio"]["currency"] == "CAD"

    def test_daily_pnl_parses_with_cad_currency(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "**Net Liquidation Value**: $500,000.00 CAD\n",
            "**Daily P&L**: +$1,234.56 CAD (+0.25% of NLV)\n",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["daily_pnl"] == pytest.approx(1234.56)

    def test_negative_daily_pnl_sign_preserved(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "**Net Liquidation Value**: $500,000.00 CAD\n",
            "**Daily P&L**: -$789.10 CAD (-0.15% of NLV)\n",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["daily_pnl"] == pytest.approx(-789.10)


class TestCurrencyLeadingFormat:
    """Some formats put the currency code before the number (e.g. "CAD 12,345.67")
    rather than trailing — the numeric-token regex must find the number anywhere
    in the value, not just the first split() token."""

    def test_currency_prefixed_value_still_parses_numeric(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "**Net Liquidation Value**: CAD 12,345.67\n",
            "",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["nlv"] == pytest.approx(12345.67)
        assert result["portfolio"]["currency"] == "CAD"


class TestUsdBaseAccountUnaffected:
    """USD-base accounts (the common case) must keep working exactly as before."""

    def test_usd_nlv_and_currency(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "**Net Liquidation Value**: $500,000.00 USD\n",
            "**Daily P&L**: +$1,234.56 USD (+0.25% of NLV)\n",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["nlv"] == pytest.approx(500000.00)
        assert result["portfolio"]["currency"] == "USD"
        assert result["portfolio"]["daily_pnl"] == pytest.approx(1234.56)


class TestNoCurrencyPresent:
    """Legacy/plain formats without any currency code must still parse the
    number and record currency as None (never crash, never fabricate a code)."""

    def test_missing_currency_is_none(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "Net Liquidation: $500,000\n",
            "Daily P&L: $1,234 (+0.25% of NLV)\n",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["nlv"] == pytest.approx(500000)
        assert result["portfolio"]["currency"] is None


class TestAsteriskKeyNormalization:
    """Real ibkr-mcp output wraps field labels in markdown bold (**Key**).
    The key-normalization step must strip that so the existing
    parsed.get("net_liquidation") / .get("cushion") / .get("leverage")
    lookups actually hit — otherwise portfolio health silently stays None
    for every real (non-test-stub) call to the account-summary tool."""

    def test_bold_wrapped_keys_still_resolve_cushion_and_leverage(self, monkeypatch):
        _stub_briefing_sources(monkeypatch)
        _fake_ibkr(
            monkeypatch,
            "**Net Liquidation Value**: $500,000.00 USD\n"
            "**Cushion**: 45.00%\n"
            "**Leverage**: 1.80x\n",
            "",
        )
        result = _data.assemble_briefing()
        assert result["portfolio"]["cushion"] == pytest.approx(45.00)
        assert result["portfolio"]["leverage"] == pytest.approx(1.80)
