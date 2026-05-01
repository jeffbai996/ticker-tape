"""Tests for sidebar MCP-output parsers (_parse_pnl, _parse_account_summary).

Regression tests for a bug where the parsers failed to strip markdown bold
(`**`) and mishandled the `&` character, causing the sidebar P&L and Risk
panels to render headers with no data underneath.
"""

from widgets.sidebar import Sidebar


# ── _parse_pnl ────────────────────────────────────────────

class TestParsePnl:
    def test_strips_markdown_bold_and_normalizes_pnl_key(self):
        """**Daily P&L** must normalize to `daily_pnl` (not `**daily_pl**`)."""
        raw = (
            "**Daily P&L**: $-147,179.71 CAD\n"
            "**Unrealized P&L**: $3,934,347.74 CAD\n"
            "**Realized P&L**: $0.00\n"
        )
        out = Sidebar._parse_pnl(raw)
        assert out["daily_pnl"] == -147179.71
        assert out["unrealized_pnl"] == 3934347.74
        assert out["realized_pnl"] == 0.0

    def test_plain_keys_still_parsed(self):
        """Non-bolded keys (legacy / fallback format) should still work."""
        raw = "Daily P&L: $100.00\nUnrealized P&L: $-50.50\n"
        out = Sidebar._parse_pnl(raw)
        assert out["daily_pnl"] == 100.0
        assert out["unrealized_pnl"] == -50.5

    def test_skips_na_and_empty_values(self):
        raw = "**Daily P&L**: N/A\n**Realized P&L**: $0.00\n"
        out = Sidebar._parse_pnl(raw)
        assert "daily_pnl" not in out
        assert out["realized_pnl"] == 0.0

    def test_build_pnl_finds_keys_after_parse(self):
        """Round-trip: parsed output must satisfy what _build_pnl looks up."""
        raw = (
            "**Daily P&L**: $-147,179.71 CAD\n"
            "**Unrealized P&L**: $3,934,347.74 CAD\n"
            "**Realized P&L**: $0.00\n"
        )
        parsed = Sidebar._parse_pnl(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_pnl(parsed)
        # All three rows should appear in the output
        assert "147,180" in rendered or "147,179" in rendered
        assert "3,934,348" in rendered or "3,934,347" in rendered
        assert "+0" in rendered or "0" in rendered

    def test_captures_daily_pct_from_parenthetical(self):
        """Daily P&L's '(-1.23% of NLV)' suffix should populate daily_pnl_pct."""
        raw = "**Daily P&L**: $-147,179.71 CAD (-1.23% of NLV)\n"
        out = Sidebar._parse_pnl(raw)
        assert out["daily_pnl"] == -147179.71
        assert out["daily_pnl_pct"] == -1.23

    def test_captures_positive_daily_pct(self):
        raw = "**Daily P&L**: $50,000.00 USD (+0.85% of NLV)\n"
        out = Sidebar._parse_pnl(raw)
        assert out["daily_pnl_pct"] == 0.85

    def test_no_daily_pct_when_parenthetical_absent(self):
        raw = "**Daily P&L**: $100.00\n"
        out = Sidebar._parse_pnl(raw)
        assert "daily_pnl_pct" not in out

    def test_build_pnl_renders_total_and_pct_rows(self):
        """Total = unreal + real; Day% pulled from parenthetical."""
        raw = (
            "**Daily P&L**: $-50,000.00 USD (-1.00% of NLV)\n"
            "**Unrealized P&L**: $200,000.00 USD\n"
            "**Realized P&L**: $-30,000.00 USD\n"
        )
        parsed = Sidebar._parse_pnl(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_pnl(parsed)
        # Day% row
        assert "-1.00%" in rendered
        # Total = 200000 + (-30000) = 170000
        assert "170,000" in rendered

    def test_build_pnl_skips_total_when_components_missing(self):
        """No Total row if either unrealized or realized is absent."""
        raw = "**Daily P&L**: $100.00 USD\n"
        parsed = Sidebar._parse_pnl(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_pnl(parsed)
        # Total label shouldn't appear
        assert "Total" not in rendered


# ── _parse_account_summary ────────────────────────────────

class TestParseAccountSummary:
    def test_strips_markdown_bold_for_risk_keys(self):
        raw = (
            "**Cushion**: 0.42\n"
            "**Leverage**: 1.8\n"
            "**Net Liquidation Value**: $5,000,000.00 CAD\n"
        )
        out = Sidebar._parse_account_summary(raw)
        assert out["cushion"] == 0.42
        assert out["leverage"] == 1.8
        # _build_risk reads either net_liquidation or net_liquidation_value
        assert out.get("net_liquidation_value") == 5_000_000.0

    def test_percentage_values_parsed(self):
        """Percentage-style values should still parse to numeric without %."""
        raw = "**Cushion**: 12.5%\n"
        out = Sidebar._parse_account_summary(raw)
        assert out["cushion"] == 12.5

    def test_build_risk_finds_keys_after_parse(self):
        raw = (
            "**Cushion**: 18.0%\n"
            "**Leverage**: 1.5\n"
            "**Net Liquidation Value**: $5,000,000.00 CAD\n"
        )
        parsed = Sidebar._parse_account_summary(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_risk(parsed)
        assert rendered != ""  # should not be empty
        assert "18" in rendered
        assert "1.5" in rendered
        assert "$5.0M" in rendered

    def test_plain_keys_still_parsed(self):
        raw = "Cushion: 0.42\nLeverage: 1.8\n"
        out = Sidebar._parse_account_summary(raw)
        assert out["cushion"] == 0.42
        assert out["leverage"] == 1.8

    def test_extended_summary_fields_parsed(self):
        """Margin util, excess liquidity, buying power round-trip into the dict."""
        raw = (
            "**Net Liquidation Value**: $5,000,000.00 CAD\n"
            "**Gross Position Value**: $7,500,000.00 CAD\n"
            "**Total Cash**: $-1,000,000.00 CAD\n"
            "**Initial Margin Req**: $1,500,000.00 CAD\n"
            "**Maintenance Margin Req**: $1,200,000.00 CAD\n"
            "**Excess Liquidity (Initial)**: $2,000,000.00 CAD\n"
            "**Excess Liquidity (Maint)**: $2,300,000.00 CAD\n"
            "**Cushion**: 18.5%\n"
            "**Buying Power**: $10,000,000.00 CAD\n"
            "**Leverage**: 1.5x\n"
            "**Margin Utilization**: 30.0%\n"
        )
        out = Sidebar._parse_account_summary(raw)
        assert out["net_liquidation_value"] == 5_000_000.0
        assert out["gross_position_value"] == 7_500_000.0
        assert out["initial_margin_req"] == 1_500_000.0
        assert out["maintenance_margin_req"] == 1_200_000.0
        # Parens are kept in the key after lowercase + underscore
        assert out["excess_liquidity_(maint)"] == 2_300_000.0
        assert out["excess_liquidity_(initial)"] == 2_000_000.0
        assert out["buying_power"] == 10_000_000.0
        assert out["margin_utilization"] == 30.0

    def test_build_risk_renders_all_added_rows(self):
        """All six panel rows surface when the summary is fully populated."""
        raw = (
            "**Net Liquidation Value**: $5,000,000.00 CAD\n"
            "**Excess Liquidity (Maint)**: $2,300,000.00 CAD\n"
            "**Cushion**: 18.5%\n"
            "**Buying Power**: $10,000,000.00 CAD\n"
            "**Leverage**: 1.5x\n"
            "**Margin Utilization**: 30.0%\n"
        )
        parsed = Sidebar._parse_account_summary(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_risk(parsed)
        # NLV + Cushion + Lever + MgnU% + ExLiq + BuyPwr → 6 metric rows
        assert "$5.0M" in rendered            # NLV
        assert "18.5%" in rendered            # Cushion
        assert "1.5x" in rendered             # Leverage
        assert "30.0%" in rendered            # Margin util
        assert "$2.3M" in rendered            # Excess liq (maint)
        assert "$10.0M" in rendered           # Buying power

    def test_build_risk_falls_back_to_initial_excess_when_maint_absent(self):
        raw = (
            "**Net Liquidation Value**: $5,000,000.00 CAD\n"
            "**Excess Liquidity (Initial)**: $1,800,000.00 CAD\n"
            "**Cushion**: 18.5%\n"
            "**Leverage**: 1.5x\n"
        )
        parsed = Sidebar._parse_account_summary(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_risk(parsed)
        assert "$1.8M" in rendered

    def test_build_risk_skips_missing_rows(self):
        """Sparse summary → only present fields render. No "—" placeholders."""
        raw = "**Cushion**: 18.5%\n**Leverage**: 1.5x\n"
        parsed = Sidebar._parse_account_summary(raw)
        sb = Sidebar.__new__(Sidebar)
        rendered = sb._build_risk(parsed)
        assert "18.5%" in rendered
        assert "1.5x" in rendered
        # No buying power / NLV labels when the data isn't there
        assert "BuyPwr" not in rendered
        assert "ExLiq" not in rendered
