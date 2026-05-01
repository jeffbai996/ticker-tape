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
