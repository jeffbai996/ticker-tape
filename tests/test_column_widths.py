"""Column-width invariants for v2.5.5 screens.

Guards against markup-drift regressions: every data row must produce the
same visible column positions, regardless of sign/color/magnitude. Strips
Rich markup tags before counting visible cells so tag-length can never
masquerade as padding.
"""

import re


_MARKUP_RE = re.compile(r"\[/?[^\[\]]*?\]")


def _visible(s: str) -> str:
    """Strip Rich markup tags so we can measure true cell width."""
    return _MARKUP_RE.sub("", s)


def _data_rows(result: str, min_cols: int = 3) -> list[str]:
    """Return non-empty, non-header, non-divider rows from a formatter output."""
    rows = []
    for raw in result.splitlines():
        stripped = _visible(raw).rstrip()
        if not stripped.strip():
            continue
        if set(stripped.strip()) <= set("─ "):
            continue
        rows.append(stripped)
    return rows


# ── format_surprises ───────────────────────────────────────────────

class TestSurprisesColumns:
    def _sample(self) -> dict:
        return {
            "symbols": {
                "NVDA": {
                    "events": [],
                    "summary": {
                        "total": 4, "beats": 4, "beat_rate": 1.0,
                        "beat_streak": 4, "avg_surprise": 8.5, "avg_move": 3.1,
                        "last_eps": 0.92, "last_surprise": 3.37, "last_move": 2.1,
                    },
                },
                "AVGO": {
                    "events": [],
                    "summary": {
                        "total": 4, "beats": 3, "beat_rate": 0.75,
                        "beat_streak": 2, "avg_surprise": 5.0, "avg_move": 2.8,
                        "last_eps": 12.20, "last_surprise": -2.1, "last_move": -1.4,
                    },
                },
                "MU": {
                    "events": [],
                    "summary": {
                        "total": 4, "beats": 0, "beat_rate": 0.0,
                        "beat_streak": 0, "avg_surprise": -3.2, "avg_move": -4.5,
                        "last_eps": -0.09, "last_surprise": -75.0, "last_move": -12.5,
                    },
                },
            },
            "watchlist_summary": {
                "total_beats": 7, "total_total": 12,
                "avg_beat_rate": 0.58, "avg_surprise": 3.4, "avg_move": -0.2,
            },
        }

    def test_data_rows_identical_width(self):
        """Every per-symbol row must be the same visible width."""
        from screens.surprises import format_surprises
        result = format_surprises(self._sample())
        rows = _data_rows(result)
        # Keep only rows that contain a symbol we put in the sample.
        sym_rows = [r for r in rows if any(s in r for s in ("NVDA", "AVGO", "MU"))]
        assert len(sym_rows) == 3, f"expected 3 symbol rows, got {len(sym_rows)}: {sym_rows}"
        widths = {len(r) for r in sym_rows}
        assert len(widths) == 1, f"row widths diverge: {widths}\n" + "\n".join(sym_rows)

    def test_sign_column_isolated_from_digits(self):
        """`$` / `-$` lives in its own column; the digit column never shifts."""
        from screens.surprises import format_surprises, _W_DSIGN, _W_EPS
        result = format_surprises(self._sample())
        rows = _data_rows(result)
        # Positions of the first digit after the symbol column should be equal
        # across positive and negative EPS rows. NVDA has +$0.92, MU has -$0.09.
        nvda = next(r for r in rows if "NVDA" in r)
        mu   = next(r for r in rows if "MU"   in r)
        # Find the EPS number column by locating "0.92" and "0.09" inside the
        # exactly-_W_EPS-wide slot right after the _W_DSIGN slot.
        assert "0.92" in nvda
        assert "0.09" in mu
        # The digits must start at the same column in both rows.
        assert nvda.index("0.92") == mu.index("0.09"), (
            f"EPS digit column drifted: NVDA col {nvda.index('0.92')}, "
            f"MU col {mu.index('0.09')}\n{nvda}\n{mu}"
        )
        # Guard constants so future edits can't quietly rewrite the contract.
        assert _W_DSIGN == 2
        assert _W_EPS == 6


# ── format_valuation ───────────────────────────────────────────────

class TestValuationColumns:
    def _sample(self) -> tuple[dict, list[str]]:
        infos = {
            "NVDA": {
                "regularMarketPrice": 137.50, "regularMarketPreviousClose": 135.00,
                "marketCap": 3.4e12, "trailingPE": 45.2, "forwardPE": 32.1,
                "enterpriseToEbitda": 38.5,
                "profitMargins": 0.55, "revenueGrowth": 0.84,
            },
            "AVGO": {
                "regularMarketPrice": 1650.00, "regularMarketPreviousClose": 1680.00,
                "marketCap": 7.7e11, "trailingPE": 38.1, "forwardPE": 29.4,
                "enterpriseToEbitda": 22.8,
                "profitMargins": 0.28, "revenueGrowth": 0.47,
            },
            "MU": {
                "regularMarketPrice": 98.25, "regularMarketPreviousClose": 101.50,
                "marketCap": 1.1e11, "trailingPE": None, "forwardPE": 14.2,
                "enterpriseToEbitda": 12.1,
                "profitMargins": -0.02, "revenueGrowth": -0.15,
            },
        }
        return infos, ["NVDA", "AVGO", "MU"]

    def test_rows_total_width_71(self):
        """Every data row must render to exactly 71 visible cells (including 2-space indent)."""
        from screens.valuation import format_valuation
        infos, syms = self._sample()
        result = format_valuation(infos, syms)
        rows = _data_rows(result)
        # Skip the header row; keep only rows containing our symbols.
        data_rows = [r for r in rows if any(s in r for s in syms)]
        assert len(data_rows) == 3, f"expected 3 rows, got: {data_rows}"
        widths = {len(r.rstrip()) for r in data_rows}
        # All rows same width
        assert len(widths) == 1, f"valuation row widths diverge: {widths}\n" + "\n".join(data_rows)
        # Contract: indent + content = 71 visible cells. The divider is 2 + '─'*71 = 73 cells
        # so it's slightly wider than the rows by design (divider has no trailing label).
        assert widths.pop() == 71

    def test_negative_pct_no_arrow_overflow(self):
        """Negative Chg% must not burn an extra cell for an arrow glyph."""
        from screens.valuation import format_valuation
        infos, syms = self._sample()
        result = format_valuation(infos, syms)
        # AVGO and MU are both down in the sample. Neither should contain the
        # arrow glyphs that `color_pct()` would have emitted.
        assert "▼" not in result
        assert "▲" not in result


# ── format_consolidated (IBKR Combined + Positions) ─────────────────

class TestConsolidatedColumns:
    SAMPLE = (
        "# Consolidated View (CAD)\n"
        "**Accounts**: 2\n"
        "\n"
        "## Account Summary\n"
        "| Account | NLV | Leverage | Margin Util | Cushion |\n"
        "|---------|-----|---------|-------------|--------|\n"
        "| U1854213 | $500,000.00 CAD | 1.85x | 54.0% | 15.2% |\n"
        "| U9876543 | $200,000.00 CAD | 1.20x | 22.0% | 35.0% |\n"
        "\n"
        "**Combined NLV**: $700,000.00 CAD\n"
        "**Combined Leverage**: 1.65x\n"
        "**Combined Margin Util**: 45.0%\n"
        "\n"
        "## All Positions\n"
        "| Account | Symbol | Shares | Value | Converted | P&L |\n"
        "|---------|--------|--------|-------|-----------|-----|\n"
        "| U1854213 | NVDA | 500 | $62,500.00 USD | $89,375.00 CAD | +$12,500.00 CAD |\n"
        "| U1854213 | AVGO | 200 | $40,000.00 USD | $57,200.00 CAD | -$3,000.00 CAD |\n"
        "| U9876543 | MU   | 100 | $9,825.00 USD  | $14,050.00 CAD |    -$425.00 CAD |\n"
        "\n"
        "**Total Position Value**: $146,575.00 CAD\n"
        "**Total Unrealized P&L**: +$9,075.00 CAD\n"
    )

    def _position_rows(self, result: str) -> list[str]:
        """Rows that look like: acct_tag + symbol + shares + values."""
        syms = ("NVDA", "AVGO", "MU")
        return [r for r in _data_rows(result) if any(s in r for s in syms)]

    def test_pnl_sign_column_aligned(self):
        """`+` and `-` sign glyphs must occupy the exact same column across rows."""
        from screens.ibkr_screen import format_consolidated
        result = format_consolidated(self.SAMPLE)
        rows = self._position_rows(result)
        assert len(rows) >= 3, f"expected 3 position rows, got {len(rows)}: {rows}"
        # Locate the sign glyph for each row by scanning right-to-left for
        # + or - that is followed by digits.
        sign_cols = []
        for r in rows:
            # Rightmost match — the P&L sign is the last signed number on the row.
            # Sign lives in its own column separated by spaces from digits (by design).
            matches = list(re.finditer(r"[+\-](?=\s*\d)", r))
            assert matches, f"no sign glyph in row: {r!r}"
            sign_cols.append(matches[-1].start())
        assert len(set(sign_cols)) == 1, f"P&L sign columns drift: {sign_cols}\n" + "\n".join(rows)

    def test_combined_heading_rendered_once(self):
        """Combined heading emits exactly once — not per metric."""
        from screens.ibkr_screen import format_consolidated
        result = format_consolidated(self.SAMPLE)
        vis = _visible(result)
        # The heading is the bare word "Combined" on its own indented line.
        combined_headings = [ln for ln in vis.splitlines() if ln.strip() == "Combined"]
        assert len(combined_headings) == 1, (
            f"expected 1 Combined heading, got {len(combined_headings)}: {combined_headings}"
        )

    def test_combined_rows_label_aligned(self):
        """Combined metric rows (NLV / Leverage / Margin Util) share label column."""
        from screens.ibkr_screen import format_consolidated
        result = format_consolidated(self.SAMPLE)
        vis_lines = _visible(result).splitlines()
        combined_labels = [ln for ln in vis_lines
                           if ln.strip().startswith(("NLV", "Leverage", "Margin Util"))
                           and "U18" not in ln and "U98" not in ln]
        # Each label should start at the same column — pad(label, 20) contract.
        starts = {len(ln) - len(ln.lstrip()) for ln in combined_labels}
        assert len(starts) == 1, (
            f"Combined label column drifts: {starts}\n" + "\n".join(combined_labels)
        )


# ── format_briefing movers ──────────────────────────────────────────

class TestBriefingMoversColumns:
    def _sample(self) -> dict:
        return {
            "portfolio": {},
            "macro": {},
            "movers": {
                "gainers": [
                    {"symbol": "NVDA", "price": 137.50, "pct": 1.85},
                    {"symbol": "AVGO", "price": 1650.00, "pct": 2.34},
                    {"symbol": "GOOG", "price": 180.00, "pct": 0.50},
                ],
                "losers": [
                    {"symbol": "MU",   "price": 98.25, "pct": -3.10},
                    {"symbol": "TSLA", "price": 412.00, "pct": -12.40},
                    {"symbol": "LRCX", "price": 875.00, "pct": -0.80},
                ],
            },
            "sectors": [],
            "news": [],
            "earnings": [],
        }

    def test_mover_row_widths_uniform(self):
        """Each mover row (left + right pair) has consistent visible width."""
        from screens.briefing import format_briefing
        result = format_briefing(self._sample())
        rows = _data_rows(result)
        # Keep rows that show both a gainer and a loser — they have the pair shape.
        # A pair row contains a ▲ AND a ▼ glyph.
        pair_rows = [r for r in rows if "▲" in r and "▼" in r]
        assert len(pair_rows) == 3, f"expected 3 paired rows, got {len(pair_rows)}"
        widths = {len(r.rstrip()) for r in pair_rows}
        assert len(widths) == 1, (
            f"mover pair-row widths diverge (the exact bug we just fixed): {widths}\n"
            + "\n".join(pair_rows)
        )
