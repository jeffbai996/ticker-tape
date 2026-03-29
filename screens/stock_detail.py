"""Stock detail screens — dividends, short interest, analyst ratings.

All sourced from yfinance .info dict. Light formatting over cached data.
"""

from i18n import t
from formatters import fmt_pct, fmt_ratio, fmt_num, fmt_cap


def _row(label: str, value: str) -> str:
    pad = max(1, 24 - len(label))
    return f"  {label}{' ' * pad}{value}"


# ── Dividends ──


def format_dividends(info: dict | None, symbol: str) -> str:
    """Format dividend information from .info dict."""
    if not info:
        return f"[#ff3232]No data for {symbol}[/]"

    dy = info.get("dividendYield")
    rate = info.get("dividendRate")
    payout = info.get("payoutRatio")
    ex_date = info.get("exDividendDate")
    avg_5y = info.get("fiveYearAvgDividendYield")
    trailing = info.get("trailingAnnualDividendYield")

    if dy is None and rate is None:
        return f"[dim]{symbol} does not pay a dividend.[/]"

    lines = []

    # Current yield & rate
    if dy is not None:
        color = "green" if dy > 0.03 else "#ffc800" if dy > 0.01 else "dim"
        lines.append(_row("Yield", f"[{color}]{dy * 100:.2f}%[/]"))
    if rate is not None:
        lines.append(_row("Annual Rate", f"${rate:.2f}"))
    if trailing is not None:
        lines.append(_row("Trailing Yield", f"{trailing * 100:.2f}%"))
    if avg_5y is not None:
        lines.append(_row("5Y Avg Yield", f"{avg_5y:.2f}%"))

    # Payout ratio
    if payout is not None:
        color = "green" if payout < 0.6 else "#ffc800" if payout < 0.8 else "#ff3232"
        lines.append(_row("Payout Ratio", f"[{color}]{payout * 100:.1f}%[/]"))

    # Ex-dividend date
    if ex_date:
        from datetime import datetime
        if isinstance(ex_date, (int, float)):
            ex_str = datetime.fromtimestamp(ex_date).strftime("%Y-%m-%d")
        else:
            ex_str = str(ex_date)[:10]
        lines.append(_row("Ex-Dividend Date", ex_str))

    return "\n".join(lines)


# ── Short Interest ──


def format_short_interest(info: dict | None, symbol: str) -> str:
    """Format short interest data from .info dict."""
    if not info:
        return f"[#ff3232]No data for {symbol}[/]"

    si_pct = info.get("shortPercentOfFloat")
    si_ratio = info.get("shortRatio")  # days to cover
    shares_short = info.get("sharesShort")
    shares_prior = info.get("sharesShortPriorMonth")
    si_date = info.get("dateShortInterest")
    shares_out = info.get("sharesOutstanding")
    float_shares = info.get("floatShares")

    if si_pct is None and shares_short is None:
        return f"[dim]No short interest data for {symbol}.[/]"

    lines = []

    # Short % of float — the headline number
    if si_pct is not None:
        color = "#ff3232" if si_pct > 0.10 else "#ffc800" if si_pct > 0.05 else "green"
        lines.append(_row("Short % of Float", f"[{color}]{si_pct * 100:.2f}%[/]"))

    # Days to cover
    if si_ratio is not None:
        color = "#ff3232" if si_ratio > 5 else "#ffc800" if si_ratio > 3 else "dim"
        lines.append(_row("Days to Cover", f"[{color}]{si_ratio:.1f}[/]"))

    # Shares short
    if shares_short is not None:
        lines.append(_row("Shares Short", fmt_num(shares_short)))

    # Month-over-month change
    if shares_short and shares_prior:
        delta = shares_short - shares_prior
        delta_pct = (delta / shares_prior) * 100 if shares_prior else 0
        color = "#ff3232" if delta > 0 else "green"
        lines.append(_row("vs Prior Month",
                          f"[{color}]{delta:+,.0f} ({delta_pct:+.1f}%)[/]"))

    # Float & outstanding for context
    if float_shares:
        lines.append(_row("Float", fmt_num(float_shares)))
    if shares_out:
        lines.append(_row("Shares Outstanding", fmt_num(shares_out)))

    # Report date
    if si_date:
        from datetime import datetime
        if isinstance(si_date, (int, float)):
            date_str = datetime.fromtimestamp(si_date).strftime("%Y-%m-%d")
        else:
            date_str = str(si_date)[:10]
        lines.append(_row("[dim]Report Date[/]", f"[dim]{date_str}[/]"))

    return "\n".join(lines)


# ── Analyst Ratings ──


def format_ratings(info: dict | None, symbol: str, recommendations: list[dict] | None = None) -> str:
    """Format analyst ratings and price targets from .info dict."""
    if not info:
        return f"[#ff3232]No data for {symbol}[/]"

    rec = info.get("recommendationKey")
    n_analysts = info.get("numberOfAnalystOpinions")
    target_mean = info.get("targetMeanPrice")
    target_median = info.get("targetMedianPrice")
    target_high = info.get("targetHighPrice")
    target_low = info.get("targetLowPrice")
    price = info.get("regularMarketPrice") or info.get("currentPrice")

    if rec is None and target_mean is None:
        return f"[dim]No analyst data for {symbol}.[/]"

    lines = []

    # Consensus rating
    if rec:
        color_map = {
            "strong_buy": "green", "buy": "green",
            "hold": "#ffc800", "underperform": "#ff3232",
            "sell": "#ff3232", "strong_sell": "#ff3232",
        }
        color = color_map.get(rec, "dim")
        label = rec.upper().replace("_", " ")
        analyst_str = f" ({n_analysts} analysts)" if n_analysts else ""
        lines.append(_row("Consensus", f"[bold {color}]{label}[/]{analyst_str}"))

    # Price targets
    if target_mean and price:
        upside = ((target_mean - price) / price) * 100
        color = "green" if upside > 0 else "#ff3232"
        lines.append(_row("Mean Target",
                          f"${target_mean:.2f} [{color}]({upside:+.1f}%)[/]"))
    if target_median:
        lines.append(_row("Median Target", f"${target_median:.2f}"))

    # Range
    if target_low and target_high:
        lines.append(_row("Target Range",
                          f"${target_low:.2f} — ${target_high:.2f}"))

    # Spread (high/low as % of mean)
    if target_low and target_high and target_mean:
        spread = ((target_high - target_low) / target_mean) * 100
        lines.append(_row("Target Spread", f"{spread:.1f}%"))

    # Recent upgrades/downgrades
    if recommendations:
        lines.append(f"  [dim]{'─' * 58}[/]")
        for rec_item in recommendations:
            firm = rec_item.get("firm", "Unknown")
            if len(firm) > 18:
                firm = firm[:17] + "…"
            to_grade = rec_item.get("toGrade", "")
            from_grade = rec_item.get("fromGrade", "")
            action = rec_item.get("action", "")

            if action in ("up", "upgrade", "init"):
                color = "green"
            elif action in ("down", "downgrade"):
                color = "#ff3232"
            else:
                color = "dim"

            grade_str = to_grade
            if from_grade and from_grade != to_grade:
                grade_str = f"{from_grade} → {to_grade}"

            # Price target if available
            pt = rec_item.get("priceTarget")
            prior = rec_item.get("priorTarget")
            pt_str = ""
            if pt and pt > 0:
                if prior and prior > 0 and prior != pt:
                    pt_str = f"  ${prior:.0f}→${pt:.0f}"
                else:
                    pt_str = f"  ${pt:.0f}"

            lines.append(f"  [{color}]{firm:<20}{grade_str:<22}{pt_str}[/]")

    return "\n".join(lines)
