"""Morning briefing screen — portfolio health, movers, earnings, macro context."""

from i18n import t


def format_briefing(data: dict) -> str:
    """Format morning briefing from assembled data.

    Args:
        data: {
            "portfolio": {nlv, cushion, leverage, daily_pnl} | None,
            "movers": {"gainers": [...], "losers": [...]},
            "earnings_week": [{symbol, date, days_until}],
            "macro": {...} | None,
            "nlv_drawdown": float | None,
        }
    """
    lines = []

    # ── Portfolio Health ──
    port = data.get("portfolio")
    # Only show if we actually have data (not just an empty/None-filled dict)
    has_port_data = port and any(
        port.get(k) is not None for k in ("nlv", "cushion", "leverage", "daily_pnl")
    )
    if has_port_data:
        lines.append(f"  [bold #00c8ff]{t('briefing.portfolio')}[/]")
        row_parts = []

        nlv = port.get("nlv")
        if nlv is not None:
            row_parts.append(f"[dim]NLV[/] [bold white]${nlv:,.0f}[/]")

        cushion = port.get("cushion")
        if cushion is not None:
            cc = "#ff3232" if cushion < 10 else "#ffc800" if cushion < 15 else "green"
            row_parts.append(f"[dim]{t('sidebar.cushion')}[/] [{cc}]{cushion:.1f}%[/]")

        lever = port.get("leverage")
        if lever is not None:
            lc = "#ff3232" if lever > 3 else "#ffc800" if lever > 2 else "green"
            row_parts.append(f"[dim]{t('sidebar.lever')}[/] [{lc}]{lever:.1f}x[/]")

        pnl = port.get("daily_pnl")
        if pnl is not None:
            pc = "green" if pnl >= 0 else "#ff3232"
            row_parts.append(f"[dim]P&L[/] [{pc}]{pnl:+,.0f}[/]")

        dd = data.get("nlv_drawdown")
        if dd is not None and dd < 0:
            dc = "#ff3232" if dd < -5 else "#ffc800"
            row_parts.append(f"[dim]DD[/] [{dc}]{dd:.1f}%[/]")

        # Compact horizontal layout — all on one or two lines
        if row_parts:
            lines.append("    " + "  │  ".join(row_parts))
        lines.append("")
    else:
        lines.append(f"  [dim]{t('briefing.no_ibkr')}[/]\n")

    # ── Macro Context ── (two-column layout for density)
    macro = data.get("macro")
    if macro:
        lines.append(f"  [bold #00c8ff]{t('briefing.macro')}[/]")

        # Build macro items as (label, formatted_value) pairs
        items = []
        for key, label, fmt_type in [
            ("sp500_pct", "S&P 500", "pct"),
            ("nasdaq_pct", "Nasdaq", "pct"),
            ("sox_pct", "SOX", "pct"),
            ("hsi_pct", "HSI", "pct"),
            ("vix", "VIX", "vix"),
            ("oil", "WTI Oil", "price"),
            ("gold", "Gold", "price"),
        ]:
            val = macro.get(key)
            if val is None:
                continue
            if fmt_type == "pct":
                mc = "green" if val >= 0 else "#ff3232"
                items.append((label, f"[{mc}]{val:+.1f}%[/]"))
            elif fmt_type == "vix":
                vc = "#ff3232" if val > 25 else "#ffc800" if val > 18 else "green"
                items.append((label, f"[{vc}]{val:.1f}[/]"))
            elif fmt_type == "price":
                items.append((label, f"[white]${val:,.0f}[/]"))

        # Render in two columns to use horizontal space
        col_w = 28  # width per column
        for i in range(0, len(items), 2):
            left = items[i]
            left_str = f"[dim]{left[0]:<10}[/] {left[1]}"
            if i + 1 < len(items):
                right = items[i + 1]
                right_str = f"[dim]{right[0]:<10}[/] {right[1]}"
                lines.append(f"    {left_str}     {right_str}")
            else:
                lines.append(f"    {left_str}")
        lines.append("")

    # ── Watchlist Movers ── (two-column: gainers left, losers right)
    movers = data.get("movers", {})
    gainers = movers.get("gainers", [])
    losers = movers.get("losers", [])
    if gainers or losers:
        lines.append(f"  [bold #00c8ff]{t('briefing.movers')}[/]")
        max_rows = max(len(gainers[:5]), len(losers[:5]))
        for i in range(max_rows):
            left = ""
            right = ""
            if i < len(gainers):
                g = gainers[i]
                left = f"[green]▲ {g['symbol']:<5}[/] [green]{g['pct']:+.1f}%[/]  [dim]${g['price']:,.2f}[/]"
            if i < len(losers):
                lo = losers[i]
                right = f"[#ff3232]▼ {lo['symbol']:<5}[/] [#ff3232]{lo['pct']:+.1f}%[/]  [dim]${lo['price']:,.2f}[/]"
            if left and right:
                lines.append(f"    {left}     {right}")
            elif left:
                lines.append(f"    {left}")
            elif right:
                lines.append(f"    {'':30}{right}")
        lines.append("")

    # ── Earnings This Week ──
    earnings = data.get("earnings_week", [])
    if earnings:
        lines.append(f"  [bold #00c8ff]{t('briefing.earnings')}[/]")
        for e in earnings[:5]:
            days = e.get("days_until")
            if days is not None and days <= 1:
                urgency = "[bold #ff3232]"
            elif days is not None and days <= 3:
                urgency = "[#ffc800]"
            else:
                urgency = "[dim]"
            lines.append(
                f"    {urgency}{e['symbol']:<6}[/]"
                f"[dim]{e['date']}[/]  "
                f"{urgency}{days}d[/]")
        lines.append("")

    if not lines:
        return f"[dim]{t('briefing.empty')}[/]"

    return "\n".join(lines)
