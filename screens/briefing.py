"""Morning briefing screen — portfolio health, movers, earnings, macro context, news, sectors."""

from i18n import t


def format_briefing(data: dict) -> str:
    """Format morning briefing from assembled data.

    Args:
        data: {
            "portfolio": {nlv, cushion, leverage, daily_pnl} | None,
            "movers": {"gainers": [...], "losers": [...]},
            "earnings_week": [{symbol, date, days_until, eps_est}],
            "macro": {...} | None,
            "nlv_drawdown": float | None,
            "news": [{symbol, headlines: [{title, publisher, age}]}],
            "sectors": [{symbol, name, pct}],
        }
    """
    lines: list[str] = []

    # ── Portfolio Health ──
    port = data.get("portfolio")
    has_port_data = port and any(
        port.get(k) is not None for k in ("nlv", "cushion", "leverage", "daily_pnl")
    )
    if has_port_data:
        lines.append(f"  [bold #00c8ff]{t('briefing.portfolio')}[/]")

        # Row 1: NLV and Daily P&L
        nlv = port.get("nlv")
        pnl = port.get("daily_pnl")
        row1_parts = []
        if nlv is not None:
            row1_parts.append(f"[dim]NLV[/]       [bold white]${nlv:,.0f}[/]")
        if pnl is not None:
            pc = "green" if pnl >= 0 else "#ff3232"
            pnl_str = f"[{pc}]{pnl:+,.0f}[/]"
            # Show % return if NLV available
            if nlv and nlv > 0:
                pnl_pct = (pnl / nlv) * 100
                pnl_str += f"  [{pc}]({pnl_pct:+.2f}%)[/]"
            row1_parts.append(f"[dim]Day P&L[/]   {pnl_str}")
        if row1_parts:
            lines.append("    " + "     ".join(row1_parts))

        # Row 2: Cushion, Leverage, Drawdown
        cushion = port.get("cushion")
        lever = port.get("leverage")
        dd = data.get("nlv_drawdown")
        row2_parts = []
        if cushion is not None:
            cc = "#ff3232" if cushion < 10 else "#ffc800" if cushion < 15 else "green"
            row2_parts.append(f"[dim]Cushion[/]   [{cc}]{cushion:.1f}%[/]")
        if lever is not None:
            lc = "#ff3232" if lever > 3 else "#ffc800" if lever > 2 else "green"
            row2_parts.append(f"[dim]Leverage[/]  [{lc}]{lever:.1f}x[/]")
        if dd is not None and dd < 0:
            dc = "#ff3232" if dd < -5 else "#ffc800"
            row2_parts.append(f"[dim]Drawdown[/]  [{dc}]{dd:.1f}%[/]")
        if row2_parts:
            lines.append("    " + "     ".join(row2_parts))

        lines.append("")
    else:
        lines.append(f"  [dim]{t('briefing.no_ibkr')}[/]\n")

    # ── Macro Context ── (two-column layout, expanded)
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
            ("dxy", "DXY", "dxy"),
            ("tnx", "10Y Yield", "yield"),
            ("oil", "WTI Oil", "commodity"),
            ("gold", "Gold", "commodity"),
            ("btc", "Bitcoin", "btc"),
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
            elif fmt_type == "dxy":
                pct = macro.get("dxy_pct")
                pct_str = ""
                if pct is not None:
                    dc = "green" if pct >= 0 else "#ff3232"
                    pct_str = f" [{dc}]{pct:+.1f}%[/]"
                items.append((label, f"[white]{val:.1f}[/]{pct_str}"))
            elif fmt_type == "yield":
                pct = macro.get("tnx_pct")
                pct_str = ""
                if pct is not None:
                    yc = "#ff3232" if pct > 0 else "green"
                    pct_str = f" [{yc}]{pct:+.1f}%[/]"
                items.append((label, f"[white]{val:.2f}%[/]{pct_str}"))
            elif fmt_type == "commodity":
                # Get the pct change for this commodity
                pct_key = key + "_pct"
                pct = macro.get(pct_key)
                pct_str = ""
                if pct is not None:
                    cc = "green" if pct >= 0 else "#ff3232"
                    pct_str = f" [{cc}]{pct:+.1f}%[/]"
                items.append((label, f"[white]${val:,.0f}[/]{pct_str}"))
            elif fmt_type == "btc":
                pct = macro.get("btc_pct")
                pct_str = ""
                if pct is not None:
                    bc = "green" if pct >= 0 else "#ff3232"
                    pct_str = f" [{bc}]{pct:+.1f}%[/]"
                items.append((label, f"[white]${val:,.0f}[/]{pct_str}"))

        # Render in two columns
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
        # Header
        lines.append(
            f"    [dim]{'GAINERS':<30}{'LOSERS':<30}[/]"
        )
        max_rows = max(len(gainers[:5]), len(losers[:5]))
        for i in range(max_rows):
            left = ""
            right = ""
            if i < len(gainers):
                g = gainers[i]
                chg = g["price"] * g["pct"] / (100 + g["pct"]) if g["pct"] != -100 else 0
                left = (
                    f"[green]▲ {g['symbol']:<6}[/]"
                    f"[green]{g['pct']:+5.1f}%[/]  "
                    f"[dim]${g['price']:>8,.2f}[/] "
                    f"[green]{chg:+.2f}[/]"
                )
            if i < len(losers):
                lo = losers[i]
                chg = lo["price"] * lo["pct"] / (100 + lo["pct"]) if lo["pct"] != -100 else 0
                right = (
                    f"[#ff3232]▼ {lo['symbol']:<6}[/]"
                    f"[#ff3232]{lo['pct']:+5.1f}%[/]  "
                    f"[dim]${lo['price']:>8,.2f}[/] "
                    f"[#ff3232]{chg:+.2f}[/]"
                )
            if left and right:
                lines.append(f"    {left}     {right}")
            elif left:
                lines.append(f"    {left}")
            elif right:
                lines.append(f"    {'':30}{right}")
        lines.append("")

    # ── Sector Snapshot ──
    sectors = data.get("sectors", [])
    if sectors:
        lines.append(f"  [bold #00c8ff]{t('briefing.sectors')}[/]")
        # Show in two columns, sorted best to worst (already sorted)
        sector_items = []
        for s in sectors:
            sc = "green" if s["pct"] >= 0 else "#ff3232"
            sector_items.append(
                f"[dim]{s['name']:<18}[/] [{sc}]{s['pct']:+5.1f}%[/]"
            )
        for i in range(0, len(sector_items), 2):
            left = sector_items[i]
            if i + 1 < len(sector_items):
                right = sector_items[i + 1]
                lines.append(f"    {left}  {right}")
            else:
                lines.append(f"    {left}")
        lines.append("")

    # ── News Headlines ──
    news = data.get("news", [])
    if news:
        lines.append(f"  [bold #00c8ff]{t('briefing.news')}[/]")
        for item in news:
            sym = item["symbol"]
            headlines = item.get("headlines", [])
            if not headlines:
                continue
            lines.append(f"    [bold white]{sym}[/]")
            for h in headlines[:3]:
                title = h.get("title", "")
                # Truncate long headlines to fit terminal
                if len(title) > 72:
                    title = title[:69] + "..."
                age = h.get("age", "")
                pub = h.get("publisher", "")
                meta_parts = []
                if pub:
                    meta_parts.append(pub)
                if age:
                    meta_parts.append(age)
                meta = " · ".join(meta_parts)
                lines.append(f"      [dim]•[/] {title}")
                if meta:
                    lines.append(f"        [dim]{meta}[/]")
        lines.append("")

    # ── Earnings This Week ──
    earnings = data.get("earnings_week", [])
    if earnings:
        lines.append(f"  [bold #00c8ff]{t('briefing.earnings')}[/]")
        for e in earnings[:8]:
            days = e.get("days_until")
            if days is not None and days <= 1:
                urgency = "[bold #ff3232]"
            elif days is not None and days <= 3:
                urgency = "[#ffc800]"
            else:
                urgency = "[dim]"
            # Build detail string with EPS estimate and timing
            detail_parts = [f"[dim]{e['date']}[/]", f"{urgency}{days}d[/]"]
            eps_est = e.get("eps_est")
            if eps_est is not None:
                detail_parts.append(f"[dim]Est EPS[/] [white]${eps_est:.2f}[/]")
            lines.append(
                f"    {urgency}{e['symbol']:<6}[/]  "
                + "  ".join(detail_parts)
            )
        lines.append("")

    if not lines:
        return f"[dim]{t('briefing.empty')}[/]"

    return "\n".join(lines)
