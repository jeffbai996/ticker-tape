"""Timeline screen — NLV history chart with drawdown and leverage trend."""

from datetime import datetime

from i18n import t


def format_timeline(snapshots: list[dict], peak: float | None) -> str:
    """Format NLV history as an ASCII chart with stats.

    Args:
        snapshots: [{timestamp, nlv, cushion, leverage, daily_pnl}, ...]
        peak: max NLV in the window (for drawdown calc)
    """
    if not snapshots:
        return f"[dim]{t('timeline.empty')}[/]"

    # Group by date — use last snapshot of each day
    by_day: dict[str, dict] = {}
    for s in snapshots:
        ts = s["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        day_key = ts.strftime("%Y-%m-%d")
        by_day[day_key] = s

    days = sorted(by_day.keys())
    if len(days) < 2:
        # Not enough data for a chart yet — show current stats
        s = snapshots[-1]
        return (
            f"[dim]{t('timeline.insufficient')}[/]\n\n"
            f"  [white]NLV[/]       [bold white]${s['nlv']:,.0f}[/]\n"
            + (f"  [white]{t('sidebar.cushion')}[/]   [white]{s['cushion']:.1f}%[/]\n" if s.get("cushion") else "")
            + (f"  [white]{t('sidebar.lever')}[/]  [white]{s['leverage']:.1f}x[/]\n" if s.get("leverage") else "")
        )

    daily_nlv = [by_day[d]["nlv"] for d in days]

    # Current values
    current = snapshots[-1]
    cur_nlv = current["nlv"]
    cur_cushion = current.get("cushion")
    cur_lever = current.get("leverage")

    # Drawdown from peak
    drawdown = None
    if peak and peak > 0:
        drawdown = ((cur_nlv - peak) / peak) * 100

    # ── Build chart ──────────────────────────────
    width = 60
    rows = 8
    lo, hi = min(daily_nlv), max(daily_nlv)
    spread = hi - lo if hi != lo else 1

    # Downsample if more days than width
    if len(daily_nlv) > width:
        step = len(daily_nlv) / width
        sampled = [daily_nlv[int(i * step)] for i in range(width)]
        sampled_days = [days[int(i * step)] for i in range(width)]
    else:
        sampled = daily_nlv
        sampled_days = days

    blocks = " ▁▂▃▄▅▆▇█"
    normalized = [(p - lo) / spread for p in sampled]
    color = "green" if sampled[-1] >= sampled[0] else "#ff3232"

    chart_lines = []
    for r in range(rows):
        row_bottom = (rows - 1 - r) / rows
        row_top = (rows - r) / rows
        row_chars = ""
        for h in normalized:
            if h >= row_top:
                row_chars += "█"
            elif h > row_bottom:
                frac = (h - row_bottom) / (row_top - row_bottom)
                idx = max(1, min(8, int(frac * 8)))
                row_chars += blocks[idx]
            else:
                row_chars += " "

        # Y-axis labels on right: top row = hi, bottom row = lo
        if r == 0:
            label = f" ${hi:,.0f}"
        elif r == rows - 1:
            label = f" ${lo:,.0f}"
        elif r == rows // 2:
            mid = (hi + lo) / 2
            label = f" ${mid:,.0f}"
        else:
            label = ""

        chart_lines.append(f"  [{color}]{row_chars}[/][dim]{label}[/]")

    # X-axis: first and last date
    first_date = sampled_days[0][5:]   # MM-DD
    last_date = sampled_days[-1][5:]
    axis_pad = width - len(first_date) - len(last_date)
    x_axis = f"  [dim]{first_date}{' ' * max(axis_pad, 1)}{last_date}[/]"

    # ── Stats section ────────────────────────────
    lines = []
    lines.append("\n".join(chart_lines))
    lines.append(x_axis)
    lines.append("")

    # Key metrics row
    stats = []
    stats.append(f"[bold white]NLV[/] ${cur_nlv:,.0f}")
    if peak:
        stats.append(f"[bold white]{t('timeline.peak')}[/] ${peak:,.0f}")
    if drawdown is not None:
        dd_color = "#ff3232" if drawdown < -5 else "#ffc800" if drawdown < 0 else "green"
        stats.append(f"[bold white]{t('timeline.drawdown')}[/] [{dd_color}]{drawdown:+.1f}%[/]")
    lines.append("  " + "  │  ".join(stats))

    # Leverage and cushion trends (last value)
    extra = []
    if cur_cushion is not None:
        c_color = "#ff3232" if cur_cushion < 10 else "#ffc800" if cur_cushion < 15 else "green"
        extra.append(f"[bold white]{t('sidebar.cushion')}[/] [{c_color}]{cur_cushion:.1f}%[/]")
    if cur_lever is not None:
        l_color = "#ff3232" if cur_lever > 3 else "#ffc800" if cur_lever > 2 else "green"
        extra.append(f"[bold white]{t('sidebar.lever')}[/] [{l_color}]{cur_lever:.1f}x[/]")
    if extra:
        lines.append("  " + "  │  ".join(extra))

    # Period stats
    change_pct = ((daily_nlv[-1] - daily_nlv[0]) / daily_nlv[0]) * 100
    ch_color = "green" if change_pct >= 0 else "#ff3232"
    lines.append(f"\n  [dim]{len(days)} days[/]  [{ch_color}]{change_pct:+.1f}%[/] [dim]period return[/]")

    return "\n".join(lines)
