"""Earnings surprise tracker — watchlist-wide historical earnings performance."""

from i18n import t


def format_surprises(data: dict) -> str:
    """Format watchlist-wide earnings surprise data.

    Args:
        data: {
            "symbols": {SYM: {"events": [...], "summary": {...}}, ...},
            "watchlist_summary": {total_beats, total_total, avg_beat_rate, avg_surprise, avg_move}
        }
    """
    symbols = data.get("symbols", {})
    if not symbols:
        return f"[dim]{t('surprises.empty')}[/]"

    # Header
    lines = []
    hdr = (
        f"  [bold #00c8ff]{'Symbol':<7}[/]"
        f"[bold #00c8ff]{'Last EPS':>9}[/]"
        f"[bold #00c8ff]{'Surprise':>10}[/]"
        f"[bold #00c8ff]{'Move':>8}[/]"
        f"[bold #00c8ff]{'Streak':>8}[/]"
        f"[bold #00c8ff]{'Beat %':>8}[/]"
        f"[bold #00c8ff]{'Avg Move':>9}[/]"
    )
    lines.append(hdr)
    lines.append(f"  [dim]{'─' * 59}[/]")

    # Sort by beat rate descending, then by symbol
    sorted_syms = sorted(
        symbols.items(),
        key=lambda x: (x[1]["summary"].get("beat_rate") or 0, x[0]),
        reverse=True,
    )

    for sym, info in sorted_syms:
        s = info["summary"]

        # Last EPS
        eps_str = f"${s['last_eps']:.2f}" if s.get("last_eps") is not None else "—"

        # Last surprise %
        if s.get("last_surprise") is not None:
            sc = "green" if s["last_surprise"] > 0 else "#ff3232"
            surp_str = f"[{sc}]{s['last_surprise']:+.1f}%[/]"
        else:
            surp_str = "[dim]—[/]"

        # Last price move
        if s.get("last_move") is not None:
            mc = "green" if s["last_move"] > 0 else "#ff3232"
            move_str = f"[{mc}]{s['last_move']:+.1f}%[/]"
        else:
            move_str = "[dim]—[/]"

        # Beat streak
        streak = s.get("beat_streak", 0)
        streak_str = f"[green]{streak}[/]" if streak > 0 else "[dim]0[/]"

        # Beat rate
        if s.get("beat_rate") is not None:
            br = s["beat_rate"] * 100
            bc = "green" if br >= 75 else "#ffc800" if br >= 50 else "#ff3232"
            rate_str = f"[{bc}]{br:.0f}%[/]"
            rate_detail = f" [dim]({s['beats']}/{s['total']})[/]"
        else:
            rate_str = "[dim]—[/]"
            rate_detail = ""

        # Avg move
        if s.get("avg_move") is not None:
            ac = "green" if s["avg_move"] > 0 else "#ff3232"
            avg_str = f"[{ac}]{s['avg_move']:+.1f}%[/]"
        else:
            avg_str = "[dim]—[/]"

        lines.append(
            f"  [bold #ffc800]{sym:<7}[/]"
            f"{eps_str:>9}"
            f"{surp_str:>20}"  # extra width for markup
            f"{move_str:>18}"
            f"{streak_str:>18}"
            f"{rate_str:>18}{rate_detail}"
            f"{avg_str:>18}"
        )

    # Watchlist summary
    ws = data.get("watchlist_summary", {})
    lines.append(f"\n  [dim]{'─' * 59}[/]")
    parts = []
    if ws.get("avg_beat_rate") is not None:
        parts.append(f"Beat rate: {ws['avg_beat_rate']*100:.0f}% ({ws['total_beats']}/{ws['total_total']})")
    if ws.get("avg_surprise") is not None:
        parts.append(f"Avg surprise: {ws['avg_surprise']:+.1f}%")
    if ws.get("avg_move") is not None:
        parts.append(f"Avg move: {ws['avg_move']:+.1f}%")
    if parts:
        lines.append(f"  [dim]{' │ '.join(parts)}[/]")

    return "\n".join(lines)
