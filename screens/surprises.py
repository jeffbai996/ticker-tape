"""Earnings surprise tracker — watchlist-wide historical earnings performance."""

from i18n import t


# Column visible widths. Every markup-wrapped value MUST render to exactly
# this many cells, otherwise downstream columns drift.
_W_SYM   = 6
_W_DSIGN = 2   # "$" or "-$" column (sign leads the dollar, not the digit)
_W_EPS   = 6   # e.g. "12.20", "0.09", "1,234"  (no sign, no $ — handled in _W_DSIGN)
_W_SURP  = 7   # e.g. "+75.0%"
_W_MOVE  = 7
_W_STRK  = 3
_W_RATE  = 4   # "100%" or " 75%"
_W_DETL  = 6   # "(3/4)" or "(4/4)"
_W_AVG   = 7


def _eps_cells(eps: float | None) -> tuple[str, str]:
    """Return (sign_col, number_col) both right-justified to their widths.

    Sign col shows '$' for positive, '-$' for negative, '  ' for missing.
    Number col shows the unsigned value or '—' when missing.
    """
    if eps is None:
        return f"{'':>{_W_DSIGN}}", f"{'—':>{_W_EPS}}"
    if eps < 0:
        sign = "-$"
        num = f"{abs(eps):.2f}"
    else:
        sign = "$"
        num = f"{eps:.2f}"
    return f"{sign:>{_W_DSIGN}}", f"{num:>{_W_EPS}}"


def _pct_cell(val: float | None, width: int) -> str:
    """Colored percentage in a fixed visible width. Negative zeros rounded out."""
    if val is None:
        return f"[dim]{'—':>{width}}[/]"
    raw = f"{val:+.1f}%"            # e.g. "+75.0%" — visible cells = len(raw)
    color = "green" if val > 0 else "#ff3232" if val < 0 else "dim"
    return f"[{color}]{raw:>{width}}[/]"


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

    # Header — visible widths mirror the row layout exactly.
    # EPS spans sign_col + 1 space + num_col so the header label "Last EPS"
    # is right-justified across that combined width.
    eps_hdr_width = _W_DSIGN + 1 + _W_EPS
    hdr = (
        f"  [bold #00c8ff]{'Symbol':<{_W_SYM}} "
        f"{'Last EPS':>{eps_hdr_width}} "
        f"{'Surprise':>{_W_SURP}} "
        f"{'Move':>{_W_MOVE}} "
        f"{'Streak':>{_W_STRK}} "
        f"{'Beat %':>{_W_RATE}} {'':<{_W_DETL}} "
        f"{'Avg Move':>{_W_AVG}}[/]"
    )
    lines = [hdr]
    total_w = _W_SYM + 1 + eps_hdr_width + 1 + _W_SURP + 1 + _W_MOVE + 1 + _W_STRK + 1 + _W_RATE + 1 + _W_DETL + 1 + _W_AVG
    lines.append(f"  [dim]{'─' * total_w}[/]")

    # Sort by beat rate descending, then by symbol
    sorted_syms = sorted(
        symbols.items(),
        key=lambda x: (x[1]["summary"].get("beat_rate") or 0, x[0]),
        reverse=True,
    )

    for sym, info in sorted_syms:
        s = info["summary"]

        dsign, eps_num = _eps_cells(s.get("last_eps"))
        surp = _pct_cell(s.get("last_surprise"), _W_SURP)
        move = _pct_cell(s.get("last_move"), _W_MOVE)

        streak = s.get("beat_streak", 0)
        strk_raw = f"{streak}"
        strk = (f"[green]{strk_raw:>{_W_STRK}}[/]" if streak > 0
                else f"[dim]{strk_raw:>{_W_STRK}}[/]")

        # Beat % + detail — treat as two adjacent fixed-width columns.
        if s.get("beat_rate") is not None:
            br = s["beat_rate"] * 100
            bc = "green" if br >= 75 else "#ffc800" if br >= 50 else "#ff3232"
            rate_raw = f"{br:.0f}%"
            rate = f"[{bc}]{rate_raw:>{_W_RATE}}[/]"
            detl_raw = f"({s['beats']}/{s['total']})"
            detl = f"[dim]{detl_raw:<{_W_DETL}}[/]"
        else:
            rate = f"[dim]{'—':>{_W_RATE}}[/]"
            detl = f"[dim]{'':<{_W_DETL}}[/]"

        avg = _pct_cell(s.get("avg_move"), _W_AVG)

        lines.append(
            f"  [bold #ffc800]{sym:<{_W_SYM}}[/] "
            f"[dim]{dsign}[/] [white]{eps_num}[/] "
            f"{surp} {move} {strk} "
            f"{rate} {detl} "
            f"{avg}"
        )

    # Watchlist summary
    ws = data.get("watchlist_summary", {})
    lines.append(f"\n  [dim]{'─' * total_w}[/]")
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
