"""Backtest / thesis-replay screen — book equity curve vs benchmark.

Mirrors timeline.py's ASCII-chart idiom, but overlays two series (the book and
a benchmark buy-and-hold, normalized to the same start) and annotates entry
(▲) / exit (▼) fills along the x-axis. The gap between the curves is the alpha.

Pure formatter: takes a BacktestResult, returns Rich markup. No I/O.
"""

from datetime import date

from i18n import t
from formatters import NEG, ACC

from backtest import BacktestResult


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1f}%" if v is not None else "N/A"


def _overlay_chart(
    dates: list[date],
    book: list[float],
    bench: list[float],
    width: int = 58,
    rows: int = 10,
) -> list[str]:
    """Two-series line chart in ASCII. Book is drawn solid (█/▀▄), benchmark as
    a dotted overlay (·). Both share one y-scale so their gap is readable.
    """
    all_vals = book + bench
    lo, hi = min(all_vals), max(all_vals)
    spread = hi - lo if hi != lo else 1.0

    def sample(series: list[float]) -> list[float]:
        if not series:
            return []
        if len(series) <= width:
            return series
        step = len(series) / width
        return [series[int(i * step)] for i in range(width)]

    s_book = sample(book)
    s_bench = sample(bench)
    n = len(s_book)

    def row_of(v: float) -> int:
        # 0 = top row, rows-1 = bottom
        norm = (v - lo) / spread
        return rows - 1 - min(rows - 1, int(norm * rows))

    book_rows = [row_of(v) for v in s_book]
    bench_rows = [row_of(v) for v in s_bench] if s_bench else []

    book_color = "green" if s_book and s_book[-1] >= s_book[0] else NEG

    lines: list[str] = []
    for r in range(rows):
        cells = []
        for c in range(n):
            ch = " "
            # benchmark dotted line first (so book overdraws it on overlap)
            if bench_rows and c < len(bench_rows) and bench_rows[c] == r:
                ch = f"[dim]·[/]"
            if book_rows[c] == r:
                ch = f"[{book_color}]█[/]"
            cells.append(ch)
        # y-axis labels: top=hi, mid, bottom=lo
        if r == 0:
            label = f" [dim]${hi:,.0f}[/]"
        elif r == rows - 1:
            label = f" [dim]${lo:,.0f}[/]"
        elif r == rows // 2:
            label = f" [dim]${(hi + lo) / 2:,.0f}[/]"
        else:
            label = ""
        lines.append("  " + "".join(cells) + label)
    return lines


def _mark_axis(dates: list[date], marks, width: int = 58) -> str:
    """A row under the chart placing ▲ (buy) / ▼ (sell) at each fill's column."""
    if not dates or not marks:
        return ""
    n = min(len(dates), width)
    # map each date to a chart column
    def col_for(d: date) -> int | None:
        if d not in dates:
            return None
        idx = dates.index(d)
        if len(dates) <= width:
            return idx
        return min(width - 1, int(idx / (len(dates) / width)))

    slots = [" "] * n
    for m in marks:
        c = col_for(m.date)
        if c is None or c >= n:
            continue
        glyph = "▲" if m.side == "BUY" else "▼"
        color = "green" if m.side == "BUY" else NEG
        slots[c] = f"[{color}]{glyph}[/]"
    return "  " + "".join(slots)


def format_backtest(result: BacktestResult, benchmark_label: str = "SOXX") -> str:
    """Render a BacktestResult as an ASCII replay chart + stats block."""
    if not result.dates or not result.book_curve:
        return f"[dim]{t('backtest.empty')}[/]"

    book = [float(v) for v in result.book_curve]
    bench = [float(v) for v in result.benchmark_curve]

    lines: list[str] = []
    lines.extend(_overlay_chart(result.dates, book, bench))
    axis = _mark_axis(result.dates, result.marks)
    if axis:
        lines.append(axis)

    # x-axis date range
    first = result.dates[0].isoformat()
    last = result.dates[-1].isoformat()
    lines.append(f"  [dim]{first}{' ' * max(58 - len(first) - len(last), 1)}{last}[/]")
    lines.append("")

    # legend
    lines.append(
        f"  [green]█[/] [dim]{t('backtest.book')}[/]   "
        f"[dim]·[/] [dim]{benchmark_label} {t('backtest.buyhold')}[/]"
    )
    lines.append("")

    # stats block
    s = result.stats
    book_ret = _fmt_pct(s.book_return_pct)
    book_color = "green" if s.book_return_pct >= 0 else NEG
    stats = [f"[bold white]{t('backtest.book')}[/] [{book_color}]{book_ret}[/]"]

    if s.benchmark_return_pct is not None:
        b_color = "green" if s.benchmark_return_pct >= 0 else NEG
        stats.append(
            f"[bold white]{benchmark_label}[/] [{b_color}]{_fmt_pct(s.benchmark_return_pct)}[/]"
        )
    else:
        stats.append(f"[bold white]{benchmark_label}[/] [dim]N/A[/]")

    if s.alpha_pct is not None:
        a_color = "green" if s.alpha_pct >= 0 else NEG
        stats.append(f"[bold white]{t('backtest.alpha')}[/] [{a_color}]{_fmt_pct(s.alpha_pct)}[/]")
    else:
        stats.append(f"[bold white]{t('backtest.alpha')}[/] [dim]N/A[/]")

    lines.append("  " + "  │  ".join(stats))

    dd_color = NEG if s.max_drawdown_pct < -5 else ACC
    lines.append(
        f"  [bold white]{t('backtest.maxdd')}[/] [{dd_color}]{s.max_drawdown_pct:+.1f}%[/]"
        f"   [dim]{len(result.dates)} {t('backtest.days')}[/]"
    )

    # horizon honesty — say where the replay actually starts
    if result.horizon_start:
        lines.append(
            f"\n  [dim]{t('backtest.horizon')}: {result.horizon_start.isoformat()}[/]"
        )

    return "\n".join(lines)
