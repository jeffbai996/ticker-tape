"""Time-travel screen — the reconstructed book rendered AS OF a past date.

Pure formatter: takes a `timetravel.AsOfBook` (+ report currency) and returns
Rich markup. No I/O. The gap from live is made unmistakable by the banner the
app writes above this block and the persistent StatusBar / sidebar banners.
"""

from __future__ import annotations

from decimal import Decimal

from i18n import t
from formatters import NEG, ACC, INF, DIM_HEX

from timetravel import AsOfBook


def _money(v: Decimal | None, ccy: str, signed: bool = False) -> str:
    if v is None:
        return "[dim]N/A[/]"
    f = float(v)
    if signed:
        return f"{f:+,.0f} {ccy}"
    return f"{f:,.0f} {ccy}"


def _pnl(v: Decimal | None, ccy: str) -> str:
    if v is None:
        return "[dim]N/A[/]"
    c = "green" if v >= 0 else NEG
    return f"[{c}]{float(v):+,.0f} {ccy}[/]"


def format_asof_book(book: AsOfBook, ccy: str = "USD") -> str:
    """Render the as-of book: banner, positions table, realized/unrealized/total."""
    lines: list[str] = []

    # Data-quality caveats first (unmatched sells, FX fallback) — read before
    # trusting the numbers below.
    for w in book.warnings:
        lines.append(f"[dim yellow]⚠ {w}[/]")
    if book.warnings:
        lines.append("")

    lines.append(f"  [bold {INF}]{t('tt.positions')} {book.as_of.isoformat()}[/]")

    if not book.positions:
        lines.append(f"  [dim]{t('tt.no_positions')}[/]")
    else:
        # header
        lines.append(
            f"  [dim]{'SYM':<6}{'QTY':>8}{'AVG':>10}{'CLOSE':>10}"
            f"{'MKT VAL':>14}{'UNREAL':>14}[/]"
        )
        for p in book.positions:
            qty = f"{float(p.qty):,.0f}"
            avg = f"{float(p.avg_cost):,.2f}"
            close = f"{float(p.close):,.2f}" if p.close is not None else "—"
            mv = f"{float(p.market_value):,.0f}" if p.market_value is not None else "—"
            if p.unrealized is None:
                unreal = "[dim]—[/]"
            else:
                uc = "green" if p.unrealized >= 0 else NEG
                unreal = f"[{uc}]{float(p.unrealized):+,.0f}[/]"
            lines.append(
                f"  [bold {ACC}]{p.symbol:<6}[/]{qty:>8}{avg:>10}{close:>10}"
                f"{mv:>14}{unreal:>14}"
            )

    lines.append("")
    lines.append(f"  [dim]{'─' * 44}[/]")
    lines.append(f"  {t('tt.cost_basis'):<14} {_money(book.cost_basis, ccy)}")
    lines.append(f"  {t('tt.market_value'):<14} {_money(book.market_value, ccy)}")
    lines.append(f"  {t('tt.realized'):<14} {_pnl(book.realized_pnl, ccy)}")
    lines.append(f"  {t('tt.unrealized'):<14} {_pnl(book.unrealized_pnl, ccy)}")
    lines.append(f"  [bold]{t('tt.total_pnl'):<14}[/] {_pnl(book.total_pnl, ccy)}")
    lines.append("")
    lines.append(f"  [{DIM_HEX}]{t('tt.nav')}[/]")

    return "\n".join(lines)
