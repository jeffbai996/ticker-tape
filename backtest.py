"""Backtest / thesis-replay core — realized equity-curve reconstruction.

Surface-agnostic: pure `(fills, bars, benchmark) -> BacktestResult`. No I/O,
no rendering. The CLI screen (and later a web view) consume this.

What it answers: "what did the book actually do vs the benchmark since the
first fill, marked at every real entry and exit." A factual replay of the
book's own history — not a strategy optimizer.

Money is `Decimal` throughout (house rule; float subtraction lies about P&L).
Missing data never fabricates a value: an absent price carries the last known
one, an absent benchmark yields None (not a fake 0), an empty book yields an
empty result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Fill:
    """One realized execution. side is "BUY" or "SELL"; qty/price positive."""
    date: date
    symbol: str
    side: str
    qty: Decimal
    price: Decimal


@dataclass(frozen=True)
class Mark:
    """An entry (▲, BUY) or exit (▼, SELL) annotation on the curve."""
    date: date
    symbol: str
    side: str
    qty: Decimal
    price: Decimal


@dataclass(frozen=True)
class BacktestStats:
    book_return_pct: float
    benchmark_return_pct: float | None
    alpha_pct: float | None
    max_drawdown_pct: float


@dataclass(frozen=True)
class BacktestResult:
    dates: list[date]
    book_curve: list[Decimal]
    benchmark_curve: list[Decimal]
    marks: list[Mark]
    stats: BacktestStats | None
    horizon_start: date | None


def _sorted_trading_days(
    fills: list[Fill],
    bars: dict[str, dict[date, Decimal]],
    benchmark: dict[date, Decimal],
    start: date,
) -> list[date]:
    """All market days >= start: the union of every price series, the
    benchmark's own dates (it's an index — its calendar is the market's), and
    fill dates (so a same-day entry/exit gets a point even if that symbol has
    no separate bar that day). A held symbol missing a bar on one of these
    days is carried at its last-known price, not dropped.
    """
    days: set[date] = set()
    for series in bars.values():
        days.update(d for d in series if d >= start)
    days.update(d for d in benchmark if d >= start)
    days.update(f.date for f in fills if f.date >= start)
    return sorted(days)


def _price_on(series: dict[date, Decimal], day: date, last: Decimal | None) -> Decimal | None:
    """Close for `day`, else the carried last-known price (never 0 from a gap)."""
    return series.get(day, last)


def assemble_backtest(
    fills: list[Fill],
    bars: dict[str, dict[date, Decimal]],
    benchmark: dict[date, Decimal],
) -> BacktestResult:
    """Reconstruct the realized equity curve and compare to a benchmark.

    Args:
        fills: realized executions (any order; sorted internally).
        bars: {symbol: {date: close}} — daily closes per held symbol.
        benchmark: {date: close} — the comparison index (e.g. SOXX).

    Returns a BacktestResult; empty/degenerate inputs yield an empty result
    rather than raising or inventing numbers.
    """
    if not fills:
        return BacktestResult([], [], [], [], None, None)

    fills = sorted(fills, key=lambda f: (f.date, f.symbol))
    horizon_start = fills[0].date
    days = _sorted_trading_days(fills, bars, benchmark, horizon_start)
    if not days:
        return BacktestResult([], [], [], [], None, None)

    marks = [Mark(f.date, f.symbol, f.side, f.qty, f.price) for f in fills]

    # Equity model (thesis-replay, account-value convention):
    #     equity(day) = cost_basis_of_all_buys + realized_gains + unrealized_gains
    # The deployed principal stays "in the book" across a round-trip — a sale
    # converts unrealized gain to realized but doesn't remove the capital, so
    # the curve reads as account value, not just P&L. Realized/unrealized use
    # average cost so partial exits are exact.
    #
    # No look-ahead: a fill on day T is applied only when the walk reaches T,
    # so days before the first buy show equity 0.
    fills_by_day: dict[date, list[Fill]] = {}
    for f in fills:
        fills_by_day.setdefault(f.date, []).append(f)

    qty: dict[str, Decimal] = {}          # open shares per symbol
    avg_cost: dict[str, Decimal] = {}     # average cost basis per open share
    cumulative_basis = Decimal("0")       # cost of every buy ever (the principal)
    realized_gains = Decimal("0")
    last_price: dict[str, Decimal] = {}

    book_curve: list[Decimal] = []
    for day in days:
        # Apply the day's fills before marking, so a same-day buy is held.
        for f in fills_by_day.get(day, []):
            held = qty.get(f.symbol, Decimal("0"))
            if f.side == "BUY":
                cumulative_basis += f.qty * f.price
                new_qty = held + f.qty
                prior_cost = avg_cost.get(f.symbol, Decimal("0")) * held
                avg_cost[f.symbol] = (prior_cost + f.qty * f.price) / new_qty
                qty[f.symbol] = new_qty
            else:  # SELL: realize gain vs average cost, principal stays in basis
                basis = avg_cost.get(f.symbol, f.price)
                realized_gains += f.qty * (f.price - basis)
                qty[f.symbol] = held - f.qty

        # Unrealized gain on open positions, marked to the day's close
        # (carry last-known price on a feed gap — never drop to 0).
        unrealized = Decimal("0")
        for sym, held in qty.items():
            if held == 0:
                continue
            px = _price_on(bars.get(sym, {}), day, last_price.get(sym))
            if px is None:
                continue
            last_price[sym] = px
            unrealized += held * (px - avg_cost[sym])

        book_curve.append(cumulative_basis + realized_gains + unrealized)

    benchmark_curve = _benchmark_curve(benchmark, days, book_curve[0])
    stats = _compute_stats(book_curve, benchmark_curve)
    return BacktestResult(days, book_curve, benchmark_curve, marks, stats, horizon_start)


def _benchmark_curve(
    benchmark: dict[date, Decimal], days: list[date], book_start: Decimal
) -> list[Decimal]:
    """Buy-and-hold the benchmark with the same starting capital as the book.

    Normalized so the first day WITH benchmark data equals book_start → the two
    curves share a y-axis and the gap between them IS the alpha. The benchmark
    feed may start a few days after the first fill (thin data) — rather than
    null the whole comparison, base off the first available price at-or-after
    the horizon and hold it flat for any leading days before coverage. Empty
    benchmark → empty curve (the caller reports None, never a fabricated line).
    """
    if not benchmark or not days:
        return []
    # First benchmark price on or after the replay's first day.
    base = None
    for day in days:
        if day in benchmark:
            base = benchmark[day]
            break
    if base is None or base == 0:
        return []
    curve: list[Decimal] = []
    last = base
    for day in days:
        px = benchmark.get(day, last)
        last = px
        curve.append(book_start * px / base)
    return curve


def _pct(start: Decimal, end: Decimal) -> float | None:
    if start == 0:
        return None
    return float((end - start) / start * 100)


def _compute_stats(
    book_curve: list[Decimal], benchmark_curve: list[Decimal]
) -> BacktestStats | None:
    if not book_curve:
        return None
    book_ret = _pct(book_curve[0], book_curve[-1])

    bench_ret = None
    if benchmark_curve:
        bench_ret = _pct(benchmark_curve[0], benchmark_curve[-1])

    alpha = None
    if book_ret is not None and bench_ret is not None:
        alpha = book_ret - bench_ret

    # Max drawdown on the book equity curve (trough vs running peak).
    peak = book_curve[0]
    max_dd = 0.0
    for v in book_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = float((v - peak) / peak * 100)
            if dd < max_dd:
                max_dd = dd

    return BacktestStats(
        book_return_pct=book_ret if book_ret is not None else 0.0,
        benchmark_return_pct=bench_ret,
        alpha_pct=alpha,
        max_drawdown_pct=max_dd,
    )
