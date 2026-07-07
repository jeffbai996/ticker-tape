"""Time-travel replay — render the book AS OF any past date.

The TUI's live surfaces (watchlist quotes, positions, P&L) all show *now*.
Time-travel freezes the terminal to a chosen past date: watchlist quotes become
that day's closes, and the book (positions / cost basis / realized + unrealized
P&L) is reconstructed from the local fills ledger AS OF that date.

Two hard rules keep this honest:

  1. It reuses the backtest engine's `PositionBook` for the position walk, so
     the matched-sell / average-cost / principal-in-basis semantics are byte-
     identical to `bt` — no reimplementation, no drift.
  2. Money is `Decimal` throughout. A missing close carries the nearest prior
     one (weekend / holiday / feed gap), never a fabricated 0.

This module is pure logic + a thin data-fetch seam (`build_dataset`). The
Textual widgets consume `TimeTravelState`; nothing here touches the UI.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from backtest import Fill, PositionBook

log = logging.getLogger(__name__)


# ── per-date snapshot dataclasses ───────────────────────────────────────

@dataclass(frozen=True)
class PositionSnapshot:
    """One open position as of the replay date."""
    symbol: str
    qty: Decimal
    avg_cost: Decimal
    close: Decimal | None          # mark on/before the as-of date, None if unknown
    market_value: Decimal | None   # qty * close, None if close unknown
    unrealized: Decimal | None     # qty * (close - avg_cost), None if close unknown


@dataclass(frozen=True)
class AsOfBook:
    """The reconstructed book as of a date: open positions + realized P&L."""
    as_of: date
    positions: list[PositionSnapshot]
    realized_pnl: Decimal          # cumulative realized gain up to & including as_of
    cost_basis: Decimal            # cumulative cost of every buy ever (principal)
    market_value: Decimal | None   # sum of position market values (None if all unknown)
    unrealized_pnl: Decimal | None # sum of position unrealized (None if all unknown)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_pnl(self) -> Decimal | None:
        """Realized + unrealized. None only if unrealized is entirely unknown."""
        if self.unrealized_pnl is None:
            return None
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class TimeTravelState:
    """Everything a scrub session needs, fetched once on entry.

    `bars` and `usdcad` are the full dated series; per-date reconstruction and
    watchlist marks are derived from them with zero extra network I/O — the
    whole point is that ← / → are pure dict lookups, not refetches.
    """
    report_ccy: str
    fills: list[Fill]                          # already FX-normalized to report_ccy
    bars: dict[str, dict[date, Decimal]]       # ledger symbols: {sym: {date: close}}
    watch_bars: dict[str, dict[date, Decimal]] # watchlist symbols: {sym: {date: close}}
    min_date: date                             # earliest scrubbable date (first fill)
    max_date: date                             # latest scrubbable date (today)
    warnings: list[str] = field(default_factory=list)
    _book_cache: dict[date, AsOfBook] = field(default_factory=dict)
    _sorted_dates: dict[str, list[date]] = field(default_factory=dict)


# ── nearest-close carry-back ────────────────────────────────────────────

def price_asof(series: dict[date, Decimal], as_of: date,
               _sorted: list[date] | None = None) -> Decimal | None:
    """Close on `as_of`, else the nearest close strictly before it.

    Weekend / holiday / feed gap → carry the last real close back. No close at
    or before `as_of` (the symbol didn't trade yet) → None, never 0.
    `_sorted` is an optional pre-sorted key list for O(log n) repeated lookups.
    """
    if not series:
        return None
    if as_of in series:
        return series[as_of]
    keys = _sorted if _sorted is not None else sorted(series)
    idx = bisect_right(keys, as_of)
    if idx == 0:
        return None
    return series[keys[idx - 1]]


# ── as-of reconstruction (reuses the backtest engine) ───────────────────

def reconstruct_as_of(
    fills: list[Fill],
    as_of: date,
    bars: dict[str, dict[date, Decimal]],
    sorted_keys: dict[str, list[date]] | None = None,
) -> AsOfBook:
    """Reconstruct the book AS OF `as_of` from the fills ledger.

    Every fill dated on/before `as_of` is applied through a `PositionBook`
    (same class the equity curve uses), then open positions are marked to each
    symbol's nearest close at-or-before `as_of`. Realized P&L is cumulative up
    to that date; positions closed out before it simply don't appear.

    Args:
        fills: FX-normalized fills (any order; applied in date order).
        as_of: the replay date.
        bars: {symbol: {date: close}} daily closes.
        sorted_keys: optional {symbol: sorted date list} to speed carry-back.
    """
    book = PositionBook()
    for f in sorted(fills, key=lambda f: (f.date, f.symbol)):
        if f.date <= as_of:
            book.apply(f)

    sorted_keys = sorted_keys or {}
    positions: list[PositionSnapshot] = []
    mv_total = Decimal("0")
    unreal_total = Decimal("0")
    any_marked = False

    for sym in sorted(book.qty):
        held = book.qty[sym]
        if held == 0:
            continue
        avg = book.avg_cost[sym]
        close = price_asof(bars.get(sym, {}), as_of, sorted_keys.get(sym))
        if close is None:
            mv = None
            unreal = None
        else:
            mv = held * close
            unreal = held * (close - avg)
            mv_total += mv
            unreal_total += unreal
            any_marked = True
        positions.append(PositionSnapshot(sym, held, avg, close, mv, unreal))

    return AsOfBook(
        as_of=as_of,
        positions=positions,
        realized_pnl=book.realized_gains,
        cost_basis=book.cumulative_basis,
        market_value=mv_total if any_marked else None,
        unrealized_pnl=unreal_total if any_marked else None,
        warnings=list(book.warnings),
    )


# ── scrub-session helpers (cached) ──────────────────────────────────────

def _sorted_for(state: TimeTravelState, bars: dict[str, dict[date, Decimal]]) -> None:
    """Populate state._sorted_dates for every symbol in `bars` (memoized)."""
    for sym, series in bars.items():
        if sym not in state._sorted_dates:
            state._sorted_dates[sym] = sorted(series)


def book_asof(state: TimeTravelState, as_of: date) -> AsOfBook:
    """Cached per-date reconstruction — the scrubber calls this every keypress."""
    cached = state._book_cache.get(as_of)
    if cached is not None:
        return cached
    _sorted_for(state, state.bars)
    book = reconstruct_as_of(state.fills, as_of, state.bars, state._sorted_dates)
    state._book_cache[as_of] = book
    return book


def watch_quotes_asof(state: TimeTravelState, as_of: date) -> dict[str, Decimal | None]:
    """{symbol: close-as-of} for the watchlist, nearest close carried back."""
    _sorted_for(state, state.watch_bars)
    return {
        sym: price_asof(series, as_of, state._sorted_dates.get(sym))
        for sym, series in state.watch_bars.items()
    }


def clamp_date(state: TimeTravelState, target: date) -> date:
    """Clamp `target` into [min_date, max_date] — scrubbing can't leave range."""
    if target < state.min_date:
        return state.min_date
    if target > state.max_date:
        return state.max_date
    return target


# ── data assembly (the one network fetch per session) ───────────────────

def build_dataset(
    watchlist_symbols: list[str],
    report_ccy: str | None = None,
    end: date | None = None,
) -> TimeTravelState | None:
    """Fetch and assemble everything a scrub session needs, once.

    Loads the fills ledger, fetches dated closes for every ledger + watchlist
    symbol over [first_fill, today], and FX-normalizes the ledger + bars to the
    report currency — mirroring the `bt` pipeline exactly so the book matches.

    Returns None if the ledger is empty (nothing to replay). Network failures
    degrade gracefully: bars that fail to fetch just yield unknown marks, not a
    crash. Demo mode is fully supported (deterministic, no network).
    """
    import config
    import backtest_data
    import backtest_fx

    report_ccy = report_ccy or getattr(config, "BACKTEST_CCY", "USD")
    if report_ccy not in ("USD", "CAD"):
        report_ccy = "USD"

    # ── fills ledger (demo seam mirrors backtest) ──
    if config.DEMO_MODE:
        import demo_data
        ledger = [backtest_data.LedgerFill(f, "USD")
                  for f in demo_data.backtest_fills()]
    else:
        ledger = backtest_data.load_fills_ledger(config.data_path("fills.csv"))
    if not ledger:
        return None

    ccy_by_symbol = backtest_fx.currency_by_symbol(ledger)
    ledger_symbols = sorted(ccy_by_symbol)
    start = ledger[0].fill.date
    end = end or date.today()

    warnings: list[str] = []
    try:
        ledger_bars = backtest_data.fetch_dated_closes(ledger_symbols, start, end)
    except Exception as e:  # never crash the session on a fetch hiccup
        log.warning("time-travel ledger bars fetch failed: %s", e)
        ledger_bars = {}

    # Watchlist symbols the ledger doesn't already cover
    extra_watch = sorted(set(watchlist_symbols) - set(ledger_symbols))
    try:
        watch_bars = backtest_data.fetch_dated_closes(extra_watch, start, end) if extra_watch else {}
    except Exception as e:
        log.warning("time-travel watchlist bars fetch failed: %s", e)
        watch_bars = {}

    # ── FX normalization (same contract as bt) ──
    watch_ccy = {s: backtest_fx.symbol_currency(s) for s in watchlist_symbols}
    needs_watch_fx = any(c != report_ccy for c in watch_ccy.values())
    if backtest_fx.needs_fx(ledger, report_ccy, report_ccy) or needs_watch_fx:
        try:
            usdcad = backtest_fx.fetch_usdcad(start, end)
            fills = backtest_fx.convert_fills(ledger, report_ccy, usdcad, warnings=warnings)
            ledger_bars = backtest_fx.convert_bars(ledger_bars, ccy_by_symbol, report_ccy, usdcad)
            watch_bars = backtest_fx.convert_bars(watch_bars, watch_ccy, report_ccy, usdcad)
        except ValueError as e:
            # No FX data when it was needed — fall back to raw fills rather than
            # crash; flag it so the numbers aren't trusted as currency-clean.
            log.warning("time-travel FX conversion failed: %s", e)
            warnings.append(f"FX unavailable ({e}) — amounts shown in native currency")
            fills = [lf.fill for lf in ledger]
    else:
        fills = [lf.fill for lf in ledger]

    # Merge ledger bars into watch_bars so a watchlist symbol that's also in the
    # ledger still gets marked (single source per symbol, ledger currency wins).
    merged_watch = dict(watch_bars)
    for sym in watchlist_symbols:
        if sym in ledger_bars:
            merged_watch[sym] = ledger_bars[sym]

    return TimeTravelState(
        report_ccy=report_ccy,
        fills=fills,
        bars=ledger_bars,
        watch_bars=merged_watch,
        min_date=start,
        max_date=end,
        warnings=warnings,
    )


def banner_text(as_of: date) -> str:
    """The unmistakable time-travel banner shown on every surface."""
    try:
        from i18n import t
        return "⏪ " + t("tt.banner").format(d=as_of.isoformat())
    except Exception:
        return f"⏪ AS OF {as_of.isoformat()} — time travel (Home = live)"
