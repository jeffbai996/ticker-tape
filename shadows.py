"""Shadow books — standing counterfactual ledgers replayed against the real one.

A shadow is an alternate fills ledger ("never trimmed", "kept the hedge", "50%
ballast") living in data/shadows/<name>.csv, same schema as data/fills.csv. Each
is replayed through the SAME backtest engine (backtest.assemble_backtest via
PositionBook) over the SAME price bars as the real book, so the only difference
in the resulting equity curve is the decision encoded in the fills — the running
dollar cost of every major choice, made visible.

Pure logic only (no Textual, no network here): the app fetches bars once and
hands them in; this module loads shadow ledgers, replays them, and computes the
delta vs the real book. Money is Decimal end-to-end because it IS the backtest
engine — we call it, we don't reimplement matched-sell/avg-cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import backtest_data
from backtest import assemble_backtest

SHADOWS_DIR = os.path.join("data", "shadows")


@dataclass(frozen=True)
class ShadowResult:
    """One shadow's replay outcome relative to the real book."""
    name: str
    final_value: Decimal | None      # shadow's final equity, or None if empty
    real_final: Decimal | None       # real book's final equity (same for all)
    delta: Decimal | None            # shadow_final - real_final
    delta_pct: float | None          # delta as % of real_final
    error: str | None = None         # set when this shadow couldn't be replayed


def list_shadow_files(shadows_dir: str = SHADOWS_DIR) -> list[str]:
    """Sorted list of shadow ledger paths. Missing dir → []."""
    if not os.path.isdir(shadows_dir):
        return []
    return sorted(
        os.path.join(shadows_dir, f)
        for f in os.listdir(shadows_dir)
        if f.endswith(".csv")
    )


def shadow_name(path: str) -> str:
    """Human name for a shadow file: 'never_trimmed.csv' → 'never trimmed'."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace("_", " ")


def replay_shadow(
    fills: list,
    bars: dict[str, dict[date, Decimal]],
    benchmark: dict[date, Decimal],
) -> Decimal | None:
    """Final equity value for a set of fills over the given bars. None if the
    replay is empty/degenerate. Thin wrapper over the real engine so a shadow
    and the real book are computed identically."""
    result = assemble_backtest(fills, bars, benchmark)
    if not result.book_curve:
        return None
    return result.book_curve[-1]


def compare_shadows(
    real_final: Decimal | None,
    shadow_ledgers: list[tuple[str, list]],
    bars: dict[str, dict[date, Decimal]],
    benchmark: dict[date, Decimal],
) -> list[ShadowResult]:
    """Replay each (name, fills) shadow and delta it against real_final.

    A shadow whose replay raises or comes back empty yields a ShadowResult with
    an `error`/None delta rather than sinking the whole comparison.
    """
    out: list[ShadowResult] = []
    for name, fills in shadow_ledgers:
        try:
            final = replay_shadow(fills, bars, benchmark)
        except Exception as e:  # a malformed shadow must not kill the panel
            out.append(ShadowResult(name, None, real_final, None, None,
                                    error=f"replay failed: {e}"))
            continue
        if final is None or real_final is None:
            out.append(ShadowResult(name, final, real_final, None, None,
                                    error="empty replay" if final is None else None))
            continue
        delta = final - real_final
        delta_pct = float(delta / real_final * 100) if real_final != 0 else None
        out.append(ShadowResult(name, final, real_final, delta, delta_pct))
    return out


def load_shadow_fills(path: str) -> list:
    """Load a shadow ledger's fills (Fill objects). Reuses the real ledger
    parser so schema/validation are identical. Missing/empty → []."""
    ledger = backtest_data.load_fills_ledger(path)
    return [lf.fill for lf in ledger]


def decision_cost_line(r: ShadowResult) -> str:
    """One-line framing of a shadow's delta as the cost/benefit of the decision.

    Positive delta → the counterfactual would have done better (the real
    decision cost you); negative → the real decision was the better call.
    Returns plain text (caller colorizes)."""
    if r.error:
        return f"{r.name}: {r.error}"
    if r.delta is None:
        return f"{r.name}: no comparable data"
    amt = _money(abs(r.delta))
    pct = f" ({r.delta_pct:+.1f}%)" if r.delta_pct is not None else ""
    if r.delta > 0:
        return f"{r.name}: +{amt}{pct} vs real — this path would have done better"
    if r.delta < 0:
        return f"{r.name}: -{amt}{pct} vs real — the real book was the better call"
    return f"{r.name}: flat vs real — no difference"


def _money(value: Decimal) -> str:
    v = abs(value)
    if v >= Decimal("1000000"):
        return f"${v / Decimal('1000000'):.2f}M"
    if v >= Decimal("1000"):
        return f"${v / Decimal('1000'):.1f}K"
    return f"${v:,.0f}"
