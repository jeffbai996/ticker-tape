"""Decision cards — pre-computed playbooks attached to a fired alert.

An alert fire used to just notify and vanish. A decision card turns "something
crossed a line" into "here is the sized, read-only action to consider" so the
next step is a decision, not a research task.

Two card types, both pure functions returning Rich-markup strings (no I/O, no
Textual imports — unit-tested as plain string builders):

  * cushion card  → a TRIM LADDER: which positions to trim, and by how much, to
    restore a target margin cushion. Sized from live IBKR state when available;
    degrades to a clearly-labeled note when it isn't.
  * price/RSI/SMA/volume card → the symbol's standing in the book (held qty,
    cost basis, unrealized) plus recent context and a one-line review prompt.

NOTHING here places an order. The ladder is a suggestion the human executes (or
doesn't) by hand.

--- Trim-ladder policy (stated so the number is auditable) ---------------------
Cushion ≈ ExcessLiquidity / NLV. Trimming a position raises excess liquidity by
(trim_value × maint_margin_rate) — you free the maintenance margin the position
was tying up — while NLV is (to first order) unchanged by a trim at market. So
to lift cushion from `current` to `target` you need to free:

    needed_excess = (target - current) * NLV

Each dollar of a position trimmed frees `maint_rate` dollars of excess. Absent a
per-name maintenance rate from IBKR we assume a uniform `DEFAULT_MAINT_RATE`
(stated on the card). Trim the LARGEST positions first (fewest tickets, least
per-name disruption), never more than a position holds, until `needed_excess` is
covered. If the whole book can't cover it, say so rather than silently stopping.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# Matches a signed number with optional $, commas, decimals — the values in an
# ibkr-mcp positions table cell (e.g. "$18,414 USD", "-952").
_NUM_RE = re.compile(r"[+-]?\$?[\d,]+\.?\d*")


def parse_positions_markdown(text: str | None) -> list[dict]:
    """Extract [{symbol, mkt_value, qty, avg_cost, unrealized}] from an
    ibkr-mcp positions table. Best-effort and total: an unparseable row is
    skipped, a missing table returns []. Column layout expected (order matters):
    Symbol | Shares | Avg Cost | Mkt Price | Mkt Value | Unreal P&L | Weight ...
    """
    if not text:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        symbol = cells[0].strip().strip("*")
        # Skip header/separator rows and anything without an A-Z ticker.
        if not re.fullmatch(r"[A-Z][A-Z.\-]*", symbol):
            continue
        qty = _first_num(cells[1])
        avg_cost = _first_num(cells[2])
        mkt_value = _first_num(cells[4])
        unrealized = _first_num(cells[5])
        if mkt_value is None:
            continue
        out.append({
            "symbol": symbol,
            "qty": qty,
            "avg_cost": avg_cost,
            "mkt_value": mkt_value,
            "unrealized": unrealized,
        })
    return out


def _first_num(cell: str) -> Decimal | None:
    """First numeric token in a table cell as Decimal, else None."""
    m = _NUM_RE.search(cell or "")
    if not m:
        return None
    return _d(m.group(0).replace("$", "").replace(",", ""))

# Uniform maintenance-margin rate assumed when IBKR doesn't give a per-name one.
# 25% is the RegT long-equity maintenance floor; portfolio margin is usually
# lower (so this is conservative — it may UNDER-state how much a trim frees,
# suggesting a slightly larger trim than strictly necessary, which is the safe
# direction for a cushion restore).
DEFAULT_MAINT_RATE = Decimal("0.25")


def _d(value) -> Decimal | None:
    """Coerce to Decimal, or None if it can't be (NaN, junk, missing)."""
    if value is None:
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if d != d:  # NaN
        return None
    return d


def trim_ladder(
    positions: list[dict],
    nlv,
    current_cushion_pct,
    target_cushion_pct,
    maint_rate: Decimal = DEFAULT_MAINT_RATE,
) -> dict:
    """Compute the trim ladder to lift cushion from current to target.

    Args:
        positions: [{symbol, mkt_value}, ...] — market value per open position
            in the account (base currency). Only positive market values (long)
            are trimmable here.
        nlv: net liquidation value (account base currency).
        current_cushion_pct / target_cushion_pct: cushion as a PERCENT (e.g.
            10.5 for 10.5%), matching how the app reads IBKR cushion.
        maint_rate: fraction of trimmed value that becomes freed excess.

    Returns a dict:
        {
          "feasible": bool,          # can the book reach target at all?
          "needed_excess": Decimal,  # $ of excess liquidity to free
          "rungs": [{symbol, trim_value, freed_excess}, ...],  # in trim order
          "shortfall": Decimal,      # unfreed remainder if not feasible (>=0)
          "reason": str | None,      # set when nothing to do / bad inputs
        }
    """
    nlv_d = _d(nlv)
    cur = _d(current_cushion_pct)
    tgt = _d(target_cushion_pct)
    if nlv_d is None or cur is None or tgt is None or nlv_d <= 0:
        return {"feasible": False, "needed_excess": Decimal("0"), "rungs": [],
                "shortfall": Decimal("0"), "reason": "insufficient account data"}

    if cur >= tgt:
        return {"feasible": True, "needed_excess": Decimal("0"), "rungs": [],
                "shortfall": Decimal("0"),
                "reason": "cushion already at or above target"}

    needed_excess = (tgt - cur) / Decimal("100") * nlv_d

    # Largest positions first — fewest tickets to move the needle.
    longs = sorted(
        ((p.get("symbol", "?"), _d(p.get("mkt_value"))) for p in positions),
        key=lambda t: (t[1] or Decimal("0")),
        reverse=True,
    )

    rungs: list[dict] = []
    remaining = needed_excess
    for symbol, mv in longs:
        if mv is None or mv <= 0 or remaining <= 0:
            continue
        # Each $ trimmed frees maint_rate $ of excess.
        trim_value_to_close_gap = remaining / maint_rate
        trim_value = min(trim_value_to_close_gap, mv)
        freed = trim_value * maint_rate
        rungs.append({
            "symbol": symbol,
            "trim_value": trim_value,
            "freed_excess": freed,
        })
        remaining -= freed

    feasible = remaining <= Decimal("0.005")  # penny tolerance
    return {
        "feasible": feasible,
        "needed_excess": needed_excess,
        "rungs": rungs,
        "shortfall": max(remaining, Decimal("0")),
        "reason": None,
    }


def build_cushion_card(
    positions: list[dict] | None,
    nlv,
    current_cushion_pct,
    target_cushion_pct,
    maint_rate: Decimal = DEFAULT_MAINT_RATE,
) -> str:
    """Rich-markup decision card for a cushion alert.

    When positions/NLV aren't available (IBKR offline at fire time), returns a
    clearly-labeled static note instead of a fabricated ladder.
    """
    header = "[bold #ffc800]◈ Cushion decision card[/]"
    cur = _d(current_cushion_pct)
    tgt = _d(target_cushion_pct)
    if not positions or _d(nlv) is None or cur is None or tgt is None:
        return (
            f"{header}\n"
            f"[dim]Connect IBKR for a sized trim ladder. "
            f"Cushion is below your threshold — review margin before adding risk.[/]"
        )

    ladder = trim_ladder(positions, nlv, cur, tgt, maint_rate)
    if ladder["reason"] == "cushion already at or above target":
        return f"{header}\n[dim]Cushion already at/above {tgt:.1f}% target — no trim needed.[/]"

    lines = [
        header,
        f"[dim]Cushion {cur:.1f}% → target {tgt:.1f}%. "
        f"Free ~{_money(ladder['needed_excess'])} of excess "
        f"(assumes {maint_rate * 100:.0f}% maint rate). Read-only — trim by hand.[/]",
    ]
    for r in ladder["rungs"]:
        lines.append(
            f"  trim [bold]{r['symbol']}[/] ~{_money(r['trim_value'])} "
            f"[dim](frees ~{_money(r['freed_excess'])})[/]"
        )
    if not ladder["feasible"]:
        lines.append(
            f"  [#ff3232]⚠ trimming every long still leaves "
            f"~{_money(ladder['shortfall'])} short — cushion can't reach target by trimming alone.[/]"
        )
    return "\n".join(lines)


def build_price_card(alert: dict, position: dict | None) -> str:
    """Rich-markup decision card for a price / RSI / SMA / volume alert.

    `position` is {symbol, qty, avg_cost, unrealized} for the alerted symbol if
    held, else None.
    """
    sym = alert.get("symbol", "?")
    header = f"[bold #00c8ff]◈ {sym} decision card[/]"
    if position and _d(position.get("qty")) and _d(position.get("qty")) != 0:
        qty = _d(position.get("qty"))
        avg = _d(position.get("avg_cost"))
        unreal = _d(position.get("unrealized"))
        held = f"[dim]Held: {qty} @ {_money(avg)} avg"
        if unreal is not None:
            color = "green" if unreal >= 0 else "#ff3232"
            held += f" · unrealized [{color}]{_money(unreal)}[/]"
        held += ".[/]"
        prompt = "[dim]Review against thesis — is this a trim, an add, or noise?[/]"
        return f"{header}\n{held}\n{prompt}"
    return (
        f"{header}\n[dim]Not currently held. "
        f"Watchlist alert fired — review the setup before acting.[/]"
    )


def _money(value: Decimal | None) -> str:
    """Compact dollar formatting: $1.2M / $850K / $1,234."""
    if value is None:
        return "n/a"
    v = abs(value)
    sign = "-" if value < 0 else ""
    if v >= Decimal("1000000"):
        return f"{sign}${v / Decimal('1000000'):.1f}M"
    if v >= Decimal("1000"):
        return f"{sign}${v / Decimal('1000'):.0f}K"
    return f"{sign}${v:,.0f}"
