"""Position sizing / what-if screen — margin impact + concentration analysis."""

import re

from i18n import t


def _parse_kv(raw: str) -> dict[str, str]:
    """Parse key: value lines from raw IBKR response."""
    result = {}
    for line in raw.strip().splitlines():
        # Strip markdown bold/headers
        line = re.sub(r"^#{1,4}\s+", "", line).replace("**", "").strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        result[key.strip().lower().replace(" ", "_")] = val.strip()
    return result


def _extract_float(s: str) -> float | None:
    """Extract numeric value from a string like '$1,234.56' or '12.5%'."""
    cleaned = re.sub(r"[$,%]", "", s.split()[0]) if s else ""
    try:
        return float(cleaned)
    except (ValueError, IndexError):
        return None


def format_sizing(
    whatif_raw: str | None,
    positions_raw: str | None,
    summary_raw: str | None,
    symbol: str,
    quantity: int,
) -> str:
    """Format pre-trade what-if analysis with concentration context.

    Combines IBKR what-if margin data with position concentration analysis.
    """
    lines = []

    # ── Section 1: Trade Summary ──
    lines.append(f"  [bold white]{symbol}[/]  [dim]×[/] [white]{quantity:,}[/] {t('sizing.shares')}\n")

    # ── Section 2: Margin Impact (from ibkr_what_if) ──
    if whatif_raw:
        lines.append(f"  [bold #00c8ff]{t('sizing.margin_impact')}[/]")
        kv = _parse_kv(whatif_raw)
        for key_pattern, label in [
            ("init", t("sizing.init_margin")),
            ("maint", t("sizing.maint_margin")),
            ("equity_with_loan", t("sizing.equity_loan")),
            ("cushion", t("sizing.cushion_label")),
            ("change", t("sizing.margin_change")),
            ("buying_power", t("sizing.buying_power")),
        ]:
            for k, v in kv.items():
                if key_pattern in k and v != "N/A":
                    lines.append(f"    [dim]{label:<16}[/] [white]{v}[/]")
                    break
        lines.append("")
    else:
        lines.append(f"  [dim]{t('sizing.whatif_unavail')}[/]\n")

    # ── Section 3: Concentration (from positions + summary) ──
    if positions_raw and summary_raw:
        lines.append(f"  [bold #00c8ff]{t('sizing.concentration')}[/]")
        summary_kv = _parse_kv(summary_raw)

        # Extract NLV
        nlv = None
        for k, v in summary_kv.items():
            if "net_liquidation" in k or "nlv" in k:
                nlv = _extract_float(v)
                if nlv:
                    break

        # Estimate trade value from what-if or fallback
        # Parse existing positions to find current weight of this symbol
        existing_weight = None
        pos_lines = positions_raw.strip().splitlines()
        for pl in pos_lines:
            if symbol in pl.upper():
                # Try to extract weight from Wt% column
                wt_match = re.search(r"(\d+\.?\d*)%", pl)
                if wt_match:
                    existing_weight = float(wt_match.group(1))
                    break

        if nlv and nlv > 0:
            lines.append(f"    [dim]{t('sizing.nlv'):<16}[/] [white]${nlv:,.0f}[/]")
            if existing_weight is not None:
                lines.append(f"    [dim]{t('sizing.current_wt'):<16}[/] [white]{existing_weight:.1f}%[/]")
        lines.append("")

    # ── Section 4: Cushion Before/After ──
    if whatif_raw:
        kv = _parse_kv(whatif_raw)
        before = None
        after = None
        for k, v in kv.items():
            if "before" in k and "cushion" in k:
                before = _extract_float(v)
            elif "after" in k and "cushion" in k:
                after = _extract_float(v)
            elif "current" in k and "cushion" in k:
                before = _extract_float(v)
            elif k == "cushion" and before is None:
                # Might be the "after" value
                after = _extract_float(v)

        if before is not None or after is not None:
            lines.append(f"  [bold #00c8ff]{t('sizing.cushion_impact')}[/]")
            if before is not None:
                bc = "#ff3232" if before < 10 else "#ffc800" if before < 15 else "green"
                lines.append(f"    [dim]{t('sizing.before'):<16}[/] [{bc}]{before:.1f}%[/]")
            if after is not None:
                ac = "#ff3232" if after < 10 else "#ffc800" if after < 15 else "green"
                lines.append(f"    [dim]{t('sizing.after'):<16}[/] [{ac}]{after:.1f}%[/]")

    return "\n".join(lines)
