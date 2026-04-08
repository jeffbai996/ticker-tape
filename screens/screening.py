"""Screening table — fundamental + technical metrics for quick comparison."""

from formatters import color_pct, rsi_color, off_high, fmt_num
from i18n import t


def format_screening(
    quotes: list[dict],
    technicals: dict[str, dict] | None = None,
    infos: dict[str, dict] | None = None,
) -> str:
    """Format multi-symbol screening table with price, RSI, P/E, 52w%, volume ratio.

    Args:
        quotes: List of quote dicts from fetch_quotes.
        technicals: {symbol: {rsi, ...}} from fetch_technicals_batch.
        infos: {symbol: yfinance .info dict} from fetch_batch_info.
    """
    if not quotes:
        return f"[dim]{t('msg.no_screening')}[/]"

    technicals = technicals or {}
    infos = infos or {}

    # Header
    lines = [
        f" [bold]{'':6} {'Price':>9} {'Chg%':>7} {'RSI':>5}"
        f" {'P/E':>7} {'MCap':>8} {'52w%':>7} {'VolR':>6}[/]",
        f" [dim]{'─' * 60}[/]",
    ]

    # Sort by % change descending
    valid = [q for q in quotes if not q.get("error") and q["price"] != 0.0]
    valid.sort(key=lambda q: q.get("pct", 0), reverse=True)

    for q in valid:
        sym = q["symbol"]
        price = q["price"]
        pct = q.get("pct", 0)
        ta = technicals.get(sym, {})
        info = infos.get(sym, {})

        # Price
        pfmt = f"{price:>9.2f}" if price < 1000 else f"{price:>9.0f}"

        # Change %
        pct_color = "green" if pct >= 0 else "#ff3232"
        pct_s = f"[{pct_color}]{pct:>+6.1f}%[/]"

        # RSI
        rsi = ta.get("rsi")
        if rsi is not None:
            rc = rsi_color(rsi)
            rsi_s = f"[{rc}]{rsi:>5.0f}[/]"
        else:
            rsi_s = f"{'—':>5}"

        # P/E
        pe = info.get("trailingPE")
        if pe is not None:
            pe_s = f"{pe:>7.1f}" if pe < 1000 else f"{pe:>7.0f}"
        else:
            pe_s = f"{'—':>7}"

        # Market cap (abbreviated)
        mcap = info.get("marketCap")
        if mcap is not None:
            if mcap >= 1e12:
                mcap_s = f"{mcap / 1e12:>6.1f}T"
            elif mcap >= 1e9:
                mcap_s = f"{mcap / 1e9:>6.0f}B"
            elif mcap >= 1e6:
                mcap_s = f"{mcap / 1e6:>6.0f}M"
            else:
                mcap_s = f"{mcap:>7.0f}"
        else:
            mcap_s = f"{'—':>8}"

        # 52-week % off high
        oh = off_high(info) if info else None
        if oh is None:
            # Fallback: use technicals off_high if available
            oh = ta.get("off_high")
        if oh is not None:
            ohc = "green" if oh > -5 else "#ffc800" if oh > -20 else "#ff3232"
            oh_s = f"[{ohc}]{oh:>+6.1f}%[/]"
        else:
            oh_s = f"{'—':>7}"

        # Volume ratio
        vol = info.get("volume")
        avg_vol = info.get("averageVolume")
        if vol and avg_vol and avg_vol > 0:
            vr = vol / avg_vol
            vrc = "bold white" if vr > 2 else "white" if vr > 1 else "dim"
            vr_s = f"[{vrc}]{vr:>5.1f}x[/]"
        else:
            vr_s = f"{'—':>6}"

        lines.append(
            f" [bold #ffc800]{sym:<6}[/] {pfmt} {pct_s} {rsi_s}"
            f" {pe_s} {mcap_s} {oh_s} {vr_s}"
        )

    return "\n".join(lines) if lines else f"[dim]{t('msg.no_screening')}[/]"
