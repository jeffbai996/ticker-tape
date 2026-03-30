"""Options chain — pure formatting function."""

import statistics

from formatters import fmt_num
from i18n import t


def _fmt_leg(
    rows: list[dict], label_key: str, is_puts: bool, lines: list[str]
) -> None:
    """Format one leg (calls or puts) and append to lines."""
    lines.append(f"  [bold]{t(label_key)}[/]")

    # Column headers
    hdr = (
        f"  {t('options.strike'):>8}  {t('options.bid'):>7}  "
        f"{t('options.ask'):>7}  {t('options.last'):>7}  "
        f"{t('options.volume'):>7}  {t('options.oi'):>7}  "
        f"{t('options.iv'):>6}"
    )
    lines.append(f"  [bold]{hdr}[/]")

    if not rows:
        return

    # Median IV for high-IV highlighting
    ivs = [r["iv"] for r in rows if r.get("iv")]
    med_iv = statistics.median(ivs) if ivs else 0

    for r in rows:
        strike = f"{r['strike']:8.2f}"
        bid = f"{r['bid']:7.2f}" if r["bid"] else "      —"
        ask = f"{r['ask']:7.2f}" if r["ask"] else "      —"
        last = f"{r['last']:7.2f}" if r["last"] else "      —"
        vol = fmt_num(r.get("volume")).rjust(7)
        oi = fmt_num(r.get("open_interest")).rjust(7)

        # IV with optional amber highlight
        iv_val = r.get("iv")
        if iv_val is not None:
            iv_str = f"{iv_val * 100:5.1f}%"
            if med_iv > 0 and iv_val > med_iv * 1.5:
                iv_str = f"[#ffc800]{iv_str}[/]"
        else:
            iv_str = "    —"

        row = f"  {strike}  {bid}  {ask}  {last}  {vol}  {oi}  {iv_str}"

        # Tags
        moneyness = r.get("moneyness", 0)
        if abs(moneyness) < 2:
            row += f"  [bold #ffc800]{t('options.atm')}[/]"
        elif r.get("itm"):
            row += f"  [dim]{t('options.itm')}[/]"
        elif is_puts:
            row += f"  [bold #c864ff]◆ {t('options.hedge')}[/]"

        lines.append(row)


def format_options(data: dict | None, symbol: str) -> str:
    """Format options chain data for Rich rendering."""
    if data is None:
        return f"[#ff3232]{t('msg.no_options').format(sym=symbol)}[/]"

    lines: list[str] = []

    # Header: symbol  $price  Exp: date  (N expirations)
    price = data.get("current_price", 0)
    exp = data.get("selected_expiration", "—")
    n_exp = len(data.get("expirations", []))
    lines.append(
        f"  [bold]{symbol}[/]  ${price:.2f}  "
        f"{t('options.exp')}: {exp}  ({n_exp} {t('options.avail')})"
    )
    lines.append(f"  [dim]{'─' * 60}[/]")

    # Calls
    _fmt_leg(data.get("calls", []), "options.calls", False, lines)

    lines.append(f"  [dim]{'─' * 60}[/]")

    # Puts
    _fmt_leg(data.get("puts", []), "options.puts", True, lines)

    # Expiration picker hint
    exps = data.get("expirations", [])
    if len(exps) > 1:
        others = [e for e in exps if e != exp][:6]
        lines.append(f"\n  [dim]{t('options.avail')}: {', '.join(others)}[/]")

    return "\n".join(lines)
