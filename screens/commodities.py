"""Commodity futures/spot prices — pure formatting function."""

from formatters import POS, NEG


def _fmt_price(price: float) -> str:
    """Format commodity price with magnitude-appropriate decimal places."""
    if price >= 10000:
        return f"{price:,.0f}"
    if price >= 100:
        return f"{price:,.2f}"
    if price >= 10:
        return f"{price:.2f}"
    return f"{price:.4f}"


def format_commodities(data: dict) -> str:
    """Format commodity futures grouped by sector.

    Args:
        data: {group: [{symbol, name, unit, price, change, pct, stale?}, ...]}
    """
    if not data:
        return "[dim]No commodity data available[/]"

    lines = []
    for group_name, items in data.items():
        lines.append(f"\n[bold #00c8ff]{group_name}[/]")
        lines.append(f"[dim]{'─' * 58}[/]")
        for item in items:
            sym   = item["symbol"]
            name  = item["name"]
            unit  = item["unit"]
            if item.get("stale"):
                lines.append(
                    f"[bold #ffc800]{sym:<8}[/]"
                    f"[dim]{name:<15}[/]"
                    f"[dim]{'—':>12}[/]  "
                    f"[dim]{unit:<10}[/]  "
                    f"[dim]N/A[/]"
                )
            else:
                price_str = _fmt_price(item["price"])
                color = POS if item["pct"] >= 0 else NEG
                arrow = "▲" if item["pct"] >= 0 else "▼"
                lines.append(
                    f"[bold #ffc800]{sym:<8}[/]"
                    f"[dim]{name:<15}[/]"
                    f"[white]{price_str:>12}[/]  "
                    f"[dim]{unit:<10}[/]  "
                    f"[{color}]{arrow} {item['pct']:+.2f}%[/]"
                )
    return "\n".join(lines)
