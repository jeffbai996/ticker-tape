"""Correlation matrix — pure formatting function."""

from i18n import t

MAX_SYMBOLS = 12


def format_correlation(data: dict | None) -> str:
    """Format a return correlation matrix with color-coded cells.

    Args:
        data: {symbols, matrix, period} or None if unavailable.
    """
    if data is None:
        return f"[dim]{t('msg.no_correlation')}[/]"

    symbols: list[str] = data["symbols"]
    matrix: list[list[float]] = data["matrix"]
    period: str = data["period"]

    # Truncate to MAX_SYMBOLS if needed
    truncated = len(symbols) > MAX_SYMBOLS
    total = len(symbols)
    if truncated:
        symbols = symbols[:MAX_SYMBOLS]
        matrix = [row[:MAX_SYMBOLS] for row in matrix[:MAX_SYMBOLS]]

    n = len(symbols)
    lines: list[str] = []

    # Title
    lines.append(f"[bold]{t('title.correlation')}[/]  ({period}, {n} symbols)")

    if truncated:
        lines.append(f"[dim]{t('corr.truncated').format(n=MAX_SYMBOLS, total=total)}[/]")

    # Column headers: left pad + right-aligned 6-char symbols
    header = " " * 5 + "".join(f"{s:>6}" for s in symbols)
    lines.append(header)

    # Data rows
    for i, sym in enumerate(symbols):
        row = f"{sym:<5}"
        for j in range(n):
            val = matrix[i][j]
            cell = f"{val:6.2f}"
            if i == j:
                cell = f"[dim]{cell}[/]"
            elif val > 0.8:
                cell = f"[bold green]{cell}[/]"
            elif val >= 0.4:
                cell = f"[white]{cell}[/]"
            elif val >= 0.0:
                cell = f"[dim]{cell}[/]"
            else:
                cell = f"[bold #ff3232]{cell}[/]"
            row += cell
        lines.append(row)

    # Average of all non-diagonal pairs
    pair_sum = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_sum += matrix[i][j]
                pair_count += 1
    avg = pair_sum / pair_count if pair_count else 0.0
    lines.append("")
    lines.append(f"{t('corr.avg')}: {avg:.2f}")

    # Legend
    lines.append(
        "[dim]Legend: [bold green]>0.8[/] strong  "
        "[white]0.4-0.8[/] moderate  "
        "[dim]<0.4[/] weak  "
        "[bold #ff3232]<0[/] inverse[/]"
    )

    return "\n".join(lines)
