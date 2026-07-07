"""Shadow-books screen — pure formatting of counterfactual replay results."""

from decimal import Decimal

from formatters import ACC, NEG
from i18n import t
from shadows import ShadowResult, decision_cost_line


def format_shadows(
    results: list[ShadowResult],
    real_final: Decimal | None,
    ccy: str = "USD",
) -> str:
    """Render the shadow comparison panel.

    `results` is the output of shadows.compare_shadows; `real_final` is the real
    book's final equity (shown once at the top as the baseline).
    """
    if not results:
        return (
            f"[dim]{t('shadows.empty')}[/]\n"
            f"[dim]{t('shadows.hint')}[/]"
        )

    lines = [f"[dim]{t('shadows.baseline')}: {_money(real_final, ccy)}[/]", ""]
    for r in results:
        cost = decision_cost_line(r)
        if r.error:
            lines.append(f"[dim]○ {r.name} — {r.error}[/]")
            continue
        if r.delta is None:
            lines.append(f"[dim]○ {r.name} — {t('shadows.no_data')}[/]")
            continue
        color = "green" if r.delta > 0 else (NEG if r.delta < 0 else "dim")
        lines.append(f"[bold {ACC}]▸ {r.name}[/]  [{color}]{cost.split(': ', 1)[1]}[/]")
    return "\n".join(lines)


def _money(value: Decimal | None, ccy: str) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.0f} {ccy}"
