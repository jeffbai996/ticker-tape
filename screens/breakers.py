"""Thesis-breakers screen — watcher verdict table + review candidates.

Pure formatter: snapshot dict (thesis_data.load_snapshot) → Rich markup.
The external watcher owns the discipline; this screen is read-only glass
over its recorded state. Distinct from screens/thesis.py (the holdings
bucket dashboard) — this one is the sell-discipline checklist.
"""

from datetime import datetime

from formatters import NEG, ACC
from i18n import t


_VERDICT_STYLE = {
    "FIRED": f"bold {NEG}",
    "CLEAR": "green",
    "INSUFFICIENT_DATA": "dim",
}


def _age(epoch: float | None) -> str:
    if not epoch:
        return ""
    delta = datetime.now() - datetime.fromtimestamp(epoch)
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{delta.total_seconds() / 60:.0f}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def format_breakers(snap: dict) -> str:
    if not snap.get("available"):
        return f"[dim]{t('breakers.unavailable')}[/]"

    lines: list[str] = []
    if snap.get("last_run"):
        lines.append(f"[dim]{t('breakers.lastrun')}: {_age(snap['last_run'])}[/]")
        lines.append("")

    # candidates first — they're the actionable part
    cands = snap.get("candidates") or []
    if cands:
        lines.append(f"[bold {NEG}]■ {t('breakers.candidates')} ({len(cands)})[/]")
        for c in cands:
            lines.append(f"  [bold]#{c['id']}[/] {c['breaker_id']} — {c['summary']}")
            lines.append(f"    [dim]{c['url']}[/]")
            lines.append(
                f"    [dim]watcher.py candidates confirm {c['id']}  ·  "
                f"candidates dismiss {c['id']}[/]")
        lines.append("")

    for b in snap.get("breakers", []):
        style = _VERDICT_STYLE.get(b["verdict"], "dim")
        flags = "".join(
            f" [dim]{f}[/]" for f, on in
            (("auto", b.get("auto")), ("swept", b.get("swept")),
             ("qtr", b.get("earnings"))) if on)
        lines.append(f"  [{style}]{b['verdict']:<18}[/] [bold]{b['id']}[/]"
                     f" [dim]({b.get('category', '')}/{b.get('severity', '')})[/]{flags}")
        if b["verdict"] == "FIRED" and b.get("reason"):
            lines.append(f"      [{NEG}]{b['reason']}[/]")

    fired = sum(1 for b in snap.get("breakers", []) if b["verdict"] == "FIRED")
    lines.append("")
    color = NEG if fired else ACC
    lines.append(f"  [{color}]{fired} {t('breakers.fired')}[/] "
                 f"[dim]/ {len(snap.get('breakers', []))} {t('breakers.total')}[/]")
    return "\n".join(lines)
