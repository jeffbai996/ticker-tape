"""Thesis-breakers screen — watcher verdict table + review candidates.

Pure formatter: snapshot dict (thesis_data.load_snapshot) → Rich markup.
The external watcher owns the discipline; this screen is read-only glass
over its recorded state. Distinct from screens/thesis.py (the holdings
bucket dashboard) — this one is the sell-discipline checklist.
"""

from datetime import datetime, timezone

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
    # House discipline (data.py's ZoneInfo/UTC usage, db.py's
    # datetime.now(timezone.utc)) does wall-clock math with explicit
    # tz-aware datetimes rather than two naive local-time calls that only
    # cancel out by coincidence of both using the same process tz.
    delta = datetime.now(timezone.utc) - datetime.fromtimestamp(epoch, tz=timezone.utc)
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{delta.total_seconds() / 60:.0f}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def _breaker_line(b: dict) -> list[str]:
    style = _VERDICT_STYLE.get(b["verdict"], "dim")
    flags = "".join(
        f" [dim]{f}[/]" for f, on in
        (("auto", b.get("auto")), ("swept", b.get("swept")),
         ("qtr", b.get("earnings"))) if on)
    out = [f"  [{style}]{b['verdict']:<18}[/] [bold]{b['id']}[/]"
           f" [dim]({b.get('category', '')}/{b.get('severity', '')})[/]{flags}"]
    if b["verdict"] == "FIRED" and b.get("reason"):
        out.append(f"      [{NEG}]{b['reason']}[/]")
    return out


def format_breakers(snap: dict) -> str:
    if not snap.get("available"):
        return f"[dim]{t('breakers.unavailable')}[/]"

    if "groups" not in snap:                      # raw load_snapshot dicts
        import thesis_data
        thesis_data.synthesize(snap)
    g, h = snap["groups"], snap["health"]

    lines: list[str] = []
    # health headline — the one-glance answer
    state_style = f"bold {NEG}" if h["state"] == "BREACHED" else "bold green"
    lines.append(
        f"  [{state_style}]{t('breakers.' + h['state'].lower())}[/]  "
        f"[dim]{h['fired']} {t('breakers.fired')} · "
        f"{h['clear']} {t('breakers.holding')} · "
        f"{h['awaiting']} {t('breakers.awaiting')}[/]")
    nxt = snap.get("next_catalyst")
    if nxt:
        lines.append(f"  [dim]{t('breakers.next_catalyst')}:[/] "
                     f"[{ACC}]{nxt['date']}[/] [dim]{nxt['what']}[/]")
    if snap.get("last_run"):
        lines.append(f"  [dim]{t('breakers.lastrun')}: {_age(snap['last_run'])}[/]")
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

    for key, label, style in (
            ("fired", "breakers.sec_fired", f"bold {NEG}"),
            ("clear", "breakers.sec_holding", "bold green"),
            ("awaiting", "breakers.sec_awaiting", "bold"),
            ("no_data", "breakers.sec_warming", "dim")):
        group = g.get(key) or []
        if not group:
            continue
        lines.append(f"[{style}]■ {t(label)} ({len(group)})[/]")
        for b in group:
            lines.extend(_breaker_line(b))
            if key == "awaiting":
                # manual breakers fire on judgment — show the exact command
                lines.append(f"      [dim]watcher.py manual {b['id']} "
                             f"--fired/--clear --note \"…\"[/]")
        lines.append("")

    cats = snap.get("catalysts") or []
    if cats:
        lines.append(f"[bold]■ {t('breakers.calendar')}[/]")
        nxt_date = (nxt or {}).get("date")
        for c in cats:
            style = ACC if c["date"] == nxt_date else "dim"
            lines.append(f"  [{style}]{c['date']}[/]  [dim]{c['what']}[/]")
    return "\n".join(lines).rstrip()
