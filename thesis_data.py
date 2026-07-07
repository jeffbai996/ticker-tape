"""Bridge to an external breaker watcher — read-only verdict snapshots.

ticker-tape renders what the watcher recorded; it never evaluates breakers
itself (the watcher owns the discipline, this terminal owns the glass). The
watcher is a separate project holding its own config and SQLite state; its
location comes from config.THESIS_WATCHER_DIR. Everything degrades honestly:
missing dir/db/registry → an "unavailable" snapshot, never a fabricated one.
"""

from __future__ import annotations

import logging
import os
import sqlite3

log = logging.getLogger(__name__)


def _empty() -> dict:
    return {"available": False, "breakers": [], "candidates": [],
            "last_run": None}


def load_snapshot(base_dir: str) -> dict:
    """Join the watcher's CURRENT registry onto its recorded state.

    A state row for a renamed/removed breaker is history, not coverage, so
    the registry drives the list; a breaker with no state yet shows
    "(never evaluated)" instead of silently missing.
    """
    base = os.path.expanduser(base_dir)
    db_path = os.path.join(base, "data", "watcher.db")
    reg_path = os.path.join(base, "breakers.yaml")
    if not (os.path.exists(db_path) and os.path.exists(reg_path)):
        return _empty()

    try:
        import yaml
        with open(reg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, Exception) as e:
        log.warning("thesis registry unreadable: %s", e)
        return _empty()

    desc: dict[str, dict] = {}
    for b in raw.get("breakers", []):
        spec = b.get("signal_spec") or {}
        desc[str(b.get("id"))] = {
            "description": str(b.get("human_description", "")).strip(),
            "category": str(b.get("category", "")),
            "severity": str(b.get("severity", "")),
            "auto": str(spec.get("type")) != "manual",
            "swept": bool(b.get("sweep")),
        }

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT breaker_id, verdict, reason, updated_at "
            "FROM breaker_state").fetchall()
        cands = con.execute(
            "SELECT id, breaker_id, summary, evidence, url, created_at "
            "FROM candidate WHERE status='new' ORDER BY id").fetchall()
        con.close()
    except sqlite3.Error as e:
        log.warning("thesis state unreadable: %s", e)
        return _empty()

    out = _empty()
    out["available"] = True
    state_by_id = {r[0]: r for r in rows}
    for bid, meta in desc.items():
        row = state_by_id.get(bid)
        out["breakers"].append({
            "id": bid,
            "verdict": row[1] if row else "(never evaluated)",
            "reason": row[2] if row else "",
            "updated_at": row[3] if row else None,
            **meta})
        if row and row[3]:
            out["last_run"] = max(out["last_run"] or 0, row[3])
    out["candidates"] = [
        {"id": c[0], "breaker_id": c[1], "summary": c[2], "evidence": c[3],
         "url": c[4], "created_at": c[5]} for c in cands]
    return out
