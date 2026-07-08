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
from datetime import date

log = logging.getLogger(__name__)


def _empty() -> dict:
    return {"available": False, "breakers": [], "candidates": [],
            "last_run": None, "catalysts": [], "next_catalyst": None,
            "rotation": {"current": None, "history": [],
                         "needs_review": False}}


def synthesize(snap: dict) -> dict:
    """Add the human layer on top of raw verdicts — pure and idempotent.

    The raw snapshot answers "what does each detector return"; the pane
    wants "is the thesis intact and what do I still owe it". Groups by
    what matters: a manual breaker with no input is actionable
    ("awaiting" a human observation), while an auto breaker without
    enough data points is merely warming up.
    """
    fired, clear, awaiting, no_data = [], [], [], []
    for b in snap.get("breakers", []):
        v = b.get("verdict")
        if v == "FIRED":
            fired.append(b)
        elif v == "CLEAR":
            clear.append(b)
        elif v == "INSUFFICIENT_DATA" and not b.get("auto", True):
            awaiting.append(b)
        else:
            no_data.append(b)
    snap["groups"] = {"fired": fired, "clear": clear,
                      "awaiting": awaiting, "no_data": no_data}
    snap["health"] = {
        "state": "BREACHED" if fired else "INTACT",
        "fired": len(fired), "clear": len(clear),
        "awaiting": len(awaiting)}
    # a FIRED reunderwrite breaker means the rotation date itself is suspect
    rot = snap.setdefault("rotation", {"current": None, "history": []})
    rot["needs_review"] = any(
        b.get("severity") == "reunderwrite" for b in fired)
    return snap


def _load_catalysts(base: str) -> list[dict]:
    """Read the watcher-owned catalyst calendar (catalysts.yaml).

    Calendar content is thesis material, so it lives with the watcher —
    never in this (public) repo's source. Missing file → empty list.
    """
    path = os.path.join(base, "catalysts.yaml")
    if not os.path.exists(path):
        return []
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return [{"date": str(c.get("date", "")), "what": str(c.get("what", ""))}
                for c in raw.get("catalysts", []) if c.get("date")]
    except (OSError, Exception) as e:
        log.warning("catalysts unreadable: %s", e)
        return []


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
            "earnings": bool(b.get("earnings")),
        }

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT breaker_id, verdict, reason, updated_at "
            "FROM breaker_state").fetchall()
        cands = con.execute(
            "SELECT id, breaker_id, summary, evidence, url, created_at "
            "FROM candidate WHERE status='new' ORDER BY id").fetchall()
        try:
            rot_rows = con.execute(
                "SELECT estimate, note, breaker_id, set_at "
                "FROM rotation_estimate ORDER BY id DESC LIMIT 10").fetchall()
        except sqlite3.Error:
            rot_rows = []   # pre-rotation watcher db
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
    out["rotation"] = {
        "current": ({"estimate": rot_rows[0][0], "note": rot_rows[0][1],
                     "breaker_id": rot_rows[0][2], "set_at": rot_rows[0][3]}
                    if rot_rows else None),
        "history": [{"estimate": r[0], "note": r[1], "breaker_id": r[2],
                     "set_at": r[3]} for r in rot_rows],
        "needs_review": False}
    out["catalysts"] = _load_catalysts(base)
    today = date.today().isoformat()
    out["next_catalyst"] = next(
        (c for c in out["catalysts"] if c["date"] >= today), None)
    return synthesize(out)
