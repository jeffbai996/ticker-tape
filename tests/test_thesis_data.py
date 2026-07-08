"""Tests for the breaker-watcher bridge — reads a WATCHER-OWNED state db.

ticker-tape never evaluates breakers itself; it renders the verdicts the
external watcher recorded. Missing watcher dir / db / registry all degrade
to an honest empty snapshot — never a fabricated table.
"""
import sqlite3
from pathlib import Path

import thesis_data


def _seed(tmp_path: Path) -> Path:
    base = tmp_path / "watcher"
    (base / "data").mkdir(parents=True)
    con = sqlite3.connect(base / "data" / "watcher.db")
    con.executescript("""
      CREATE TABLE breaker_state (breaker_id TEXT PRIMARY KEY, verdict TEXT,
        evidence TEXT DEFAULT '{}', reason TEXT DEFAULT '', updated_at REAL,
        alerted_at REAL);
      CREATE TABLE candidate (id INTEGER PRIMARY KEY, breaker_id TEXT,
        summary TEXT, evidence TEXT, url TEXT, status TEXT, created_at REAL);
    """)
    con.execute("INSERT INTO breaker_state VALUES ('rates_high','CLEAR','{}','',1751900000,NULL)")
    con.execute("INSERT INTO breaker_state VALUES ('old_renamed','FIRED','{}','stale',1751000000,NULL)")
    con.execute("INSERT INTO candidate VALUES (1,'rates_high','rates spiked','q','https://x','new',1751900000)")
    con.commit(); con.close()
    (base / "breakers.yaml").write_text(
        "breakers:\n"
        "  - id: rates_high\n    category: macro\n    human_description: rates too high\n"
        "    signal_spec: {type: sustained_level, threshold: 3}\n"
        "  - id: never_run\n    category: capex\n    human_description: capex cut\n"
        "    signal_spec: {type: manual}\n")
    return base


def test_snapshot_joins_registry_and_state(tmp_path):
    snap = thesis_data.load_snapshot(str(_seed(tmp_path)))
    assert snap["available"] is True
    ids = {b["id"]: b for b in snap["breakers"]}
    assert ids["rates_high"]["verdict"] == "CLEAR"
    assert ids["never_run"]["verdict"] == "(never evaluated)"
    assert "old_renamed" not in ids          # stale state row dropped
    assert snap["candidates"][0]["summary"] == "rates spiked"


def test_missing_watcher_dir_is_honest_empty(tmp_path):
    snap = thesis_data.load_snapshot(str(tmp_path / "nope"))
    assert snap["available"] is False and snap["breakers"] == []


# ── synthesis: groups / health / catalysts ─────────────────────────────
# The raw snapshot answers "what does each detector return"; the surfaces
# want "is the thesis intact and what do I still owe it". synthesize() is
# pure + idempotent: grouping by verdict AND detector type (a manual
# breaker with no input is actionable "awaiting"; an auto one is merely
# "warming up"), a one-line health verdict, and the next catalyst date.

def _bk(bid, verdict, auto=True, severity="trim"):
    return {"id": bid, "verdict": verdict, "auto": auto,
            "severity": severity, "reason": "", "category": "x"}


def test_synthesis_groups_fired_clear_awaiting_nodata():
    snap = {"available": True, "breakers": [
        _bk("a", "FIRED"), _bk("b", "CLEAR"),
        _bk("c", "INSUFFICIENT_DATA", auto=False),   # manual → awaiting
        _bk("d", "INSUFFICIENT_DATA", auto=True),    # auto → warming up
        _bk("e", "(never evaluated)", auto=True),
    ], "candidates": [], "last_run": None}
    thesis_data.synthesize(snap)
    g = snap["groups"]
    assert [b["id"] for b in g["fired"]] == ["a"]
    assert [b["id"] for b in g["clear"]] == ["b"]
    assert [b["id"] for b in g["awaiting"]] == ["c"]
    assert {b["id"] for b in g["no_data"]} == {"d", "e"}


def test_health_intact_when_none_fired():
    snap = {"available": True, "breakers": [
        _bk("b", "CLEAR"), _bk("c", "INSUFFICIENT_DATA", auto=False)],
        "candidates": [], "last_run": None}
    thesis_data.synthesize(snap)
    assert snap["health"]["state"] == "INTACT"
    # counts surfaced for the formatter's headline (i18n lives there)
    assert (snap["health"]["fired"], snap["health"]["clear"],
            snap["health"]["awaiting"]) == (0, 1, 1)


def test_health_breached_when_fired():
    snap = {"available": True, "breakers": [_bk("a", "FIRED")],
            "candidates": [], "last_run": None}
    thesis_data.synthesize(snap)
    assert snap["health"]["state"] == "BREACHED"


def test_synthesize_is_idempotent():
    snap = {"available": True, "breakers": [_bk("b", "CLEAR")],
            "candidates": [], "last_run": None}
    thesis_data.synthesize(snap)
    first = snap["health"]
    thesis_data.synthesize(snap)
    assert snap["health"] == first


def test_catalysts_loaded_from_watcher_yaml(tmp_path):
    base = _seed(tmp_path)
    (base / "catalysts.yaml").write_text(
        "catalysts:\n"
        "  - date: '2099-01-15'\n    what: MEGACORP earnings\n"
        "  - date: '2099-03-01'\n    what: CHIPCO earnings\n")
    snap = thesis_data.load_snapshot(str(base))
    assert [c["date"] for c in snap["catalysts"]] == ["2099-01-15", "2099-03-01"]
    assert snap["next_catalyst"]["date"] == "2099-01-15"   # first future date


def test_past_catalysts_not_next(tmp_path):
    base = _seed(tmp_path)
    (base / "catalysts.yaml").write_text(
        "catalysts:\n"
        "  - date: '2001-01-01'\n    what: long gone\n"
        "  - date: '2099-03-01'\n    what: CHIPCO earnings\n")
    snap = thesis_data.load_snapshot(str(base))
    assert snap["next_catalyst"]["date"] == "2099-03-01"


def test_missing_catalysts_file_is_empty_list(tmp_path):
    snap = thesis_data.load_snapshot(str(_seed(tmp_path)))
    assert snap["catalysts"] == [] and snap["next_catalyst"] is None
