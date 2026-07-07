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
