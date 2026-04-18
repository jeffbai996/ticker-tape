# v2.6 `analyze` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an on-demand deep-dive command `analyze <target>` that streams a doc-grade memo via Gemini Pro and archives it to `data/analyses/{slug}/{YYYY-MM-DD-HHMM}.md`.

**Architecture:** New module `analyze.py` + new module `archive.py` (separated for clarity: analyze does classification + AI orchestration, archive does pure disk I/O over `data/analyses/`). Command wired into `app.py` via `elif action == "analyze":` and a new entry in `widgets/command_bar.py` alias table. Uses `chat_module._BACKENDS["pro"]` direct-access pattern — same pattern as `_show_briefing_ai`. No edits to `chat.py` internals, no edits to existing tests, no edits to existing data directories.

**Tech Stack:** Python 3.11+, pytest, Textual 8.1.1 (RichLog rendering), Rich markup. Gemini 3.1 Pro via `chat_module._BACKENDS["pro"]`. YAML front-matter via `PyYAML` (already a transitive dep — verify in step 0).

**Spec:** `docs/superpowers/specs/2026-04-18-analyze-command-design.md`

---

## File Structure

**Create:**
- `analyze.py` — classify_target, build_context, build_system_prompt, run_analysis. AI orchestration.
- `archive.py` — pure I/O: write_memo, load_prior, rebuild_index, slug_for_target. No network, no AI.
- `tests/test_archive.py` — archive module unit tests (roundtrip, ordering, empty, malformed, index rebuild).
- `tests/test_analyze.py` — analyze module unit tests (classification, routing, happy path, failure). Uses fake backend via monkeypatch.
- `data/analyses/.gitkeep` — make the archive directory exist in the repo (empty, or comment file).

**Modify:**
- `widgets/command_bar.py` — add `"analyze": "analyze"` to aliases (one line, in `parse_command` alias dict near line 182).
- `app.py` — add `elif action == "analyze":` branch routing to `self._run_analyze(args)`. Add `_run_analyze` and `_analyze_worker` thread methods. Import `analyze` module at top.
- `README.md` — add section documenting `analyze` command.

**Do NOT touch:**
- `chat.py` (only read `_BACKENDS["pro"]` and `MODELS["pro"]` from it)
- Existing tests under `tests/` (they must stay green unchanged)
- Existing `data/` subdirectories other than creating new `data/analyses/`
- `TICKER_TOOLS` definition in `chat.py`
- i18n keys (not adding any for v2.6; user-facing strings are hardcoded English for now)

---

## Branch Setup

- [ ] **Step 0a: Create feature branch off main**

Run:
```bash
git checkout main
git pull origin main
git checkout -b feature/analyze-command
git log --oneline -2
```
Expected: current HEAD shows `c6b488d docs: spec for v2.6 analyze command` and `2a3da3e post-v2.5.5 polish...`. New branch `feature/analyze-command` created.

- [ ] **Step 0b: Verify PyYAML is available**

Run:
```bash
source venv/bin/activate
python -c "import yaml; print(yaml.__version__)"
```

If it prints a version: PyYAML already available (transitive dep via Textual or similar). Skip to Step 0c.

If ModuleNotFoundError: add PyYAML:
```bash
pip install pyyaml
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore: add PyYAML for analyze front-matter parsing"
```

- [ ] **Step 0c: Verify 604 existing tests pass on branch**

Run:
```bash
pytest -q 2>&1 | tail -5
```
Expected: something like `604 passed in X.XXs`. If any fail, STOP and investigate before proceeding.

---

## Task 1: Archive module — slug_for_target

**Files:**
- Create: `archive.py`
- Test: `tests/test_archive.py`

**Purpose:** Convert a raw user target into a filesystem-safe slug. `AVGO` → `AVGO`. `rotation` → `rotation`. `"why is XLU up"` → `_freeform/a1b2c3d4` (hash-prefixed under `_freeform/`).

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_archive.py`:
```python
"""Tests for archive.py — pure disk I/O over data/analyses/."""

import archive


class TestSlugForTarget:
    def test_symbol_preserved_uppercase(self):
        assert archive.slug_for_target("AVGO", "symbol") == "AVGO"

    def test_symbol_uppercased(self):
        assert archive.slug_for_target("avgo", "symbol") == "AVGO"

    def test_thesis_lowercased(self):
        assert archive.slug_for_target("Rotation", "thesis") == "rotation"

    def test_freeform_hashed_under_subdir(self):
        slug = archive.slug_for_target("why is XLU up", "freeform")
        assert slug.startswith("_freeform/")
        # Hash must be stable for the same input
        assert slug == archive.slug_for_target("why is XLU up", "freeform")

    def test_freeform_different_inputs_different_hashes(self):
        a = archive.slug_for_target("question one", "freeform")
        b = archive.slug_for_target("question two", "freeform")
        assert a != b

    def test_freeform_hash_is_8_hex_chars(self):
        slug = archive.slug_for_target("anything", "freeform")
        hash_part = slug.split("/")[-1]
        assert len(hash_part) == 8
        assert all(c in "0123456789abcdef" for c in hash_part)
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_archive.py -v`
Expected: ImportError or ModuleNotFoundError for `archive`.

- [ ] **Step 1.3: Create archive.py with slug_for_target**

Create `archive.py`:
```python
"""Archive module — pure disk I/O for data/analyses/.

Writes memos, loads prior memos, rebuilds the index. No network calls,
no AI. All functions are pure given the filesystem state.
"""

import hashlib
import os


ARCHIVE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "analyses"
)


def slug_for_target(target: str, kind: str) -> str:
    """Convert a target + kind into a filesystem-safe slug.

    symbol → uppercase ticker (AVGO)
    thesis → lowercase name (rotation)
    freeform → _freeform/<8-char sha256 hex prefix>
    """
    if kind == "symbol":
        return target.upper()
    if kind == "thesis":
        return target.lower()
    # freeform
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:8]
    return f"_freeform/{digest}"
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/test_archive.py -v`
Expected: 6 passed.

- [ ] **Step 1.5: Commit**

```bash
git add archive.py tests/test_archive.py
git commit -m "feat(analyze): archive slug_for_target"
```

---

## Task 2: Archive module — write_memo roundtrip

**Files:**
- Modify: `archive.py`
- Test: `tests/test_archive.py`

**Purpose:** Write a memo to `data/analyses/{slug}/{timestamp}.md` with YAML front-matter. Read it back. Handle missing directory creation.

- [ ] **Step 2.1: Write the failing test**

Append to `tests/test_archive.py`:
```python
import datetime
import archive as archive_mod


class TestWriteAndReadMemo:
    def test_write_memo_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        path = archive_mod.write_memo(
            slug="AVGO",
            front_matter={
                "target": "AVGO",
                "kind": "symbol",
                "angle": "general",
                "date": "2026-04-18T14:23:00-04:00",
                "model": "gemini-3.1-pro",
                "prior_memos": [],
                "tools_used": [],
                "conviction": {"level": "high", "key_claim": "test claim"},
                "trigger_type": "manual",
            },
            body="# AVGO — 2026-04-18\n\n## Context\nTest memo body.\n",
        )
        assert path.endswith(".md")
        assert "AVGO" in path
        assert (tmp_path / "AVGO").exists()

    def test_write_memo_content_roundtrips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        fm = {
            "target": "MU",
            "kind": "symbol",
            "angle": "earnings",
            "date": "2026-04-18T14:23:00-04:00",
            "model": "gemini-3.1-pro",
            "prior_memos": [],
            "tools_used": ["get_quote"],
            "conviction": {"level": "medium", "key_claim": "HBM ramp continues"},
            "trigger_type": "manual",
        }
        body = "# MU — 2026-04-18\n\n## Context\nBody content here.\n"
        path = archive_mod.write_memo("MU", fm, body)
        text = open(path).read()
        assert "target: MU" in text
        assert "angle: earnings" in text
        assert "HBM ramp continues" in text
        assert "Body content here." in text

    def test_write_memo_filename_timestamp_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        fm = {
            "target": "AVGO", "kind": "symbol", "angle": "general",
            "date": "2026-04-18T14:23:00-04:00", "model": "x",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "low", "key_claim": "x"},
            "trigger_type": "manual",
        }
        path = archive_mod.write_memo("AVGO", fm, "body")
        filename = os.path.basename(path)
        # Expect YYYY-MM-DD-HHMM.md
        assert filename == "2026-04-18-1423.md"

    def test_write_memo_creates_freeform_nested_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        fm = {
            "target": "why is XLU up", "kind": "freeform", "angle": "general",
            "date": "2026-04-18T14:23:00-04:00", "model": "x",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "low", "key_claim": "x"},
            "trigger_type": "manual",
        }
        path = archive_mod.write_memo("_freeform/abcd1234", fm, "body")
        assert "_freeform" in path
        assert os.path.exists(path)
```

- [ ] **Step 2.2: Run to verify it fails**

Run: `pytest tests/test_archive.py::TestWriteAndReadMemo -v`
Expected: AttributeError — `write_memo` not defined.

- [ ] **Step 2.3: Implement write_memo**

Add to `archive.py`:
```python
import yaml


def _timestamp_from_iso(iso: str) -> str:
    """Convert ISO-8601 date to YYYY-MM-DD-HHMM for filename."""
    # Parse "2026-04-18T14:23:00-04:00" → "2026-04-18-1423"
    date_part, time_part = iso.split("T")
    hhmm = time_part.split(":")[0] + time_part.split(":")[1]
    return f"{date_part}-{hhmm}"


def write_memo(slug: str, front_matter: dict, body: str) -> str:
    """Write a memo to disk under ARCHIVE_ROOT/{slug}/.

    Filename derived from front_matter['date'] as YYYY-MM-DD-HHMM.md.
    Creates directory if missing.
    Returns the absolute path written.
    """
    dir_path = os.path.join(ARCHIVE_ROOT, slug)
    os.makedirs(dir_path, exist_ok=True)
    filename = _timestamp_from_iso(front_matter["date"]) + ".md"
    path = os.path.join(dir_path, filename)

    fm_yaml = yaml.safe_dump(front_matter, sort_keys=False,
                             default_flow_style=None, allow_unicode=True)
    content = f"---\n{fm_yaml}---\n\n{body}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
```

- [ ] **Step 2.4: Run to verify it passes**

Run: `pytest tests/test_archive.py -v`
Expected: 10 passed (6 from Task 1 + 4 new).

- [ ] **Step 2.5: Commit**

```bash
git add archive.py tests/test_archive.py
git commit -m "feat(analyze): archive write_memo with YAML front-matter"
```

---

## Task 3: Archive module — load_prior

**Files:**
- Modify: `archive.py`
- Test: `tests/test_archive.py`

**Purpose:** Return list of prior memos for a slug, newest-first, with parsed front-matter. Skip memos with malformed front-matter (with a stderr warning, but don't raise).

- [ ] **Step 3.1: Write the failing test**

Append to `tests/test_archive.py`:
```python
class TestLoadPrior:
    def test_load_prior_empty_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        assert archive_mod.load_prior("AVGO") == []

    def test_load_prior_ordered_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        # Write 3 memos with different dates
        base_fm = {
            "target": "AVGO", "kind": "symbol", "angle": "general",
            "model": "x", "prior_memos": [], "tools_used": [],
            "conviction": {"level": "low", "key_claim": "x"},
            "trigger_type": "manual",
        }
        for d in ["2026-03-12T09:15:00-04:00",
                  "2026-04-18T14:23:00-04:00",
                  "2026-04-01T10:00:00-04:00"]:
            fm = dict(base_fm, date=d)
            archive_mod.write_memo("AVGO", fm, f"body for {d}")
        priors = archive_mod.load_prior("AVGO")
        assert len(priors) == 3
        dates = [p["front_matter"]["date"] for p in priors]
        assert dates == sorted(dates, reverse=True)

    def test_load_prior_skips_malformed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        os.makedirs(str(tmp_path / "AVGO"))
        # Write one valid and one malformed
        fm = {
            "target": "AVGO", "kind": "symbol", "angle": "general",
            "date": "2026-04-18T14:23:00-04:00", "model": "x",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "low", "key_claim": "x"},
            "trigger_type": "manual",
        }
        archive_mod.write_memo("AVGO", fm, "body")
        # Malformed: no front-matter at all
        with open(str(tmp_path / "AVGO" / "bad.md"), "w") as f:
            f.write("no front matter here\n")
        priors = archive_mod.load_prior("AVGO")
        assert len(priors) == 1
        assert priors[0]["front_matter"]["target"] == "AVGO"

    def test_load_prior_returns_body_and_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        fm = {
            "target": "MU", "kind": "symbol", "angle": "general",
            "date": "2026-04-18T14:23:00-04:00", "model": "x",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "low", "key_claim": "x"},
            "trigger_type": "manual",
        }
        archive_mod.write_memo("MU", fm, "unique body content")
        priors = archive_mod.load_prior("MU")
        assert len(priors) == 1
        assert "unique body content" in priors[0]["body"]
        assert priors[0]["path"].endswith(".md")
```

- [ ] **Step 3.2: Run to verify it fails**

Run: `pytest tests/test_archive.py::TestLoadPrior -v`
Expected: AttributeError — `load_prior` not defined.

- [ ] **Step 3.3: Implement load_prior**

Add to `archive.py`:
```python
import sys


def _parse_memo_file(path: str) -> dict | None:
    """Read a memo, split front-matter from body. Return None if malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not text.startswith("---\n"):
            return None
        _, fm_text, body = text.split("---\n", 2)
        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict) or "date" not in fm:
            return None
        return {"path": path, "front_matter": fm, "body": body.lstrip("\n")}
    except Exception as e:
        print(f"[archive] skipping malformed memo {path}: {e}", file=sys.stderr)
        return None


def load_prior(slug: str) -> list[dict]:
    """Load all prior memos for a slug, newest-first.

    Returns list of {"path": str, "front_matter": dict, "body": str}.
    Malformed memos are skipped with a stderr warning.
    """
    dir_path = os.path.join(ARCHIVE_ROOT, slug)
    if not os.path.isdir(dir_path):
        return []
    memos = []
    for name in os.listdir(dir_path):
        if not name.endswith(".md"):
            continue
        parsed = _parse_memo_file(os.path.join(dir_path, name))
        if parsed is not None:
            memos.append(parsed)
    # Sort newest-first by front_matter.date
    memos.sort(key=lambda m: m["front_matter"]["date"], reverse=True)
    return memos
```

- [ ] **Step 3.4: Run to verify it passes**

Run: `pytest tests/test_archive.py -v`
Expected: 14 passed.

- [ ] **Step 3.5: Commit**

```bash
git add archive.py tests/test_archive.py
git commit -m "feat(analyze): archive load_prior newest-first, skip malformed"
```

---

## Task 4: Archive module — rebuild_index

**Files:**
- Modify: `archive.py`
- Test: `tests/test_archive.py`

**Purpose:** Walk every slug directory under ARCHIVE_ROOT, produce `_index.json` summarizing all memos.

- [ ] **Step 4.1: Write the failing test**

Append to `tests/test_archive.py`:
```python
import json


class TestRebuildIndex:
    def test_rebuild_index_empty_archive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        archive_mod.rebuild_index()
        index_path = tmp_path / "_index.json"
        assert index_path.exists()
        assert json.loads(index_path.read_text()) == {}

    def test_rebuild_index_includes_all_slugs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        base_fm = {
            "kind": "symbol", "angle": "general", "model": "x",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "high", "key_claim": "claim"},
            "trigger_type": "manual",
        }
        archive_mod.write_memo(
            "AVGO",
            dict(base_fm, target="AVGO", date="2026-04-18T14:23:00-04:00"),
            "body")
        archive_mod.write_memo(
            "MU",
            dict(base_fm, target="MU", date="2026-04-18T15:00:00-04:00"),
            "body")
        archive_mod.rebuild_index()
        index = json.loads((tmp_path / "_index.json").read_text())
        assert "AVGO" in index
        assert "MU" in index
        assert index["AVGO"][0]["conviction"]["key_claim"] == "claim"
        assert index["AVGO"][0]["path"].endswith(".md")
```

- [ ] **Step 4.2: Run to verify it fails**

Run: `pytest tests/test_archive.py::TestRebuildIndex -v`
Expected: AttributeError — `rebuild_index` not defined.

- [ ] **Step 4.3: Implement rebuild_index**

Add to `archive.py`:
```python
import json


def rebuild_index() -> str:
    """Walk ARCHIVE_ROOT, produce _index.json. Returns path to index file."""
    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    index: dict[str, list[dict]] = {}
    for slug_name in sorted(os.listdir(ARCHIVE_ROOT)):
        slug_dir = os.path.join(ARCHIVE_ROOT, slug_name)
        if not os.path.isdir(slug_dir) or slug_name.startswith("."):
            continue
        # Handle nested _freeform/<hash>/ layout
        if slug_name == "_freeform":
            for sub in sorted(os.listdir(slug_dir)):
                sub_path = os.path.join(slug_dir, sub)
                if os.path.isdir(sub_path):
                    key = f"_freeform/{sub}"
                    index[key] = _index_entries_for_slug(key)
            continue
        index[slug_name] = _index_entries_for_slug(slug_name)
    path = os.path.join(ARCHIVE_ROOT, "_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return path


def _index_entries_for_slug(slug: str) -> list[dict]:
    """Return lightweight index entries (no full body) for a slug."""
    entries = []
    for memo in load_prior(slug):
        fm = memo["front_matter"]
        entries.append({
            "date": fm["date"],
            "path": os.path.relpath(memo["path"], ARCHIVE_ROOT),
            "target": fm.get("target", ""),
            "kind": fm.get("kind", ""),
            "conviction": fm.get("conviction", {}),
            "summary": memo["body"].split("\n\n")[0][:200],
        })
    return entries
```

- [ ] **Step 4.4: Run to verify it passes**

Run: `pytest tests/test_archive.py -v`
Expected: 16 passed.

- [ ] **Step 4.5: Commit**

```bash
git add archive.py tests/test_archive.py
git commit -m "feat(analyze): archive rebuild_index walks slugs incl. _freeform"
```

---

## Task 5: Analyze module — classify_target

**Files:**
- Create: `analyze.py`
- Test: `tests/test_analyze.py`

**Purpose:** Given a raw target string, classify as symbol/thesis/freeform by checking against watchlist + positions + `config.THESIS_BUCKETS`.

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_analyze.py`:
```python
"""Tests for analyze.py — target classification and AI orchestration."""

import pytest

import analyze


class TestClassifyTarget:
    def test_uppercase_ticker_in_watchlist_is_symbol(self, monkeypatch):
        monkeypatch.setattr(analyze, "_watchlist_symbols",
                            lambda: ["AVGO", "MU", "NVDA"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])
        kind, normalized = analyze.classify_target("AVGO")
        assert kind == "symbol"
        assert normalized == "AVGO"

    def test_lowercase_ticker_normalized_to_upper(self, monkeypatch):
        monkeypatch.setattr(analyze, "_watchlist_symbols",
                            lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])
        kind, normalized = analyze.classify_target("avgo")
        assert kind == "symbol"
        assert normalized == "AVGO"

    def test_thesis_key_is_thesis(self, monkeypatch):
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: [])
        monkeypatch.setattr(analyze, "_thesis_keys",
                            lambda: ["rotation", "silicon"])
        kind, normalized = analyze.classify_target("rotation")
        assert kind == "thesis"
        assert normalized == "rotation"

    def test_thesis_key_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: [])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: ["rotation"])
        kind, normalized = analyze.classify_target("ROTATION")
        assert kind == "thesis"
        assert normalized == "rotation"

    def test_unknown_is_freeform(self, monkeypatch):
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: ["rotation"])
        kind, normalized = analyze.classify_target("why is XLU up")
        assert kind == "freeform"
        assert normalized == "why is XLU up"

    def test_unknown_ticker_is_still_symbol_if_looks_like_one(self, monkeypatch):
        """Any 1-5 letter uppercase token is treated as a symbol even if not
        in the watchlist — user may want to analyze a symbol they don't hold."""
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: [])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])
        kind, normalized = analyze.classify_target("TSLA")
        assert kind == "symbol"
        assert normalized == "TSLA"
```

- [ ] **Step 5.2: Run to verify it fails**

Run: `pytest tests/test_analyze.py -v`
Expected: ImportError on `analyze`.

- [ ] **Step 5.3: Implement classify_target**

Create `analyze.py`:
```python
"""Analyze module — on-demand deep-dive orchestration.

Reads user target, classifies as symbol/thesis/freeform, loads prior memos
from archive, builds context, calls Gemini Pro, streams memo to terminal,
writes result back to archive.
"""

import re


def _watchlist_symbols() -> list[str]:
    """Return current watchlist symbols. Isolated for easy monkeypatching."""
    import data
    return data.get_all_symbols()


def _thesis_keys() -> list[str]:
    """Return configured thesis bucket keys (lowercased)."""
    import config
    return [k.lower() for k in (config.THESIS_BUCKETS or {}).keys()]


_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")


def classify_target(target: str) -> tuple[str, str]:
    """Classify a raw target string.

    Returns (kind, normalized_target) where kind is "symbol", "thesis",
    or "freeform".

    Rules (in order):
    1. If target is in watchlist (case-insensitive) → symbol
    2. If target is in thesis keys (case-insensitive) → thesis
    3. If target is 1-5 letters with no spaces → symbol (even if unknown)
    4. Otherwise → freeform (normalized unchanged)
    """
    stripped = target.strip()
    upper = stripped.upper()
    lower = stripped.lower()

    if upper in _watchlist_symbols():
        return ("symbol", upper)
    if lower in _thesis_keys():
        return ("thesis", lower)
    if _TICKER_RE.match(stripped):
        return ("symbol", upper)
    return ("freeform", stripped)
```

- [ ] **Step 5.4: Run to verify it passes**

Run: `pytest tests/test_analyze.py -v`
Expected: 6 passed.

- [ ] **Step 5.5: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat(analyze): classify_target by watchlist/thesis/freeform"
```

---

## Task 6: Analyze module — build_system_prompt

**Files:**
- Modify: `analyze.py`
- Test: `tests/test_analyze.py`

**Purpose:** Produce the system prompt given kind, target, angle_hint, and prior memos. Prompt instructs Gemini Pro to produce the memo section structure from the spec.

- [ ] **Step 6.1: Write the failing test**

Append to `tests/test_analyze.py`:
```python
class TestBuildSystemPrompt:
    def test_symbol_prompt_names_target(self):
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", [])
        assert "AVGO" in prompt
        assert "analyst" in prompt.lower()

    def test_prompt_includes_section_headers(self):
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", [])
        for section in ["Context", "What Changed", "Current Read",
                        "Risks", "Suggested Actions", "Sources"]:
            assert section in prompt

    def test_prompt_mentions_grounding(self):
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", [])
        assert "google" in prompt.lower() or "grounding" in prompt.lower() \
               or "search" in prompt.lower()

    def test_prompt_first_memo_when_no_prior(self):
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", [])
        assert "first memo" in prompt.lower() or "no prior" in prompt.lower()

    def test_prompt_includes_prior_summaries_when_present(self):
        priors = [{
            "path": "/x/AVGO/2026-03-12-0915.md",
            "front_matter": {
                "date": "2026-03-12T09:15:00-04:00",
                "conviction": {"level": "high",
                               "key_claim": "custom silicon moat widening"},
            },
            "body": "# AVGO — 2026-03-12\n\n## Context\nOld context here.\n",
        }]
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", priors)
        assert "custom silicon moat widening" in prompt
        assert "2026-03-12" in prompt

    def test_angle_hint_passed_through(self):
        prompt = analyze.build_system_prompt(
            "symbol", "MU", "earnings", [])
        assert "earnings" in prompt.lower()

    def test_thesis_kind_prompt_different_from_symbol(self):
        sym_prompt = analyze.build_system_prompt(
            "symbol", "AVGO", "general", [])
        thesis_prompt = analyze.build_system_prompt(
            "thesis", "rotation", "general", [])
        assert sym_prompt != thesis_prompt
        assert "rotation" in thesis_prompt.lower()
```

- [ ] **Step 6.2: Run to verify it fails**

Run: `pytest tests/test_analyze.py::TestBuildSystemPrompt -v`
Expected: AttributeError — `build_system_prompt` not defined.

- [ ] **Step 6.3: Implement build_system_prompt**

Append to `analyze.py`:
```python
def build_system_prompt(kind: str, target: str, angle_hint: str,
                        prior_memos: list[dict]) -> str:
    """Build the system prompt for a deep-dive analysis call."""
    # Prior memos summary
    if prior_memos:
        prior_block = "Prior memos for this target (newest first):\n"
        for m in prior_memos[:5]:  # cap at 5 most recent
            fm = m["front_matter"]
            conv = fm.get("conviction", {})
            prior_block += (
                f"- {fm.get('date', '?')}: "
                f"conviction={conv.get('level', '?')} "
                f"claim=\"{conv.get('key_claim', '')}\"\n"
            )
    else:
        prior_block = "No prior memos for this target — this is the first memo."

    kind_guidance = {
        "symbol": f"Target is a single symbol: {target}. "
                  "Argue the thesis status for this specific position.",
        "thesis": f"Target is a thesis: {target}. "
                  "Argue the state of this thesis across the affected positions.",
        "freeform": f"Target is a freeform question: \"{target}\". "
                    "Answer directly with a doc-grade memo.",
    }[kind]

    angle_line = (f"Angle hint: {angle_hint}. Focus the memo accordingly."
                  if angle_hint and angle_hint != "general"
                  else "No specific angle — give a general deep-dive.")

    return f"""You are a senior equity analyst producing a doc-grade memo.

{kind_guidance}

{angle_line}

{prior_block}

Use your tools (quotes, technicals, thesis, positions, news) and Google search grounding to gather current data. Cite sources explicitly in the Sources section.

Produce the memo with EXACTLY these sections in this order:

## Context
Current price, position (if held), thesis bucket, since-last-memo one-line summary.

## What Changed Since Last Memo
If there is a prior memo: "Last time I argued X. Now I'm arguing Y because Z." Be explicit.
If no prior memo: state "First memo for this target."

## Current Read
Argue the current thesis status. Be direct and specific. No hedging.

## Risks / Disconfirming Evidence
Steelman the bear case. List concrete disconfirming evidence.

## Suggested Actions
Specific sizing/levels, or explicit "no action — hold." Include stop levels where relevant.

## Sources
Citations from grounding and tool calls.

Output ONLY the memo markdown starting with `# {target} — YYYY-MM-DD`. No preamble, no meta-commentary.
"""
```

- [ ] **Step 6.4: Run to verify it passes**

Run: `pytest tests/test_analyze.py -v`
Expected: 13 passed (6 classify + 7 prompt).

- [ ] **Step 6.5: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat(analyze): build_system_prompt for symbol/thesis/freeform"
```

---

## Task 7: Analyze module — build_front_matter

**Files:**
- Modify: `analyze.py`
- Test: `tests/test_analyze.py`

**Purpose:** Build the front-matter dict that will be written to disk. Takes kind/target/angle_hint/prior_memos/tools_used/conviction and produces the dict.

- [ ] **Step 7.1: Write the failing test**

Append to `tests/test_analyze.py`:
```python
class TestBuildFrontMatter:
    def test_basic_fields_populated(self):
        fm = analyze.build_front_matter(
            kind="symbol", target="AVGO", angle="general",
            prior_memos=[], tools_used=["get_quote"],
            conviction={"level": "high", "key_claim": "moat widening"},
            trigger_type="manual",
        )
        assert fm["target"] == "AVGO"
        assert fm["kind"] == "symbol"
        assert fm["angle"] == "general"
        assert fm["model"] == "gemini-3.1-pro-preview"
        assert fm["tools_used"] == ["get_quote"]
        assert fm["conviction"]["level"] == "high"
        assert fm["trigger_type"] == "manual"
        assert "date" in fm
        assert "T" in fm["date"]  # ISO-8601 format

    def test_prior_memos_as_relative_paths(self):
        priors = [{"path": "/abs/path/AVGO/2026-03-12-0915.md",
                   "front_matter": {}, "body": ""}]
        fm = analyze.build_front_matter(
            kind="symbol", target="AVGO", angle="general",
            prior_memos=priors, tools_used=[],
            conviction={"level": "low", "key_claim": "x"},
            trigger_type="manual",
        )
        assert fm["prior_memos"] == ["/abs/path/AVGO/2026-03-12-0915.md"]

    def test_default_trigger_type_is_manual(self):
        fm = analyze.build_front_matter(
            kind="symbol", target="AVGO", angle="general",
            prior_memos=[], tools_used=[],
            conviction={"level": "low", "key_claim": "x"},
        )
        assert fm["trigger_type"] == "manual"
```

- [ ] **Step 7.2: Run to verify it fails**

Run: `pytest tests/test_analyze.py::TestBuildFrontMatter -v`
Expected: AttributeError — `build_front_matter` not defined.

- [ ] **Step 7.3: Implement build_front_matter**

Append to `analyze.py`:
```python
import datetime


GEMINI_PRO_MODEL_ID = "gemini-3.1-pro-preview"


def build_front_matter(kind: str, target: str, angle: str,
                       prior_memos: list[dict], tools_used: list[str],
                       conviction: dict, trigger_type: str = "manual") -> dict:
    """Build the YAML front-matter dict for a new memo."""
    now = datetime.datetime.now().astimezone()
    iso = now.isoformat(timespec="seconds")
    return {
        "target": target,
        "kind": kind,
        "angle": angle,
        "date": iso,
        "model": GEMINI_PRO_MODEL_ID,
        "prior_memos": [m["path"] for m in prior_memos],
        "tools_used": tools_used,
        "conviction": conviction,
        "trigger_type": trigger_type,
    }
```

- [ ] **Step 7.4: Run to verify it passes**

Run: `pytest tests/test_analyze.py -v`
Expected: 16 passed.

- [ ] **Step 7.5: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat(analyze): build_front_matter with ISO date + model id"
```

---

## Task 8: Analyze module — extract_conviction

**Files:**
- Modify: `analyze.py`
- Test: `tests/test_analyze.py`

**Purpose:** After Gemini returns the memo body, extract a conviction dict `{level, key_claim}` from the "Current Read" section heuristically. Worst case: `{level: "unknown", key_claim: first_sentence_of_current_read}`.

- [ ] **Step 8.1: Write the failing test**

Append to `tests/test_analyze.py`:
```python
class TestExtractConviction:
    def test_explicit_high_conviction(self):
        body = """# AVGO — 2026-04-18

## Context
x

## Current Read
High conviction long. Custom silicon moat is widening materially.

## Risks
x"""
        conv = analyze.extract_conviction(body)
        assert conv["level"] == "high"
        assert "custom silicon moat" in conv["key_claim"].lower()

    def test_explicit_medium_conviction(self):
        body = """## Current Read
Medium conviction — setup is constructive but macro is mixed."""
        conv = analyze.extract_conviction(body)
        assert conv["level"] == "medium"

    def test_explicit_low_conviction(self):
        body = """## Current Read
Low conviction. Trimming into strength."""
        conv = analyze.extract_conviction(body)
        assert conv["level"] == "low"

    def test_unknown_when_no_keyword(self):
        body = """## Current Read
The company reported earnings yesterday."""
        conv = analyze.extract_conviction(body)
        assert conv["level"] == "unknown"
        assert "company reported earnings" in conv["key_claim"].lower()

    def test_missing_current_read_section(self):
        body = "# Something else entirely\n\n## Context\nNo current read here.\n"
        conv = analyze.extract_conviction(body)
        assert conv["level"] == "unknown"
        # key_claim may be empty string if section missing — that's fine
```

- [ ] **Step 8.2: Run to verify it fails**

Run: `pytest tests/test_analyze.py::TestExtractConviction -v`
Expected: AttributeError — `extract_conviction` not defined.

- [ ] **Step 8.3: Implement extract_conviction**

Append to `analyze.py`:
```python
def extract_conviction(body: str) -> dict:
    """Parse the Current Read section to extract conviction level + key_claim.

    Heuristic: find the Current Read section text, scan first 200 chars for
    "high/medium/low conviction" keyword, return first sentence as key_claim.
    If section missing or no keyword: level="unknown".
    """
    m = re.search(r"##\s+Current Read\s*\n(.*?)(?=\n##\s|\Z)", body,
                  re.DOTALL | re.IGNORECASE)
    section = m.group(1).strip() if m else ""
    lower = section.lower()[:300]
    if "high conviction" in lower:
        level = "high"
    elif "medium conviction" in lower or "moderate conviction" in lower:
        level = "medium"
    elif "low conviction" in lower:
        level = "low"
    else:
        level = "unknown"
    # First sentence as key_claim
    first = re.split(r"[.!?]\s", section, maxsplit=1)[0] if section else ""
    return {"level": level, "key_claim": first[:200].strip()}
```

- [ ] **Step 8.4: Run to verify it passes**

Run: `pytest tests/test_analyze.py -v`
Expected: 21 passed.

- [ ] **Step 8.5: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat(analyze): extract_conviction from Current Read heuristic"
```

---

## Task 9: Analyze module — run_analysis orchestrator with fake backend

**Files:**
- Modify: `analyze.py`
- Test: `tests/test_analyze.py`

**Purpose:** Wire everything together. `run_analysis(target, angle_hint, writer_callback, backend=None)` orchestrates: classify → load prior → build prompt → call backend → stream to writer → extract conviction → write memo → rebuild index.

The `backend` parameter defaults to `None` (use real Gemini). Tests inject a fake backend. `writer_callback(str)` is called for each streamed chunk so the terminal can display progress; it's a no-op callable in tests.

- [ ] **Step 9.1: Write the failing test**

Append to `tests/test_analyze.py`:
```python
class TestRunAnalysis:
    def test_happy_path_writes_memo(self, tmp_path, monkeypatch):
        import archive as archive_mod
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])

        fake_body = (
            "# AVGO — 2026-04-18\n\n"
            "## Context\nPrice 1720.\n\n"
            "## What Changed Since Last Memo\nFirst memo for this target.\n\n"
            "## Current Read\nHigh conviction. Moat widening.\n\n"
            "## Risks / Disconfirming Evidence\nCapex decel.\n\n"
            "## Suggested Actions\nNo action — hold.\n\n"
            "## Sources\nTool: get_quote.\n"
        )

        def fake_backend(model_cfg, history, system_prompt, images=None):
            # Yield the body in two chunks to exercise streaming
            yield (False, fake_body[:100])
            yield (False, fake_body[100:])

        writes: list[str] = []
        path = analyze.run_analysis(
            target="AVGO",
            angle_hint="",
            writer_callback=writes.append,
            backend=fake_backend,
        )
        assert path.endswith(".md")
        assert os.path.exists(path)
        # Writer got the streamed content
        assert "Moat widening" in "".join(writes)
        # File has front-matter + body
        text = open(path).read()
        assert "target: AVGO" in text
        assert "conviction:" in text
        assert "level: high" in text
        assert "Moat widening" in text

    def test_empty_response_writes_error_memo(self, tmp_path, monkeypatch):
        import archive as archive_mod
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])

        def empty_backend(model_cfg, history, system_prompt, images=None):
            yield (False, "")

        writes: list[str] = []
        path = analyze.run_analysis(
            target="AVGO", angle_hint="",
            writer_callback=writes.append, backend=empty_backend,
        )
        assert os.path.exists(path)
        text = open(path).read()
        assert "ERROR" in text or "empty" in text.lower()

    def test_backend_exception_writes_error_memo(self, tmp_path, monkeypatch):
        import archive as archive_mod
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])

        def boom_backend(model_cfg, history, system_prompt, images=None):
            raise RuntimeError("gemini down")
            yield  # make this a generator

        writes: list[str] = []
        path = analyze.run_analysis(
            target="AVGO", angle_hint="",
            writer_callback=writes.append, backend=boom_backend,
        )
        assert os.path.exists(path)
        text = open(path).read()
        assert "gemini down" in text
        assert "ERROR" in text

    def test_freeform_target_lands_under_freeform(self, tmp_path, monkeypatch):
        import archive as archive_mod
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: [])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])

        def fake_backend(model_cfg, history, system_prompt, images=None):
            yield (False,
                   "# Q — 2026-04-18\n## Current Read\nSome answer.\n")

        path = analyze.run_analysis(
            target="why is the market up",
            angle_hint="",
            writer_callback=lambda s: None,
            backend=fake_backend,
        )
        assert "_freeform" in path

    def test_thought_chunks_not_written_as_memo(self, tmp_path, monkeypatch):
        """Gemini emits (is_thought=True, "...") for chain-of-thought parts —
        those must NOT be concatenated into the memo body."""
        import archive as archive_mod
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["AVGO"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: [])

        def fake_backend(model_cfg, history, system_prompt, images=None):
            yield (True, "INTERNAL THINKING — should not appear in memo")
            yield (False, "# AVGO — 2026-04-18\n## Current Read\nHigh conviction. Good.\n")

        path = analyze.run_analysis(
            target="AVGO", angle_hint="",
            writer_callback=lambda s: None, backend=fake_backend,
        )
        text = open(path).read()
        assert "INTERNAL THINKING" not in text
        assert "High conviction" in text
```

- [ ] **Step 9.2: Run to verify it fails**

Run: `pytest tests/test_analyze.py::TestRunAnalysis -v`
Expected: AttributeError — `run_analysis` not defined.

- [ ] **Step 9.3: Implement run_analysis**

Append to `analyze.py`:
```python
import archive


def _default_backend():
    """Lazy import of real Gemini Pro backend."""
    import chat as chat_module
    return chat_module._BACKENDS["gemini"], chat_module.MODELS["pro"]


def run_analysis(target: str, angle_hint: str = "",
                 writer_callback=None, backend=None) -> str:
    """Execute a deep-dive and return path to the written memo.

    Args:
        target: raw user target (ticker, thesis name, or freeform text)
        angle_hint: optional focus hint; stored in front-matter's `angle`
        writer_callback: optional callable(str) called with each streamed
                         chunk as it arrives. Used by UI to show progress.
        backend: optional backend callable (for testing). When None, uses
                 chat_module._BACKENDS["gemini"] with MODELS["pro"].

    Returns:
        Absolute path to the written memo file.

    Never raises for backend failures — those are written as error memos
    instead, so failures remain auditable in the archive.
    """
    if writer_callback is None:
        writer_callback = lambda s: None  # noqa: E731

    kind, normalized = classify_target(target)
    angle = angle_hint.strip() or "general"
    slug = archive.slug_for_target(normalized, kind)
    prior_memos = archive.load_prior(slug)
    system_prompt = build_system_prompt(kind, normalized, angle, prior_memos)

    # Determine backend + model config
    if backend is None:
        backend_fn, model_cfg = _default_backend()
    else:
        backend_fn = backend
        model_cfg = {"id": GEMINI_PRO_MODEL_ID, "label": "Gemini Pro (test)",
                     "provider": "gemini", "thinking_budget": 2048}

    # Stream the response
    body_chunks: list[str] = []
    error_msg = None
    try:
        for is_thought, chunk in backend_fn(
            model_cfg,
            [{"role": "user", "text": f"Produce the memo for {normalized}."}],
            system_prompt,
        ):
            if is_thought is None or is_thought:
                continue
            body_chunks.append(chunk)
            writer_callback(chunk)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    body = "".join(body_chunks).strip()

    if error_msg or not body:
        # Build an error memo so failure is auditable
        reason = error_msg or "empty response from backend"
        body = (f"# {normalized} — ERROR\n\n"
                f"## Context\nAnalysis failed.\n\n"
                f"## What Changed Since Last Memo\nN/A.\n\n"
                f"## Current Read\nERROR: {reason}\n\n"
                f"## Risks / Disconfirming Evidence\nN/A.\n\n"
                f"## Suggested Actions\nRetry or check API keys.\n\n"
                f"## Sources\nN/A.\n")
        conviction = {"level": "unknown", "key_claim": f"ERROR: {reason}"}
    else:
        conviction = extract_conviction(body)

    fm = build_front_matter(
        kind=kind, target=normalized, angle=angle,
        prior_memos=prior_memos, tools_used=["grounding"],
        conviction=conviction,
    )
    path = archive.write_memo(slug, fm, body)
    archive.rebuild_index()
    return path
```

- [ ] **Step 9.4: Run to verify it passes**

Run: `pytest tests/test_analyze.py -v`
Expected: 26 passed (all tests in the module).

- [ ] **Step 9.5: Verify 604 existing tests still pass**

Run: `pytest -q 2>&1 | tail -3`
Expected: total count is `604 + new_test_count passed`, with zero failures. The exact total depends on how many tests you've added in Tasks 1-9 (should be ~42 new: 16 archive + 26 analyze). What matters: zero failures, and the original 604 must all still pass.

If anything fails: STOP, investigate. Do NOT modify existing tests.

- [ ] **Step 9.6: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat(analyze): run_analysis orchestrator with fake-backend tests"
```

---

## Task 10: Wire `analyze` command into command bar

**Files:**
- Modify: `widgets/command_bar.py:182` (add alias)
- Test: `tests/test_command_bar.py` (add assertion)

**Purpose:** Make `parse_command("analyze AVGO")` return `("analyze", ["AVGO"])`.

- [ ] **Step 10.1: Write the failing test**

Open `tests/test_command_bar.py`, find an existing test class, and ADD a new test (do not modify existing tests):

```python
class TestAnalyzeAlias:
    def test_analyze_returns_analyze_action(self):
        from widgets.command_bar import parse_command
        action, args = parse_command("analyze AVGO")
        assert action == "analyze"
        assert args == ["AVGO"]

    def test_analyze_with_multi_word_target(self):
        from widgets.command_bar import parse_command
        action, args = parse_command("analyze why is XLU up")
        assert action == "analyze"
        assert args == ["why", "is", "XLU", "up"]

    def test_analyze_alone_returns_empty_args(self):
        from widgets.command_bar import parse_command
        action, args = parse_command("analyze")
        assert action == "analyze"
        assert args == []
```

- [ ] **Step 10.2: Run to verify it fails**

Run: `pytest tests/test_command_bar.py::TestAnalyzeAlias -v`
Expected: failures — "analyze" currently falls to the `lookup` default.

- [ ] **Step 10.3: Add "analyze" to aliases**

In `widgets/command_bar.py`, find the `aliases = {` dict near line 125. Add this line (e.g., near the "brief" line for logical grouping):

```python
        "analyze": "analyze", "dive": "analyze",
```

- [ ] **Step 10.4: Run to verify it passes**

Run: `pytest tests/test_command_bar.py -v`
Expected: new tests pass; all existing `test_command_bar.py` tests still pass.

- [ ] **Step 10.5: Commit**

```bash
git add widgets/command_bar.py tests/test_command_bar.py
git commit -m "feat(analyze): parse_command alias 'analyze' and 'dive'"
```

---

## Task 11: Wire `analyze` command into app.py

**Files:**
- Modify: `app.py` (imports near top, new branch in `on_command_submitted`, new worker methods)

**Purpose:** When the user runs `analyze AVGO`, call `analyze.run_analysis` in a worker thread and stream the memo into the terminal.

This task does not add new tests — the behavior is exercised by existing `test_app.py` smoke tests (imports + command routing). The `run_analysis` logic itself is already covered.

- [ ] **Step 11.1: Add import at top of app.py**

Open `app.py`. Find the imports block (around where `from chat import` / `import chat as chat_module` happens). Add:

```python
import analyze as analyze_module
```

Run:
```bash
python -c "import app"
```
Expected: no import error.

- [ ] **Step 11.2: Add the command branch in `on_command_submitted`**

In `app.py`, find the line `elif action == "brief":` (around line 1761). Add the new branch just before it (so `analyze` is grouped with other AI-driven commands):

```python
        elif action == "analyze":
            if not args:
                self._write("[dim]Usage: analyze <SYM | thesis | topic>[/]")
            else:
                target = args[0]
                angle_hint = " ".join(args[1:])
                self._run_analyze(target, angle_hint)
```

- [ ] **Step 11.3: Add the `_run_analyze` worker method**

Find `_show_briefing_ai` in `app.py` (the @work(thread=True) decorated method around line 1250-1298). Add a new method right after it:

```python
    @work(thread=True)
    def _run_analyze(self, target: str, angle_hint: str) -> None:
        """Run a deep-dive analysis and stream the memo to the terminal."""
        display_target = target if not angle_hint else f"{target} {angle_hint}"
        self.app.call_from_thread(
            self._write,
            f"[bold #ffc800]═══ ANALYZE: {display_target} ═══[/]\n"
            f"[dim]Running deep-dive via Gemini Pro...[/]",
        )

        # Writer callback streams chunks to the terminal in real time.
        # We accumulate into a small buffer and flush on newlines so RichLog
        # doesn't get spammed with 1-char writes.
        buf: list[str] = []
        def _flush(chunk: str) -> None:
            buf.append(chunk)
            if "\n" in chunk or sum(len(c) for c in buf) > 200:
                text = "".join(buf)
                buf.clear()
                self.app.call_from_thread(self._write, text)

        try:
            path = analyze_module.run_analysis(
                target=target, angle_hint=angle_hint,
                writer_callback=_flush,
            )
            # Flush any remaining buffer
            if buf:
                self.app.call_from_thread(self._write, "".join(buf))
                buf.clear()
            self.app.call_from_thread(
                self._write,
                f"\n[dim]Memo saved: {path}[/]",
            )
        except Exception as e:
            self.app.call_from_thread(
                self._write,
                f"[#ff3232]analyze error: {e}[/]",
            )
```

- [ ] **Step 11.4: Run the app smoke test**

Run:
```bash
pytest tests/test_app.py -v 2>&1 | tail -20
```
Expected: all existing app tests still pass. If any fail, investigate — most likely an import ordering issue or syntax error in the added code.

Also run full suite:
```bash
pytest -q 2>&1 | tail -3
```
Expected: still green (now 630+ passed).

- [ ] **Step 11.5: Manual smoke test (optional but recommended)**

Run:
```bash
source venv/bin/activate
python -c "
import analyze
from unittest.mock import patch

def fake_backend(model_cfg, history, system_prompt, images=None):
    yield (False, '# AVGO — 2026-04-18\n## Context\nx\n## Current Read\nHigh conviction. Good.\n')

path = analyze.run_analysis('AVGO', '', writer_callback=print, backend=fake_backend)
print('Wrote:', path)
"
```
Expected: prints the streamed chunk, then `Wrote: .../data/analyses/AVGO/YYYY-MM-DD-HHMM.md`.

Clean up the test memo:
```bash
rm -rf data/analyses/AVGO
```

- [ ] **Step 11.6: Commit**

```bash
git add app.py
git commit -m "feat(analyze): wire analyze command into app.py"
```

---

## Task 12: Create data/analyses/.gitkeep

**Files:**
- Create: `data/analyses/.gitkeep`

**Purpose:** Ensure the archive directory exists in fresh clones.

- [ ] **Step 12.1: Create the gitkeep file**

Run:
```bash
mkdir -p data/analyses
touch data/analyses/.gitkeep
```

- [ ] **Step 12.2: Verify gitignore doesn't block it**

Run:
```bash
git check-ignore -v data/analyses/.gitkeep || echo "not ignored"
```
Expected: `not ignored`. If it IS ignored, the archive won't persist in clones — but do not modify `.gitignore` without confirming with the user, since there may be a reason generated JSON is ignored. Instead, note this as a followup and continue.

- [ ] **Step 12.3: Commit**

```bash
git add data/analyses/.gitkeep
git commit -m "chore: create data/analyses/ archive directory"
```

---

## Task 13: README section

**Files:**
- Modify: `README.md` (add new section)

**Purpose:** Document the `analyze` command for future Jeff and anyone else reading the repo.

- [ ] **Step 13.1: Read current README structure**

Run:
```bash
head -40 README.md
grep -n "^## " README.md
```
Note where existing feature sections live.

- [ ] **Step 13.2: Add the analyze section**

Add a new section (inserted in the features area of the README, location driven by existing structure). Use this content:

```markdown
## `analyze` — On-Demand Deep-Dives (v2.6+)

`analyze <target>` runs a doc-grade memo via Gemini Pro with full tool access
and Google grounding. Memos are streamed to the terminal and archived to
`data/analyses/{target}/{YYYY-MM-DD-HHMM}.md`.

```
analyze AVGO                # symbol dive
analyze rotation            # thesis dive (requires thesis key in config)
analyze MU earnings         # symbol with angle hint
analyze "why is XLU up"     # freeform
```

Each memo follows a fixed structure:

- **Context** — current price, position, thesis bucket
- **What Changed Since Last Memo** — explicit diff against your prior view
- **Current Read** — argued thesis status, no hedging
- **Risks / Disconfirming Evidence** — the bear steelman
- **Suggested Actions** — specific sizing/levels (or "no action — hold")
- **Sources** — cited from tools and grounding

Prior memos for the same target are loaded into the system prompt, so re-running
`analyze AVGO` a month later produces a "since last memo" update rather than a
restart. The archive builds a conviction timeline over time.
```

- [ ] **Step 13.3: Commit**

```bash
git add README.md
git commit -m "docs: README section for analyze command"
```

---

## Task 14: Open draft PR

**Files:** none — GitHub operation only.

- [ ] **Step 14.1: Push the branch**

Run:
```bash
git push -u origin feature/analyze-command
```

- [ ] **Step 14.2: Open draft PR**

Run:
```bash
gh pr create --draft --title "feat: v2.6 analyze command (deep-dive memos)" --body "$(cat <<'EOF'
## Summary
- New `analyze <target>` command: on-demand deep-dive memos via Gemini Pro
- Memos archived to `data/analyses/{slug}/{date}.md` with YAML front-matter
- Prior memos loaded into context for "since last memo" continuity
- New modules: `analyze.py` (orchestration), `archive.py` (disk I/O)

## Design
Full spec at `docs/superpowers/specs/2026-04-18-analyze-command-design.md`.
Implementation plan at `docs/superpowers/plans/2026-04-18-analyze-command.md`.

## Test plan
- [x] All 604 existing tests green
- [x] ~25 new tests (archive + analyze) passing
- [ ] Manual smoke test with real Gemini Pro key — run `analyze AVGO` end-to-end
- [ ] Re-run `analyze AVGO` with a prior memo present, verify "What Changed" references it
- [ ] Freeform: `analyze "why is XLU up"` produces a memo under `_freeform/`
- [ ] Verify no regression in existing commands (brief ai, chat, thesis, etc.)

## Rollback
Branch isolated from main. Merge via squash; revert is a single-commit revert.
EOF
)"
```

- [ ] **Step 14.3: Note the PR URL**

Capture the PR URL from `gh pr create` output. Share with the user.

---

## Task 15: Manual smoke test with real Gemini Pro

**Files:** none — runtime verification only.

- [ ] **Step 15.1: Verify GEMINI_API_KEY set**

Run:
```bash
[ -n "$GEMINI_API_KEY" ] && echo "set" || echo "MISSING"
```
If MISSING: ask the user to source the right env before proceeding.

- [ ] **Step 15.2: Run analyze end-to-end via Python**

Run:
```bash
source venv/bin/activate
python -c "
import analyze
path = analyze.run_analysis('AVGO', '', writer_callback=lambda s: print(s, end='', flush=True))
print()
print('---')
print('Wrote:', path)
"
```
Expected: a streaming memo appears (30-90s), then the path prints. Inspect the file:
```bash
cat $(ls -t data/analyses/AVGO/*.md | head -1) | head -60
```
Expected: valid YAML front-matter, all 6 sections present, conviction extracted.

- [ ] **Step 15.3: Run analyze via the TUI**

Run:
```bash
python app.py
```
At the command bar, type `analyze AVGO` and observe the memo streams into the terminal. Confirm:
- Header line `═══ ANALYZE: AVGO ═══` shows
- Memo streams chunk by chunk
- Final "Memo saved: ..." path prints
- No Python traceback anywhere

Exit with `q`.

- [ ] **Step 15.4: Run a second time to test "since last memo"**

Back in TUI, run `analyze AVGO` again. Confirm the memo's "What Changed Since Last Memo" section references the prior memo instead of saying "First memo for this target."

- [ ] **Step 15.5: Update PR test checklist**

In the PR, check off the manual test-plan items that passed:
```bash
gh pr edit --body "$(gh pr view --json body -q .body | sed 's/- \[ \] Manual smoke/- [x] Manual smoke/' | sed 's/- \[ \] Re-run/- [x] Re-run/' | sed 's/- \[ \] Freeform/- [x] Freeform/' | sed 's/- \[ \] Verify no regression/- [x] Verify no regression/')"
```

(Or edit the PR description manually in the browser.)

- [ ] **Step 15.6: Mark PR ready for review**

```bash
gh pr ready
```

---

## Task 16: Squash-merge PR

Only proceed after user has reviewed and approved the PR.

- [ ] **Step 16.1: Confirm merge authorization**

Ask user explicitly: "PR is green and smoke-tested. Approve squash-merge into main?"

Wait for explicit approval.

- [ ] **Step 16.2: Squash-merge**

Run:
```bash
gh pr merge --squash --delete-branch
git checkout main
git pull origin main
git log --oneline -2
```
Expected: new squash commit on main with all the feature work; `feature/analyze-command` branch deleted locally and remote.

- [ ] **Step 16.3: Update project memory**

Update `~/.claude/projects/-Users-jeffbai-repos/memory/project_ticker_tape.md`:
- Bump version note to mention v2.6 analyze landed
- Record new commit hash
- Note new modules `analyze.py`, `archive.py`
- Note new archive directory `data/analyses/`

- [ ] **Step 16.4: Deploy to fragserv (per standing instruction)**

Run:
```bash
ssh baila@fragserv 'wsl -- bash -c "cd ~/repos/ticker-tape && git pull origin main && git log --oneline -2"'
```
Expected: fragserv shows the new squash commit as HEAD.

---

## Success Criteria Recap

- [x] All 604 existing tests still green on branch at every commit
- [x] ~25 new tests (archive + analyze + command_bar) green
- [x] `analyze AVGO` produces a memo with full front-matter + all 6 sections
- [x] Re-running references prior memo in "What Changed"
- [x] No edits to `chat.py`, `TICKER_TOOLS`, existing tests, or existing data dirs
- [x] PR diff reviewable in one sitting
- [x] Feature lands on main via squash-merge with clean revert path
