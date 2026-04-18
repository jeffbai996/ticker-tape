"""Tests for archive.py — pure disk I/O over data/analyses/."""

import os

import archive
import archive as archive_mod


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
        text = open(path, encoding="utf-8").read()
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
