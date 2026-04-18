"""Tests for archive.py — pure disk I/O over data/analyses/."""

import json
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

    def test_load_prior_skips_malformed(self, tmp_path, monkeypatch):
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
        with open(str(tmp_path / "AVGO" / "bad.md"), "w", encoding="utf-8") as f:
            f.write("no front matter here\n")
        priors = archive_mod.load_prior("AVGO")
        assert len(priors) == 1
        assert priors[0]["front_matter"]["target"] == "AVGO"

    def test_load_prior_skips_front_matter_without_date(self, tmp_path, monkeypatch):
        """Front-matter block present but missing the required `date` key."""
        monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", str(tmp_path))
        os.makedirs(str(tmp_path / "AVGO"))
        with open(str(tmp_path / "AVGO" / "nodate.md"), "w", encoding="utf-8") as f:
            f.write("---\ntarget: AVGO\nkind: symbol\n---\nbody\n")
        assert archive_mod.load_prior("AVGO") == []

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
