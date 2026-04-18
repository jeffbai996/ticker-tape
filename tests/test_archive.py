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
