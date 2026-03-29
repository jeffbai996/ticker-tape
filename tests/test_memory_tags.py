"""Tests for AI memory tag parsing and history delete range parsing."""

import re


# ── Memory tag regexes (same patterns used in app.py _stream_chat) ──

_RE_MEMORY_SAVE = re.compile(r"\[MEMORY:\s*(.+?)\]")
_RE_MEMORY_DELETE = re.compile(r"\[MEMORY_DELETE:\s*(\d+)\]")
_RE_MEMORY_STRIP = re.compile(r"\[MEMORY(?:_DELETE)?:\s*.+?\]")
_RE_DELETE_RANGE = re.compile(r"^(\d+)-(\d+)$")


class TestMemorySaveTag:
    def test_basic_save(self):
        text = "Done. [MEMORY: MU reports June 25] I've noted that."
        match = _RE_MEMORY_SAVE.search(text)
        assert match
        assert match.group(1) == "MU reports June 25"

    def test_save_with_extra_spaces(self):
        match = _RE_MEMORY_SAVE.search("[MEMORY:   AVGO target $250  ]")
        assert match
        assert match.group(1).strip() == "AVGO target $250"

    def test_multiple_saves(self):
        text = "[MEMORY: fact one] and [MEMORY: fact two]"
        matches = _RE_MEMORY_SAVE.findall(text)
        assert matches == ["fact one", "fact two"]

    def test_no_match_on_empty(self):
        assert not _RE_MEMORY_SAVE.search("no tags here")

    def test_no_match_on_malformed(self):
        assert not _RE_MEMORY_SAVE.search("[MEMORY:]")

    def test_save_with_special_chars(self):
        match = _RE_MEMORY_SAVE.search("[MEMORY: P/E < 15, yield > 3%]")
        assert match
        assert "P/E" in match.group(1)


class TestMemoryDeleteTag:
    def test_basic_delete(self):
        match = _RE_MEMORY_DELETE.search("Done. [MEMORY_DELETE: 5] Removed.")
        assert match
        assert match.group(1) == "5"

    def test_multi_digit_id(self):
        match = _RE_MEMORY_DELETE.search("[MEMORY_DELETE: 42]")
        assert match
        assert match.group(1) == "42"

    def test_no_match_on_text_id(self):
        assert not _RE_MEMORY_DELETE.search("[MEMORY_DELETE: abc]")

    def test_multiple_deletes(self):
        text = "[MEMORY_DELETE: 3] and [MEMORY_DELETE: 7]"
        matches = _RE_MEMORY_DELETE.findall(text)
        assert matches == ["3", "7"]


class TestMemoryStripTag:
    def test_strip_save_tag(self):
        text = "Sure. [MEMORY: some fact] Done."
        assert _RE_MEMORY_STRIP.sub("", text) == "Sure.  Done."

    def test_strip_delete_tag(self):
        text = "Removed. [MEMORY_DELETE: 5] Gone."
        assert _RE_MEMORY_STRIP.sub("", text) == "Removed.  Gone."

    def test_strip_both_tags(self):
        text = "[MEMORY: new fact] and [MEMORY_DELETE: 3]"
        assert _RE_MEMORY_STRIP.sub("", text).strip() == "and"

    def test_no_tags_unchanged(self):
        text = "No special tags here."
        assert _RE_MEMORY_STRIP.sub("", text) == text


class TestHistoryDeleteRange:
    def test_single_number_not_range(self):
        assert not _RE_DELETE_RANGE.match("5")

    def test_valid_range(self):
        m = _RE_DELETE_RANGE.match("6-10")
        assert m
        assert int(m.group(1)) == 6
        assert int(m.group(2)) == 10

    def test_single_item_range(self):
        m = _RE_DELETE_RANGE.match("3-3")
        assert m
        assert int(m.group(1)) == 3

    def test_large_range(self):
        m = _RE_DELETE_RANGE.match("1-50")
        assert m
        assert int(m.group(1)) == 1
        assert int(m.group(2)) == 50

    def test_no_match_text(self):
        assert not _RE_DELETE_RANGE.match("abc")

    def test_no_match_partial(self):
        assert not _RE_DELETE_RANGE.match("6-")
        assert not _RE_DELETE_RANGE.match("-10")
