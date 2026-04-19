"""Tests for screens/archive_screen.py — rendering of archive views."""

from screens.archive_screen import (
    format_archive_list,
    format_slug_memos,
    format_memo_view,
)


class TestFormatArchiveList:
    def test_empty_shows_empty_state(self):
        result = format_archive_list({})
        assert "No memos yet" in result or "archive.empty" in result

    def test_contains_all_slugs(self):
        index = {
            "AVGO": [
                {"date": "2026-04-19T14:00:00-04:00", "path": "AVGO/a.md",
                 "target": "AVGO", "kind": "symbol",
                 "conviction": {"level": "high", "key_claim": "x"},
                 "summary": "s"}
            ],
            "MU": [
                {"date": "2026-04-18T10:00:00-04:00", "path": "MU/a.md",
                 "target": "MU", "kind": "symbol",
                 "conviction": {"level": "medium", "key_claim": "y"},
                 "summary": "s"}
            ],
        }
        result = format_archive_list(index)
        assert "AVGO" in result
        assert "MU" in result

    def test_sorted_by_newest_date_desc(self):
        index = {
            "OLDER": [
                {"date": "2026-01-01T00:00:00-04:00", "path": "OLDER/a.md",
                 "target": "OLDER", "kind": "symbol",
                 "conviction": {"level": "high", "key_claim": "x"}, "summary": "s"}
            ],
            "NEWER": [
                {"date": "2026-04-19T00:00:00-04:00", "path": "NEWER/a.md",
                 "target": "NEWER", "kind": "symbol",
                 "conviction": {"level": "high", "key_claim": "x"}, "summary": "s"}
            ],
        }
        result = format_archive_list(index)
        assert result.index("NEWER") < result.index("OLDER")

    def test_memo_count_shown(self):
        index = {
            "AVGO": [
                {"date": "2026-04-19T14:00:00-04:00", "path": f"AVGO/{i}.md",
                 "target": "AVGO", "kind": "symbol",
                 "conviction": {"level": "high", "key_claim": "x"}, "summary": "s"}
                for i in range(3)
            ],
        }
        result = format_archive_list(index)
        avgo_line = next(l for l in result.split("\n") if "AVGO" in l)
        assert "3" in avgo_line

    def test_newest_date_shown_as_YYYY_MM_DD(self):
        index = {
            "AVGO": [
                {"date": "2026-04-19T14:00:00-04:00", "path": "AVGO/a.md",
                 "target": "AVGO", "kind": "symbol",
                 "conviction": {"level": "high", "key_claim": "x"}, "summary": "s"}
            ],
        }
        result = format_archive_list(index)
        assert "2026-04-19" in result

    def test_freeform_slug_shown_in_full(self):
        """Freeform slugs must NOT be truncated — otherwise the user can't copy
        the slug from the listing to `memos <slug>`."""
        index = {
            "_freeform/ae098c01": [
                {"date": "2026-04-19T14:00:00-04:00",
                 "path": "_freeform/ae098c01/a.md",
                 "target": "why is XLU up", "kind": "freeform",
                 "conviction": {"level": "low", "key_claim": "x"},
                 "summary": "s"}
            ],
        }
        result = format_archive_list(index)
        assert "_freeform/ae098c01" in result
