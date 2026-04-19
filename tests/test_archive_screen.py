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


class TestFormatSlugMemos:
    def _entry(self, level="high", claim="c", date="2026-04-19T14:00:00-04:00"):
        return {
            "date": date, "path": "AVGO/a.md",
            "target": "AVGO", "kind": "symbol",
            "conviction": {"level": level, "key_claim": claim},
            "summary": "s",
        }

    def test_empty_entries_shows_not_found(self):
        result = format_slug_memos("FOO", [])
        assert "FOO" in result

    def test_header_shows_slug(self):
        result = format_slug_memos("AVGO", [self._entry()])
        assert "AVGO" in result

    def test_numbered_rows_start_at_1(self):
        entries = [self._entry(), self._entry(), self._entry()]
        result = format_slug_memos("AVGO", entries)
        body_rows = [l for l in result.split("\n") if "2026-04-19" in l]
        assert len(body_rows) == 3
        assert body_rows[0].lstrip().startswith("1")
        assert body_rows[-1].lstrip().startswith("3")

    def test_conviction_high_green(self):
        result = format_slug_memos("AVGO", [self._entry(level="high")])
        assert "#32ff32" in result

    def test_conviction_medium_amber(self):
        result = format_slug_memos("AVGO", [self._entry(level="medium")])
        assert "#ffc800" in result

    def test_conviction_low_red(self):
        result = format_slug_memos("AVGO", [self._entry(level="low")])
        assert "#ff3232" in result

    def test_conviction_unknown_dim(self):
        result = format_slug_memos("AVGO", [self._entry(level="unknown")])
        assert "[dim]" in result

    def test_long_key_claim_truncated_with_ellipsis(self):
        long_claim = "x" * 500
        result = format_slug_memos("AVGO", [self._entry(claim=long_claim)])
        assert "…" in result
        assert "x" * 500 not in result

    def test_date_is_YYYY_MM_DD_only(self):
        """Per-slug table shows date only, not the full ISO timestamp."""
        result = format_slug_memos("AVGO", [self._entry(date="2026-04-19T14:23:05-04:00")])
        assert "2026-04-19" in result
        assert "14:23:05" not in result


class TestFormatMemoView:
    def _memo(self, front_matter=None, body=""):
        fm = front_matter or {
            "target": "AVGO", "kind": "symbol", "angle": "general",
            "date": "2026-04-19T14:23:00-04:00",
            "model": "gemini-3.1-pro-preview",
            "prior_memos": [], "tools_used": [],
            "conviction": {"level": "high", "key_claim": "Custom silicon moat widening"},
            "trigger_type": "manual",
        }
        return {"path": "/x/y.md", "front_matter": fm, "body": body}

    def test_banner_contains_target_and_date(self):
        result = format_memo_view(self._memo())
        assert "AVGO" in result
        assert "2026-04-19" in result

    def test_banner_contains_conviction_level(self):
        result = format_memo_view(self._memo())
        assert "HIGH" in result

    def test_banner_contains_key_claim(self):
        result = format_memo_view(self._memo())
        assert "Custom silicon moat widening" in result

    def test_banner_contains_angle_and_model(self):
        result = format_memo_view(self._memo())
        assert "general" in result
        assert "gemini-3.1-pro-preview" in result

    def test_body_rendered_via_md_to_rich(self):
        """Body should be passed through _md_to_rich — headings become bold cyan."""
        body = "# AVGO — 2026-04-19\n\n## Context\nSome context here.\n"
        result = format_memo_view(self._memo(body=body))
        # _md_to_rich converts ## headings to bold #00c8ff
        assert "#00c8ff" in result
        assert "Context" in result

    def test_missing_conviction_shows_unknown(self):
        fm = {
            "target": "X", "kind": "symbol", "angle": "general",
            "date": "2026-04-19T14:23:00-04:00", "model": "m",
            "prior_memos": [], "tools_used": [],
            "conviction": {},  # empty
            "trigger_type": "manual",
        }
        result = format_memo_view({"path": "/x.md", "front_matter": fm, "body": ""})
        assert "UNKNOWN" in result

    def test_prior_memos_count_shown(self):
        fm = {
            "target": "AVGO", "kind": "symbol", "angle": "general",
            "date": "2026-04-19T14:23:00-04:00", "model": "m",
            "prior_memos": ["a.md", "b.md", "c.md"],
            "tools_used": [],
            "conviction": {"level": "high", "key_claim": "x"},
            "trigger_type": "manual",
        }
        result = format_memo_view({"path": "/x.md", "front_matter": fm, "body": ""})
        assert "3 prior" in result or "3" in result
