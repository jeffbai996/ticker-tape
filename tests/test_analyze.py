"""Tests for analyze.py — target classification and AI orchestration."""

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

    def test_watchlist_wins_over_thesis_collision(self, monkeypatch):
        """If the same name exists as both a ticker and a thesis key,
        symbol wins (checked first). Pins rule precedence."""
        monkeypatch.setattr(analyze, "_watchlist_symbols", lambda: ["NVDA"])
        monkeypatch.setattr(analyze, "_thesis_keys", lambda: ["nvda"])
        kind, normalized = analyze.classify_target("NVDA")
        assert kind == "symbol"
        assert normalized == "NVDA"


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

    def test_freeform_kind_renders_question(self):
        """Freeform branch must surface the raw question in kind_guidance."""
        prompt = analyze.build_system_prompt(
            "freeform", "why is XLU up", "general", [])
        assert "why is XLU up" in prompt
        assert "freeform" in prompt.lower()

    def test_prior_memos_capped_at_limit(self):
        """Only the first MAX_PRIOR_MEMOS_IN_PROMPT priors appear in prompt."""
        cap = analyze.MAX_PRIOR_MEMOS_IN_PROMPT
        priors = [
            {"path": f"/x/AVGO/{i}.md",
             "front_matter": {
                 "date": f"2026-04-{(cap + 5) - i:02d}T10:00:00-04:00",
                 "conviction": {"level": "high", "key_claim": f"claim-{i}"},
             },
             "body": ""}
            for i in range(cap + 3)
        ]
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", priors)
        for i in range(cap):
            assert f"claim-{i}" in prompt
        # Entries beyond the cap must not appear
        for i in range(cap, cap + 3):
            assert f"claim-{i}" not in prompt

    def test_prior_memo_missing_conviction_uses_defaults(self):
        """Prior memo whose front-matter lacks conviction renders `?` defaults
        rather than crashing."""
        priors = [{
            "path": "/x/AVGO/2026-03-12-0915.md",
            "front_matter": {"date": "2026-03-12T09:15:00-04:00"},
            "body": "",
        }]
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", priors)
        assert "conviction=?" in prompt

    def test_prior_memo_key_claim_with_embedded_quotes(self):
        """Embedded double quotes in key_claim must be escaped so the list
        item stays well-formed."""
        priors = [{
            "path": "/x/AVGO/2026-03-12-0915.md",
            "front_matter": {
                "date": "2026-03-12T09:15:00-04:00",
                "conviction": {
                    "level": "high",
                    "key_claim": 'Jensen said "the more you buy" again',
                },
            },
            "body": "",
        }]
        prompt = analyze.build_system_prompt("symbol", "AVGO", "general", priors)
        assert '\\"the more you buy\\"' in prompt


class TestBuildFrontMatter:
    def test_basic_fields_populated(self):
        fm = analyze.build_front_matter(
            kind="symbol", target="AVGO", angle="general",
            prior_memos=[], tools_used=["get_quote"],
            conviction={"level": "high", "key_claim": "moat widening"},
            trigger_type="manual",
        )
        import chat
        import datetime as _dt
        assert fm["target"] == "AVGO"
        assert fm["kind"] == "symbol"
        assert fm["angle"] == "general"
        # Model id sourced from chat.MODELS so it never drifts from the
        # version actually running the analysis
        assert fm["model"] == chat.MODELS["pro"]["id"]
        assert fm["tools_used"] == ["get_quote"]
        assert fm["conviction"]["level"] == "high"
        assert fm["trigger_type"] == "manual"
        # ISO-8601 date must actually parse and carry a timezone
        parsed = _dt.datetime.fromisoformat(fm["date"])
        assert parsed.tzinfo is not None

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
