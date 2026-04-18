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
