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
