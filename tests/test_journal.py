"""Tests for the trade journal module."""

import json
import os
import pytest
import journal


@pytest.fixture(autouse=True)
def tmp_journal(tmp_path, monkeypatch):
    """Redirect journal to a temp file for each test."""
    path = str(tmp_path / "journal.json")
    monkeypatch.setattr(journal, "JOURNAL_FILE", path)
    return path


class TestAddEntry:
    def test_basic_add(self):
        entry = journal.add_entry("Sold 200 AVGO at $185")
        assert entry["id"] == 1
        assert "AVGO" in entry["text"]
        assert entry["ts"]

    def test_sequential_ids(self):
        journal.add_entry("First")
        e2 = journal.add_entry("Second")
        assert e2["id"] == 2

    def test_symbol_extraction(self):
        entry = journal.add_entry("Trimmed NVDA and MU for margin relief")
        assert "NVDA" in entry["symbols"]
        assert "MU" in entry["symbols"]

    def test_stopwords_excluded(self):
        entry = journal.add_entry("I SELL MY AVGO FOR A GOOD PRICE")
        assert "AVGO" in entry["symbols"]
        assert "SELL" not in entry["symbols"]
        assert "GOOD" not in entry["symbols"]
        assert "FOR" not in entry["symbols"]


class TestLoadEntries:
    def test_empty_when_no_file(self):
        assert journal.load_entries() == []

    def test_round_trip(self):
        journal.add_entry("Test entry")
        entries = journal.load_entries()
        assert len(entries) == 1
        assert entries[0]["text"] == "Test entry"


class TestRemoveEntry:
    def test_remove_existing(self):
        journal.add_entry("To be deleted")
        assert journal.remove_entry(1) is True
        assert journal.load_entries() == []

    def test_remove_nonexistent(self):
        assert journal.remove_entry(999) is False

    def test_remove_preserves_others(self):
        journal.add_entry("Keep this")
        journal.add_entry("Delete this")
        journal.remove_entry(2)
        entries = journal.load_entries()
        assert len(entries) == 1
        assert entries[0]["text"] == "Keep this"


class TestSearchEntries:
    def test_search_finds_match(self):
        journal.add_entry("Bought NVDA at $118")
        journal.add_entry("Sold AVGO at $185")
        results = journal.search_entries("NVDA")
        assert len(results) == 1
        assert "NVDA" in results[0]["text"]

    def test_search_case_insensitive(self):
        journal.add_entry("Margin cushion dropping fast")
        results = journal.search_entries("margin")
        assert len(results) == 1

    def test_search_no_match(self):
        journal.add_entry("Some entry")
        assert journal.search_entries("nonexistent") == []


class TestSymbolExtraction:
    def test_common_tickers(self):
        syms = journal._extract_symbols("Looking at AAPL MSFT GOOGL")
        assert "AAPL" in syms
        assert "MSFT" in syms
        assert "GOOGL" in syms

    def test_no_duplicates(self):
        syms = journal._extract_symbols("NVDA is great, NVDA is king")
        assert syms.count("NVDA") == 1

    def test_mixed_text(self):
        syms = journal._extract_symbols("I think AVGO will hit $250 by Q3")
        assert "AVGO" in syms
        assert "I" not in syms
