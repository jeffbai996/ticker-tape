"""Tests for fill_notes — sidecar storage, key stability, annotation."""

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import fill_notes as fn


@dataclass(frozen=True)
class _Fill:
    date: date
    symbol: str
    side: str
    qty: Decimal
    price: Decimal


def _tmp(tmp_path):
    return os.path.join(str(tmp_path), "fill_notes.json")


def test_key_is_stable_across_equivalent_values():
    k1 = fn.fill_key(date(2026, 1, 5), "AAPL", "BUY", Decimal("100"), Decimal("150.00"))
    k2 = fn.fill_key("2026-01-05", " aapl ", "buy", 100, 150.0)
    assert k1 == k2  # normalization collapses equivalents
    assert len(k1) == 12


def test_key_differs_on_different_fills():
    k1 = fn.fill_key(date(2026, 1, 5), "AAPL", "BUY", 100, 150)
    k2 = fn.fill_key(date(2026, 1, 5), "AAPL", "SELL", 100, 150)
    assert k1 != k2


def test_save_load_roundtrip(tmp_path):
    p = _tmp(tmp_path)
    fn.save_notes({"abc": "starter position"}, p)
    assert fn.load_notes(p) == {"abc": "starter position"}


def test_missing_file_returns_empty(tmp_path):
    assert fn.load_notes(_tmp(tmp_path)) == {}


def test_corrupt_file_returns_empty(tmp_path):
    p = _tmp(tmp_path)
    with open(p, "w") as fh:
        fh.write("{not valid json")
    assert fn.load_notes(p) == {}


def test_set_and_remove_note(tmp_path):
    p = _tmp(tmp_path)
    fn.set_note("k1", "trimmed into strength", p)
    assert fn.load_notes(p)["k1"] == "trimmed into strength"
    # Empty text removes.
    fn.set_note("k1", "   ", p)
    assert "k1" not in fn.load_notes(p)


def test_remove_note_reports_existence(tmp_path):
    p = _tmp(tmp_path)
    fn.set_note("k1", "note", p)
    assert fn.remove_note("k1", p) is True
    assert fn.remove_note("k1", p) is False


def test_note_attached_to_correct_fill(tmp_path):
    p = _tmp(tmp_path)
    f1 = _Fill(date(2026, 1, 5), "AAPL", "BUY", Decimal("100"), Decimal("150"))
    f2 = _Fill(date(2026, 2, 1), "MSFT", "SELL", Decimal("50"), Decimal("300"))
    key1 = fn.fill_key(f1.date, f1.symbol, f1.side, f1.qty, f1.price)
    fn.set_note(key1, "my thesis", p)
    notes = fn.load_notes(p)
    assert fn.note_for_fill(f1, notes) == "my thesis"
    assert fn.note_for_fill(f2, notes) is None


def test_annotate_fills(tmp_path):
    p = _tmp(tmp_path)
    f1 = _Fill(date(2026, 1, 5), "AAPL", "BUY", Decimal("100"), Decimal("150"))
    f2 = _Fill(date(2026, 2, 1), "MSFT", "SELL", Decimal("50"), Decimal("300"))
    key1 = fn.fill_key(f1.date, f1.symbol, f1.side, f1.qty, f1.price)
    fn.set_note(key1, "starter", p)
    rows = fn.annotate_fills([f1, f2], p)
    assert rows[0]["note"] == "starter"
    assert rows[1]["note"] is None
    assert rows[0]["key"] == key1


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = _tmp(tmp_path)
    fn.save_notes({"a": "b"}, p)
    leftovers = [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert leftovers == []
