"""Shared fixtures for ticker_tape tests."""

import json
import os
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_json(tmp_path):
    """Return a factory that creates temp JSON files."""
    def _make(data, name="test.json"):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return str(path)
    return _make


@pytest.fixture
def alerts_file(tmp_path, monkeypatch):
    """Provide a temp alerts file and patch ALERTS_FILE to use it."""
    path = str(tmp_path / "alerts.json")
    monkeypatch.setattr("data.ALERTS_FILE", path)
    return path


@pytest.fixture
def watchlist_file(tmp_path, monkeypatch):
    """Provide a temp watchlist file and patch WATCHLIST_FILE to use it."""
    path = str(tmp_path / "watchlist.json")
    monkeypatch.setattr("data.WATCHLIST_FILE", path)
    return path
