"""Shared JSON list persistence — used by memory and journal modules."""

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class JsonStore:
    """Persistent list-of-dicts store with auto-incrementing IDs."""

    def __init__(self, file_path: str, max_entries: int) -> None:
        self.file_path = file_path
        self.max_entries = max_entries

    def load(self) -> list[dict]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load %s: %s", self.file_path, e)
            return []

    def save(self, entries: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            with open(self.file_path, "w") as f:
                json.dump(entries, f, indent=2)
        except OSError as e:
            log.warning("Failed to save %s: %s", self.file_path, e)

    def next_id(self, entries: list[dict]) -> int:
        if not entries:
            return 1
        return max(e.get("id", 0) for e in entries) + 1

    def add(self, extra_fields: dict) -> dict:
        """Append an entry with auto ID and timestamp. Returns the new entry."""
        entries = self.load()
        entry = {
            "id": self.next_id(entries),
            "ts": datetime.now(timezone.utc).isoformat(),
            **extra_fields,
        }
        entries.append(entry)
        if len(entries) > self.max_entries:
            entries = entries[-self.max_entries:]
        self.save(entries)
        return entry

    def update(self, entry_id: int, fields: dict) -> bool:
        """Update fields of an entry by ID. Returns True if found."""
        entries = self.load()
        for e in entries:
            if e.get("id") == entry_id:
                e.update(fields)
                self.save(entries)
                return True
        return False

    def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        entries = self.load()
        count = len(entries)
        self.save([])
        return count

    def remove(self, entry_id: int) -> bool:
        """Remove an entry by ID. Resequences remaining IDs."""
        entries = self.load()
        before = len(entries)
        entries = [e for e in entries if e.get("id") != entry_id]
        if len(entries) == before:
            return False
        # Resequence IDs to stay contiguous (1, 2, 3, ...)
        for i, e in enumerate(entries, 1):
            e["id"] = i
        self.save(entries)
        return True
