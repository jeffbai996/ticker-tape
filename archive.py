"""Archive module — pure disk I/O for data/analyses/.

Writes memos, loads prior memos, rebuilds the index. No network calls,
no AI. All functions are pure given the filesystem state.
"""

import hashlib
import os

import yaml


ARCHIVE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "analyses"
)


def slug_for_target(target: str, kind: str) -> str:
    """Convert a target + kind into a filesystem-safe slug.

    symbol → uppercase ticker (AVGO)
    thesis → lowercase name (rotation)
    freeform → _freeform/<8-char sha256 hex prefix>
    """
    if kind == "symbol":
        return target.upper()
    if kind == "thesis":
        return target.lower()
    # freeform
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:8]
    return f"_freeform/{digest}"


def _timestamp_from_iso(iso: str) -> str:
    """Convert ISO-8601 date to YYYY-MM-DD-HHMM for filename."""
    # Parse "2026-04-18T14:23:00-04:00" → "2026-04-18-1423"
    date_part, time_part = iso.split("T")
    hhmm = time_part.split(":")[0] + time_part.split(":")[1]
    return f"{date_part}-{hhmm}"


def write_memo(slug: str, front_matter: dict, body: str) -> str:
    """Write a memo to disk under ARCHIVE_ROOT/{slug}/.

    Filename derived from front_matter['date'] as YYYY-MM-DD-HHMM.md.
    Creates directory if missing.
    Returns the absolute path written.
    """
    dir_path = os.path.join(ARCHIVE_ROOT, slug)
    os.makedirs(dir_path, exist_ok=True)
    filename = _timestamp_from_iso(front_matter["date"]) + ".md"
    path = os.path.join(dir_path, filename)

    fm_yaml = yaml.safe_dump(front_matter, sort_keys=False,
                             default_flow_style=None, allow_unicode=True)
    content = f"---\n{fm_yaml}---\n\n{body}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
