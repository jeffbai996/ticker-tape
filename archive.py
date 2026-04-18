"""Archive module — pure disk I/O for data/analyses/.

Writes memos, loads prior memos, rebuilds the index. No network calls,
no AI. All functions are pure given the filesystem state.
"""

import hashlib
import json
import os
import sys

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
    """Convert ISO-8601 date to YYYY-MM-DD-HHMMSS for filename.

    Seconds precision prevents same-minute collisions when two `analyze`
    invocations land back-to-back (e.g. manual retry after an error memo).
    """
    # "2026-04-18T14:23:05-04:00" → "2026-04-18-142305"
    date_part, time_part = iso.split("T")
    segs = time_part.split(":")
    hh = segs[0]
    mm = segs[1] if len(segs) > 1 else "00"
    # segs[2] includes seconds + optional tz offset; strip non-digits
    ss_raw = segs[2] if len(segs) > 2 else "00"
    ss = "".join(c for c in ss_raw if c.isdigit())[:2].zfill(2)
    return f"{date_part}-{hh}{mm}{ss}"


def _unique_memo_path(dir_path: str, base: str) -> str:
    """Return a non-colliding path: base.md, or base-2.md, -3.md, ..."""
    path = os.path.join(dir_path, f"{base}.md")
    if not os.path.exists(path):
        return path
    for suffix in range(2, 1000):
        path = os.path.join(dir_path, f"{base}-{suffix}.md")
        if not os.path.exists(path):
            return path
    raise RuntimeError(f"Too many memos in one second at {dir_path}/{base}")


def write_memo(slug: str, front_matter: dict, body: str) -> str:
    """Write a memo to disk under ARCHIVE_ROOT/{slug}/.

    Filename derived from front_matter['date'] as YYYY-MM-DD-HHMMSS.md.
    On collision, appends `-2`, `-3`, etc. Creates directory if missing.
    Returns the absolute path written.
    """
    dir_path = os.path.join(ARCHIVE_ROOT, slug)
    os.makedirs(dir_path, exist_ok=True)
    base = _timestamp_from_iso(front_matter["date"])
    path = _unique_memo_path(dir_path, base)

    fm_yaml = yaml.safe_dump(front_matter, sort_keys=False,
                             default_flow_style=None, allow_unicode=True)
    content = f"---\n{fm_yaml}---\n\n{body}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path


def _parse_memo_file(path: str) -> dict | None:
    """Read a memo, split front-matter from body. Return None if malformed."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not text.startswith("---\n"):
            return None
        _, fm_text, body = text.split("---\n", 2)
        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict) or "date" not in fm:
            return None
        return {"path": path, "front_matter": fm, "body": body.lstrip("\n")}
    except Exception as e:
        print(f"[archive] skipping malformed memo {path}: {e}", file=sys.stderr)
        return None


def load_prior(slug: str) -> list[dict]:
    """Load all prior memos for a slug, newest-first.

    Returns list of {"path": str, "front_matter": dict, "body": str}.
    Malformed memos are skipped with a stderr warning.
    """
    dir_path = os.path.join(ARCHIVE_ROOT, slug)
    if not os.path.isdir(dir_path):
        return []
    memos = []
    for name in os.listdir(dir_path):
        if not name.endswith(".md"):
            continue
        parsed = _parse_memo_file(os.path.join(dir_path, name))
        if parsed is not None:
            memos.append(parsed)
    # Sort newest-first by front_matter.date
    memos.sort(key=lambda m: m["front_matter"]["date"], reverse=True)
    return memos


def _first_body_paragraph(body: str) -> str:
    """Return the first non-empty, non-H1 paragraph of a memo body.

    Memos start with `# {target} — {date}` which is useless as a summary.
    Skip it and take the next paragraph.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    for p in paragraphs:
        if not p.lstrip().startswith("# "):
            return p
    return ""


def _index_entries_for_slug(slug: str) -> list[dict]:
    """Return lightweight index entries (no full body) for a slug."""
    entries = []
    for memo in load_prior(slug):
        fm = memo["front_matter"]
        first = _first_body_paragraph(memo["body"])
        summary = first if len(first) <= 200 else first[:199] + "…"
        entries.append({
            "date": fm["date"],
            "path": os.path.relpath(memo["path"], ARCHIVE_ROOT),
            "target": fm.get("target", ""),
            "kind": fm.get("kind", ""),
            "conviction": fm.get("conviction", {}),
            "summary": summary,
        })
    return entries


def rebuild_index() -> str:
    """Walk ARCHIVE_ROOT, produce _index.json. Returns path to index file."""
    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    index: dict[str, list[dict]] = {}
    for slug_name in sorted(os.listdir(ARCHIVE_ROOT)):
        slug_dir = os.path.join(ARCHIVE_ROOT, slug_name)
        if not os.path.isdir(slug_dir) or slug_name.startswith("."):
            continue
        # Handle nested _freeform/<hash>/ layout
        if slug_name == "_freeform":
            for sub in sorted(os.listdir(slug_dir)):
                sub_path = os.path.join(slug_dir, sub)
                if os.path.isdir(sub_path):
                    key = f"_freeform/{sub}"
                    index[key] = _index_entries_for_slug(key)
            continue
        index[slug_name] = _index_entries_for_slug(slug_name)
    path = os.path.join(ARCHIVE_ROOT, "_index.json")
    # Atomic replace: write to .tmp, then rename. Prevents a concurrent
    # reader from seeing a half-written file if two analyze calls overlap.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_path, path)
    return path
