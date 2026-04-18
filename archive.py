"""Archive module — pure disk I/O for data/analyses/.

Writes memos, loads prior memos, rebuilds the index. No network calls,
no AI. All functions are pure given the filesystem state.
"""

import hashlib
import os


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
