"""Archive navigation views — pure rendering from the _index.json dict."""

from rich.markup import escape

from archive import format_conviction_color
from i18n import t


_HEADER_COLOR = "#ffc800"    # amber — matches other ═══ banners


def _newest_date(entries: list[dict]) -> str:
    """Return the newest date from an entry list as YYYY-MM-DD."""
    if not entries:
        return ""
    # Entries stored newest-first (rebuild_index sorts descending)
    return entries[0]["date"][:10]


def format_archive_list(index: dict) -> str:
    """Render the global archive listing (slug · memo count · newest date)."""
    if not index:
        return f"[dim]{escape(t('archive.empty'))}[/]"

    # Sort slugs by the date of each slug's newest memo, descending.
    ordered = sorted(
        index.items(),
        key=lambda kv: _newest_date(kv[1]),
        reverse=True,
    )

    slug_w = max(len("SLUG"), max(len(s) for s, _ in ordered))
    header = f"  {'SLUG':<{slug_w}}  {'MEMOS':>5}  {'NEWEST':<10}"
    lines = [
        f"\n[bold {_HEADER_COLOR}]═══ {t('archive.header')} ═══[/]",
        f"[bold]{header}[/]",
    ]
    for slug, entries in ordered:
        count = len(entries)
        newest = _newest_date(entries)
        lines.append(f"  {escape(slug):<{slug_w}}  {count:>5}  {newest:<10}")
    return "\n".join(lines)


# Fixed column widths chosen to leave room for a key_claim that can eat
# the remainder of the terminal. Sum of leading columns + padding =
# approx 30 chars, so key_claim gets ~terminal_width - 30.
_CLAIM_FIXED_BUDGET = 30
_DEFAULT_TERMINAL_WIDTH = 120


def _truncate(s: str, limit: int) -> str:
    """Truncate with ellipsis so the visible length is `limit` chars."""
    if limit <= 1:
        return "…" if s else ""
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def format_slug_memos(slug: str, entries: list[dict],
                      terminal_width: int = _DEFAULT_TERMINAL_WIDTH) -> str:
    """Render all memos for a given slug (list view before opening one)."""
    if not entries:
        msg = t("archive.not_found").replace("{slug}", slug)
        return f"[dim]{escape(msg)}[/]"

    claim_budget = max(20, terminal_width - _CLAIM_FIXED_BUDGET)

    lines = [
        f"\n[bold {_HEADER_COLOR}]═══ {t('archive.header')} — {escape(slug)} ═══[/]",
        f"[bold]  {'#':<3}{'DATE':<12}{'CONV':<8}KEY CLAIM[/]",
    ]
    for i, entry in enumerate(entries, start=1):
        date = entry["date"][:10]
        conv_level = (entry.get("conviction") or {}).get("level", "unknown") or "unknown"
        conv_label = conv_level.upper()
        conv_color = format_conviction_color(conv_level)
        claim = (entry.get("conviction") or {}).get("key_claim", "") or ""
        claim = _truncate(claim, claim_budget)
        if conv_color == "dim":
            conv_markup = f"[dim]{conv_label:<7}[/]"
        else:
            conv_markup = f"[{conv_color}]{conv_label:<7}[/]"
        lines.append(f"  {i:<3}{date:<12}{conv_markup} {escape(claim)}")
    return "\n".join(lines)


def format_memo_view(memo_data: dict) -> str:
    """Render a single memo's full content (analysis, conviction, etc)."""
    # Placeholder for Task 5
    pass
