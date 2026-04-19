"""Archive navigation views — pure rendering from the _index.json dict."""

from rich.markup import escape

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


def format_slug_memos(slug: str, entries: list[dict]) -> str:
    """Render all memos for a given slug (list view before opening one)."""
    # Placeholder for Task 4
    pass


def format_memo_view(memo_data: dict) -> str:
    """Render a single memo's full content (analysis, conviction, etc)."""
    # Placeholder for Task 5
    pass
