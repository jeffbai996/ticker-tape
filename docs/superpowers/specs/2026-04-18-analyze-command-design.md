# ticker-tape v2.6 — `analyze` command design

**Date:** 2026-04-18
**Status:** Approved, ready for implementation plan
**Scope:** v2.6 core feature. v2.7–v3.2 are footnotes, not in scope.

## Context

ticker-tape today is a decision tool + chat frontend + research lab. The gap in its decision-making role is **synthesis/judgment** (all raw data is there, but user assembles the narrative manually) and **staging** (decision happens in terminal, trade happens in IBKR web). The user explicitly does not want execution/trigger functionality — staging up to a copy-pasteable level is enough.

v2.6 addresses the synthesis gap with an on-demand deep-dive command. The terminal becomes an analyst with an archive.

## Feature: `analyze` command

### Invocation

Single command, Gemini Pro routes by first token:

```
analyze AVGO                  # symbol dive
analyze rotation              # thesis dive
analyze MU earnings           # symbol + angle hint
analyze semis correlation     # topic dive
analyze "why is XLU up"       # freeform
```

Classification:
- First token matches ticker in watchlist/positions → `kind=symbol`
- First token matches a thesis key in config → `kind=thesis`
- Neither → `kind=freeform`

Remaining tokens become an optional `angle_hint` passed through to the system prompt. `angle_hint` is free-form text (e.g., `"earnings"`, `"correlation"`, `"why is XLU up"`), stored verbatim in the memo's `angle` front-matter field. Default value when no hint is given: `"general"`.

### Output

1. **Memo streams to terminal** — rendered via existing `chat.py` markdown-to-Rich pipeline (no new renderer, no new duplication of that logic)
2. **Memo saved to disk** at `data/analyses/{slug}/{YYYY-MM-DD-HHMM}.md`
3. **`_index.json` updated** with new entry
4. **No staging tickets file** (user explicitly deprioritized)
5. **No follow-up chat drill-down** (user explicitly deprioritized — memo is the output, not the start of a conversation)

### Memo format

Markdown with YAML front-matter:

```markdown
---
target: AVGO
kind: symbol
angle: general
date: 2026-04-18T14:23:00-04:00
model: gemini-2.5-pro
prior_memos: [data/analyses/AVGO/2026-03-12-0915.md]
tools_used: [get_quote, get_technicals, get_news, grounding]
conviction: {level: high, key_claim: "AVGO custom-silicon moat widening"}
trigger_type: manual
---

# AVGO — 2026-04-18

## Context
{current price, position size, thesis bucket, since-last-memo summary}

## What Changed Since Last Memo
{diff against most recent prior memo — explicit "last time I argued X,
now I'm arguing Y because Z"}

## Current Read
{argued thesis status, not just tiles}

## Risks / Disconfirming Evidence
{explicit steelman of the bear case}

## Suggested Actions
{staged sizes/levels, or explicit "no action — hold"}

## Sources
{grounded citations, tool-call results referenced}
```

**Front-matter commitments (lock from day one, even if unused until later versions):**
- `conviction: {level, key_claim}` — read by v2.9 thesis dashboard
- `trigger_type: manual | alert | tripwire` — distinguishes v2.6 manual invocations from v2.8 auto-fired dives
- `prior_memos: [paths]` — enables v2.7 diff/history
- `tools_used: [...]` — enables retrospective audit

### Archive layout

```
data/analyses/
  AVGO/
    2026-04-18-1423.md
    2026-03-12-0915.md
  rotation/
    2026-04-18-1430.md
  _index.json
```

`_index.json` is a cache, regenerable from walking the directory. Structure:

```json
{
  "AVGO": [
    {"date": "2026-04-18T14:23:00-04:00", "path": "AVGO/2026-04-18-1423.md",
     "conviction": {"level": "high", "key_claim": "..."},
     "summary": "..."}
  ]
}
```

Rebuilt on every write (cheap at this scale, <10k memos expected over years).

### Architecture

New module: `analyze.py`

```python
def run_analysis(target: str, angle_hint: str = "") -> str:
    """Execute a deep-dive, stream memo to terminal, write to archive.

    Returns: path to the written memo file.
    """
```

Internal flow:

```
1. classify_target(target)           → (kind, slug, normalized_target)
2. archive.load_prior(slug)          → list of prior memo paths + front-matter
3. build_context()                   → positions, thesis snapshot, watchlist tech, live quotes
4. build_system_prompt(kind, context, prior_memos, angle_hint)
5. chat_module._BACKENDS["gemini"]   → streaming call with TICKER_TOOLS + grounding
6. render streamed content to RichLog (reuse existing markdown-to-Rich)
7. archive.write(slug, full_markdown, front_matter) → path
8. archive.rebuild_index()
9. return path
```

**Key constraint:** reuses `chat_module._BACKENDS["gemini"]` direct-access pattern (same pattern `brief ai` and memory compact already use). Does NOT touch chat history, does NOT change user's current chat model.

### Error handling

- **Gemini call fails** → error memo written to archive with failure reason + stack trace. User sees error in terminal. Memo file still exists so failure is auditable.
- **Archive write fails** (permissions, disk full) → memo still renders to terminal, red warning shown, error logged. Memo is not lost — it's still in the terminal buffer.
- **No prior memos for target** → memo generates normally; "What Changed Since Last Memo" section says "First memo for this target." explicitly.
- **Malformed front-matter in prior memo** → skip that prior memo with a dim warning; don't abort.
- **Target classifies as `freeform`** → memo's `target` field stores the raw user string; slug is a hash prefix so it lands in `data/analyses/_freeform/{hash}-{date}.md`.

### Testing

New file: `tests/test_analyze.py`

- `test_classify_target_symbol` — ticker in positions → kind=symbol
- `test_classify_target_thesis` — thesis key → kind=thesis
- `test_classify_target_freeform` — unknown → kind=freeform
- `test_archive_write_roundtrip` — write, then read back front-matter + body
- `test_archive_load_prior_ordered` — newest-first
- `test_archive_load_empty` — missing slug returns empty list
- `test_index_rebuild` — walks directory, produces correct `_index.json`
- `test_malformed_front_matter_skipped` — corrupt prior memo doesn't abort
- `test_analyze_routes_to_gemini` — with fake backend, verifies correct model + system prompt shape
- `test_analyze_writes_memo_on_success` — full happy path with fake backend
- `test_analyze_writes_error_memo_on_failure` — fake backend raises → error memo exists

Existing 604 tests must stay green at every commit.

## Hard constraints (do not break)

- No edits to `chat.py` internals beyond reading `_BACKENDS["gemini"]`
- No edits to `TICKER_TOOLS` — only call the tools
- No edits to existing `data/` subdirectories — only create new `data/analyses/`
- No edits to existing tests — only add new test files
- No schema changes to existing JSON stores, SQLite tables, memories, journal
- i18n: new keys only, no renames or deletions
- 604 existing tests must stay green at every commit on the feature branch

## Branch & PR workflow

- Branch: `feature/analyze-command` off current `main` (`2a3da3e`)
- Open draft PR early for incremental review
- Main stays untouched and deployable throughout
- Final merge: squash-merge from PR

**Commit plan on the branch:**

1. `feat: analyze module scaffold + classify_target + tests` — module skeleton, target classification, unit tests for classification only
2. `feat: analyze archive read/write + tests` — `data/analyses/` layout, `_index.json` logic, archive unit tests (roundtrip, ordering, empty, malformed)
3. `feat: analyze Gemini Pro integration + tests` — wire to `_BACKENDS["gemini"]`, streaming, fake-backend tests for routing/happy-path/failure
4. `feat: analyze command wired into app` — `analyze <target>` command handler in `app.py`
5. `docs: README section for analyze`

Tests land with the code that makes them pass — not all upfront. This keeps each commit green-on-its-own.

Each commit independently reversible. If step 3 breaks something, steps 1-2 stay valid.

## v2.7 → v3.2 (footnotes, not in scope)

These depend on v2.6 getting the front-matter schema and archive layout right. Listed only so v2.6 can design the schema correctly:

- **v2.7** — `history AVGO`, `diff AVGO`, `analyze AVGO since 2026-03-01` (needs `prior_memos` field)
- **v2.8** — alerts/trip-wires trigger auto-analyze (needs `trigger_type` field)
- **v2.9** — thesis dashboard shows argued conviction from latest memo (needs `conviction` field)
- **v3.0** — static-site web companion for read-only archive viewing
- **v3.1** — cross-symbol synthesis (`analyze semis`) reading multiple archived memos
- **v3.2** — new grounding sources: transcripts, 10-Qs, Fed minutes

## Success criteria

v2.6 ships when:
- `analyze AVGO` produces a doc-grade memo in terminal and writes to `data/analyses/AVGO/{date}.md` with full front-matter
- Re-running `analyze AVGO` references the prior memo explicitly in "What Changed"
- All 604 existing tests green + ~11 new tests green
- No regressions in any existing command, screen, or data flow
- PR diff reviewable in one sitting (target <800 lines net added, excluding tests)
- Feature branch merged to main via squash-merge from reviewed PR
