"""Analyze module — on-demand deep-dive orchestration.

Reads user target, classifies as symbol/thesis/freeform, loads prior memos
from archive, builds context, calls Gemini Pro, streams memo to terminal,
writes result back to archive.
"""

import datetime
import re


def _watchlist_symbols() -> list[str]:
    """Return current watchlist symbols. Isolated for easy monkeypatching."""
    import data
    return data.get_all_symbols()


def _thesis_keys() -> list[str]:
    """Return configured thesis bucket keys (lowercased)."""
    import config
    return [k.lower() for k in (config.THESIS_BUCKETS or {}).keys()]


_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")

# Prior memos fed into the system prompt are capped — token budget + older
# memos lose relevance. 5 covers "recent conviction evolution" without bloat.
MAX_PRIOR_MEMOS_IN_PROMPT = 5

def _gemini_pro_model_id() -> str:
    """Fetch Gemini Pro's model id from chat.py's registry so memo stamps
    stay truthful when the model version bumps. Lazy import avoids a hard
    coupling at module load."""
    import chat
    return chat.MODELS["pro"]["id"]


def classify_target(target: str) -> tuple[str, str]:
    """Classify a raw target string.

    Returns (kind, normalized_target) where kind is "symbol", "thesis",
    or "freeform".

    Rules (in order):
    1. If target is in watchlist (case-insensitive) → symbol
    2. If target is in thesis keys (case-insensitive) → thesis
    3. If target is 1-5 letters with no spaces → symbol (even if unknown)
    4. Otherwise → freeform (normalized unchanged)
    """
    stripped = target.strip()
    upper = stripped.upper()
    lower = stripped.lower()

    if upper in _watchlist_symbols():
        return ("symbol", upper)
    if lower in _thesis_keys():
        return ("thesis", lower)
    if _TICKER_RE.match(stripped):
        return ("symbol", upper)
    return ("freeform", stripped)


def build_system_prompt(kind: str, target: str, angle_hint: str,
                        prior_memos: list[dict]) -> str:
    """Build the system prompt for a deep-dive analysis call."""
    # Prior memos summary
    if prior_memos:
        prior_block = "Prior memos for this target (newest first):\n"
        for m in prior_memos[:MAX_PRIOR_MEMOS_IN_PROMPT]:
            fm = m.get("front_matter", {})
            conv = fm.get("conviction", {})
            # Escape quotes and flatten newlines so one claim can't
            # break the list structure.
            claim = str(conv.get("key_claim", "")).replace('"', '\\"').replace("\n", " ")
            prior_block += (
                f"- {fm.get('date', '?')}: "
                f"conviction={conv.get('level', '?')} "
                f"claim=\"{claim}\"\n"
            )
    else:
        prior_block = "No prior memos for this target — this is the first memo."

    kind_guidance = {
        "symbol": f"Target is a single symbol: {target}. "
                  "Argue the thesis status for this specific position.",
        "thesis": f"Target is a thesis: {target}. "
                  "Argue the state of this thesis across the affected positions.",
        "freeform": f"Target is a freeform question: \"{target}\". "
                    "Answer directly with a doc-grade memo.",
    }[kind]

    angle_line = (f"Angle hint: {angle_hint}. Focus the memo accordingly."
                  if angle_hint and angle_hint != "general"
                  else "No specific angle — give a general deep-dive.")

    return f"""You are a senior equity analyst producing a doc-grade memo.

{kind_guidance}

{angle_line}

{prior_block}

Use your tools (quotes, technicals, thesis, positions, news) and Google search grounding to gather current data. Cite sources explicitly in the Sources section.

Produce the memo with EXACTLY these sections in this order:

## Context
Current price, position (if held), thesis bucket, since-last-memo one-line summary.

## What Changed Since Last Memo
If there is a prior memo: "Last time I argued X. Now I'm arguing Y because Z." Be explicit.
If no prior memo: state "First memo for this target."

## Current Read
Argue the current thesis status. Be direct and specific. No hedging.

## Risks / Disconfirming Evidence
Steelman the bear case. List concrete disconfirming evidence.

## Suggested Actions
Specific sizing/levels, or explicit "no action — hold." Include stop levels where relevant.

## Sources
Citations from grounding and tool calls.

Output ONLY the memo markdown starting with `# {target} — YYYY-MM-DD`. No preamble, no meta-commentary.
"""


def build_front_matter(kind: str, target: str, angle: str,
                       prior_memos: list[dict], tools_used: list[str],
                       conviction: dict, trigger_type: str = "manual") -> dict:
    """Build the YAML front-matter dict for a new memo."""
    now = datetime.datetime.now().astimezone()
    iso = now.isoformat(timespec="seconds")
    return {
        "target": target,
        "kind": kind,
        "angle": angle,
        "date": iso,
        "model": _gemini_pro_model_id(),
        "prior_memos": [m["path"] for m in prior_memos],
        "tools_used": tools_used,
        "conviction": conviction,
        "trigger_type": trigger_type,
    }
