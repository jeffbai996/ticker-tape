"""Gemini chatbot with portfolio context."""

import logging

from config import GEMINI_API_KEY, GEMINI_MODEL, THESIS_BUCKETS, SYMBOLS, CYAN, DIM, BOLD, RESET, RED, YELLOW

log = logging.getLogger(__name__)

_client = None
_client_checked = False


def _get_client():
    """Lazy singleton for Gemini client. Returns None if no API key."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:
        log.warning("Failed to create Gemini client: %s", e)
        return None


def _build_system_prompt(quotes: list[dict],
                         ibkr_snapshot: str | None = None,
                         ibkr_pnl: str | None = None) -> str:
    """System prompt with thesis framework + live context."""
    parts = [
        "You are a trading assistant for an independent trader running a concentrated "
        "semiconductor portfolio (Silicon Railroads thesis: AI infrastructure buildout "
        "through 2026, value shifts to applications layer after).",
        "",
        "Portfolio symbols: " + ", ".join(SYMBOLS),
        "Thesis buckets: " + ", ".join(
            f"{k}: {', '.join(v)}" for k, v in THESIS_BUCKETS.items()
        ),
        "",
        "Be direct, concise, and quantitative. Lead with insight. "
        "No hedging or disclaimers. The user is an experienced trader.",
    ]

    # Live quotes context
    if quotes:
        parts.append("")
        parts.append("Current quotes:")
        for q in quotes:
            if q.get("price"):
                parts.append(f"  {q['symbol']}: ${q['price']:.2f} ({q['pct']:+.2f}%)")

    # IBKR context if available
    if ibkr_snapshot:
        parts.append("")
        parts.append("IBKR Portfolio Snapshot:")
        parts.append(ibkr_snapshot)
    if ibkr_pnl:
        parts.append("")
        parts.append("IBKR P&L:")
        parts.append(ibkr_pnl)

    return "\n".join(parts)


def chat_single_turn(question: str, quotes: list[dict]) -> str | None:
    """One-shot question with portfolio context."""
    client = _get_client()
    if client is None:
        return None

    # Fetch IBKR context (best-effort)
    from data import fetch_ibkr_snapshot, fetch_ibkr_pnl
    ibkr_snapshot = fetch_ibkr_snapshot()
    ibkr_pnl = fetch_ibkr_pnl()

    system_prompt = _build_system_prompt(quotes, ibkr_snapshot, ibkr_pnl)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        )
        return response.text
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return f"Error: {e}"


def chat_multi_turn(quotes: list[dict]) -> None:
    """Interactive chat loop with conversation history."""
    client = _get_client()
    if client is None:
        print(f"\n  {RED}Set GEMINI_API_KEY in .env to use chat{RESET}\n")
        return

    # Pre-fetch IBKR context once for the session
    from data import fetch_ibkr_snapshot, fetch_ibkr_pnl
    print(f"\n  {DIM}Loading portfolio context...{RESET}", end="", flush=True)
    ibkr_snapshot = fetch_ibkr_snapshot()
    ibkr_pnl = fetch_ibkr_pnl()
    print(f"\r\033[2K", end="")

    system_prompt = _build_system_prompt(quotes, ibkr_snapshot, ibkr_pnl)

    print(f"\n  {BOLD}{YELLOW}═══ CHAT MODE ═══{RESET}")
    print(f"  {DIM}Type 'q' or 'exit' to return to ticker{RESET}\n")

    history = []

    while True:
        try:
            user_input = input(f"  {CYAN}chat>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

        history.append({"role": "user", "parts": [{"text": user_input}]})

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                },
            )
            reply = response.text
            history.append({"role": "model", "parts": [{"text": reply}]})
            print(f"\n  {BOLD}{CYAN}AI:{RESET} {reply}\n")
        except Exception as e:
            print(f"\n  {RED}Error: {e}{RESET}\n")

    print(f"\n  {DIM}Exited chat mode{RESET}\n")
