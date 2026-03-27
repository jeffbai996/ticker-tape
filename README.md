# ticker_tape — Interactive CLI Trading Terminal
*version 2.0.3*

Real-time quotes, thesis-driven portfolio views, technical analysis, and AI chat — all in a TUI that fits in a tmux pane.

<p>
  <img src="assets/screens.png" width="49%" />
  <img src="assets/chat-zh.png" width="49%" />
</p>

## Architecture

Built on Textual (Python TUI framework) with Rich markup rendering. Data layer uses yfinance with a TTL-cached info pipeline and parallel batch fetching via `ThreadPoolExecutor`. IBKR integration through MCP streamable HTTP with multi-account support.

**Refresh**: 30-second quote cycle, parallel sparkline fetches (6 workers), quotes display before charts load
**Cache**: 30-second TTL on `.info` calls — eliminates redundant yfinance requests across sidebar, thesis, and lookup views
**Compact mode**: Toggle with `c` — two-line-per-symbol view with technicals, auto-enables on narrow terminals
**Watchlist groups**: Named symbol groups with sidebar headers, synced to thesis buckets at runtime
**Alerts**: Price-level alerts with fire-once trigger and auto-removal
**i18n**: Full English/Chinese with ~230 translation keys, CJK display-width-aware column alignment via `unicodedata.east_asian_width()`

## Screens

| View | What it shows |
|------|--------------|
| **Thesis** | Two-line per symbol: price, change, ext hours, sparkline, RSI, 52-week range bar, earnings countdown, SMA signals, volume ratio, ATH%, relative strength. Portfolio header with breadth metrics and 10-indicator market context. |
| **Heatmap** | Color-coded performance grid sorted by daily change |
| **Technicals** | SMA 20/50/200, RSI, MACD with crossover detection, Bollinger Bands, ATR, relative strength vs benchmark |
| **Intraday** | 5-minute bars with VWAP overlay, multi-row tall sparkline charts |
| **Lookup** | Full stock profile: valuation (P/E, P/S, PEG, EV), margins, financials, ownership breakdown, analyst consensus |
| **Earnings** | Calendar with countdown, EPS estimates from yfinance calendar dict |
| **Economic** | FOMC, CPI, NFP, GDP, PCE dates with urgency coloring |
| **Insider** | Recent insider transactions with type/value/shares |
| **Options** | Options chain with IV, greeks, ATM/ITM/OTM tagging, moneyness filtering, expiration picker |
| **Correlation** | NxN correlation matrix across watchlist, color-coded by strength, avg pairwise metric |
| **Chat** | 7-model AI chat (Gemini Flash/Pro, Claude Haiku/Sonnet/Opus, GPT/GPT-mini) with terminal screen context awareness, DuckDuckGo web search, streaming responses, chain-of-thought display, persistent history, shared memory system |

## AI Chat

Seven models across three providers — switch mid-conversation with `model`. Gemini Flash-Lite and Flash for fast answers, Pro for depth. Claude Haiku, Sonnet, and Opus with extended thinking and DuckDuckGo web search. GPT and GPT-mini with reasoning.

The assistant sees everything you see. Live quotes, IBKR portfolio snapshots, and the last 50 lines of terminal output are injected into context automatically. Run `pos` or `acct`, switch to chat, and ask about what's on screen — no copy-pasting needed.

Conversation history persists across sessions and model switches. A shared memory system lets the assistant remember facts across conversations — portfolio context, prior analysis, user preferences. Token usage and per-model cost tracked in `history`.

## Status Bar

Static indices (S&P, Nasdaq, HSI, VIX, WTI, Brent, Gold, Silver, Natgas) + toggleable scrolling ticker tape with 18 symbols. Character-level scroll using Rich `Text` object slicing to preserve per-segment coloring. VIX and natgas color-coded by absolute level.

## IBKR Integration

Multi-account MCP client — positions, account summary, P&L, margin impact calculator (buy/sell), what-if order analysis, today's executions. Per-account labels with gateway-down detection. Compact column formatting with currency filtering. Parallel context fetch for AI chat system prompt.

## Stack

- `textual` — TUI framework, reactive properties, CSS styling
- `rich` — Markup rendering, Text objects for scrolling tape
- `yfinance` — Market data, technicals, earnings, insider transactions
- `google-genai` — Gemini chat with streaming and chain-of-thought
- `anthropic` — Claude chat with extended thinking and tool use
- `openai` — GPT chat with reasoning
- `ddgs` — DuckDuckGo search (Claude web tool)
- `mcp` — IBKR MCP client (streamable HTTP)
- `httpx` — Async HTTP transport
- `pytest` — 429 tests covering data layer, formatters, screens, MCP pipeline

## Demo

Multi-model AI chat with web search, chain-of-thought, and model switching mid-conversation.

<img src="assets/chat-multi-model.png" width="75%" />

Fully integrated Chinese language support with CJK-aware column alignment.

<p>
  <img src="assets/dashboard-en.png" width="49%" />
  <img src="assets/dashboard-zh.png" width="49%" />
</p>
