# ticker_tape — Interactive CLI Trading Terminal
*v2.5.1*

Real-time quotes, thesis-driven portfolio views, technical analysis, and AI chat — all in a TUI that fits in a tmux pane.

<p>
  <img src="assets/screens.png" width="49%" />
  <img src="assets/chat-zh.png" width="49%" />
</p>

## Architecture

Built on Textual (Python TUI framework) with Rich markup rendering. Data layer uses yfinance with a TTL-cached info pipeline and parallel batch fetching via `ThreadPoolExecutor`. IBKR integration through MCP streamable HTTP with multi-account support.

**Refresh**: 30-second quote cycle, parallel sparkline fetches (6 workers), quotes display before charts load
**Cache**: 30-second TTL on `.info` calls — eliminates redundant yfinance requests across sidebar, thesis, and lookup views
**Compact mode**: Toggle with `c` — two-line-per-symbol view with sparklines, earnings, technicals, and sidebar watchlist. Auto-enables on narrow terminals
**Watchlist groups**: Named symbol groups with sidebar headers, synced to thesis buckets at runtime
**Alerts**: Smart alerts — price levels, RSI thresholds, SMA crossovers, volume spikes, margin cushion. Fire-once trigger with auto-removal. Technical alerts on 60s eval cycle
**NLV History**: SQLite-backed NLV snapshots every 60s via peewee ORM (WAL mode). `timeline` shows 90-day ASCII chart with drawdown and leverage trend
**Morning Briefing**: `brief` assembles portfolio health, macro context (7 indicators), watchlist movers, upcoming earnings. `brief ai` adds AI synthesis
**Position Sizing**: `size SYM QTY` runs IBKR what-if with concentration and cushion analysis
**Earnings Tracker**: `surprises` shows watchlist-wide EPS beat/miss history with persistence to SQLite
**i18n**: Full English/Chinese with ~230 translation keys

## Screens

| View | What it shows |
|------|--------------|
| **Thesis** | Two-line per symbol: price, change, ext hours, sparkline, RSI, 52-week range bar, earnings countdown, SMA signals, volume ratio, ATH%, relative strength. Portfolio header with breadth metrics and 10-indicator market context (incl. gold). |
| **Heatmap** | Color-coded performance grid sorted by daily change |
| **Technicals** | SMA 20/50/200, RSI, MACD with crossover detection, Bollinger Bands, ATR, relative strength vs benchmark |
| **Intraday** | 5-minute bars with VWAP overlay, multi-row tall sparkline charts |
| **Lookup** | Full stock profile: valuation (P/E, P/S, PEG, EV), margins, financials, ownership breakdown, analyst consensus |
| **Earnings** | Calendar with countdown, EPS estimates from yfinance calendar dict |
| **Economic** | FOMC, CPI, NFP, GDP, PCE dates with urgency coloring |
| **Insider** | Recent insider transactions with type/value/shares |
| **Options** | Options chain with IV, greeks, ATM/ITM/OTM tagging, moneyness filtering, expiration picker |
| **Correlation** | NxN correlation matrix across watchlist, color-coded by strength, avg pairwise metric |
| **Comparison** | Side-by-side multi-symbol performance comparison |
| **Screening** | Quick multi-symbol comparison table for filtering ideas |
| **Timeline** | 90-day NLV ASCII chart with drawdown from peak, leverage trend, cushion |
| **Surprises** | Watchlist-wide earnings surprise history: EPS beat/miss, price reaction, streaks |
| **Sizing** | Pre-trade what-if: margin impact, concentration weight, cushion before/after |
| **Briefing** | Morning briefing: portfolio health, macro (7 indicators), movers, earnings this week |

## AI Chat

Seven models across three providers — switch mid-conversation with `model`.

| Model | Provider | Thinking | Context | Notes |
|-------|----------|----------|---------|-------|
| Gemini Flash | Google | — | 900K | Fast answers, cheapest |
| Gemini Pro | Google | 2,000 | 900K | Deep analysis |
| Haiku 4.5 | Anthropic | — | 200K | Fast summarization |
| Sonnet 4.6 | Anthropic | 4,000 | 180K | Balanced |
| Opus 4.6 | Anthropic | 8,000 | 180K | Strongest reasoning |
| GPT-5.4 mini | OpenAI | — | 120K | Fast GPT |
| GPT-5.4 | OpenAI | — | 120K | Full GPT |

```
ticker> model

═══ MODELS ═══
Type 'model' to list, 'model <name>' to switch.

  ◆ flash        Gemini Flash              gemini-3-flash-preview         ✓
    pro          Gemini Pro                gemini-3.1-pro-preview         ✓
    haiku        Haiku 4.5                 claude-haiku-4-5-20251001      ✓
    sonnet       Sonnet 4.6                claude-sonnet-4-6              ✓
    opus         Opus 4.6                  claude-opus-4-6                ✓
    gpt-mini     GPT-5.4 mini              gpt-5.4-mini                   ✓
    gpt          GPT-5.4                   gpt-5.4                        ✓
```

### Context System

The AI assistant has a layered context system — it knows who you are, what the market is doing, and what you've told it before.

**Live market context** — The system prompt is rebuilt on every chat session with current data: real-time quotes for all watched symbols (price, change, extended hours), technical signals (RSI, SMA position, distance from 52-week high), and portfolio thesis bucket groupings. If IBKR is connected, condensed account summaries (NLV, leverage, margin cushion, top holdings) from all configured accounts are fetched in parallel and injected. The AI can answer "what's my cushion?" or "which positions are above their 50-day?" without running any commands first. Anthropic calls use prompt caching on the system block — the static persona and context are written once and read at 90% discount on subsequent turns.

**Slash commands** — Feed specific data to the AI mid-conversation: `/ta AAPL` runs technicals and injects the output, `/pos` pulls IBKR positions, `/acct` gets account summary. The AI analyzes the data in context.

**Web search** — When the AI needs current information beyond what's in context, it searches the web automatically via Tavily (optimized for LLM consumption — returns clean extracted text, not raw HTML) with DuckDuckGo as a free fallback. No manual trigger required.

### Image Input

Paste images into chat with `Ctrl+P` (macOS clipboard) or drop file paths directly into the input. Works across all seven models — Anthropic, Gemini, and OpenAI all receive the image as base64 content blocks in their native format. Use it for chart analysis, screenshot questions, or anything visual.

### Memory

Memories are persistent facts that survive across sessions, model switches, and history compaction. Stored as JSON on disk and injected into every model's system prompt, so all seven models share the same knowledge base.

**Three ways to save:** `memory add <text>` from the command bar, `remember <text>` while in chat mode (direct, no API call), or just tell the AI conversationally — "remember that AAPL reports Jan 30" — and it saves automatically.

**Deleting works the same way** — `memory delete 5` from the command bar, or tell the AI "forget memory 5" in conversation.

**Compaction** — `memory compact` sends all stored memories through Gemini Flash, which distills them into a smaller set of consolidated facts. Useful when you've built up 20+ memories and want to keep the signal without the noise.

```
ticker> memory

═══ MEMORIES ═══

  #1  AAPL reports Jan 30, after hours                   2026-03-28
  #2  Sold 200 MSFT at $420 for rebalancing              2026-03-26
  #3  Earnings week — no new positions until Friday      2026-03-24
  #4  TSLA support at $180, resistance at $210           2026-03-22

memory <ID> (full text) · add <text> · delete <ID> · compact
```

### History

Conversation history persists to disk across sessions and model switches. `history` shows a paginated view of past exchanges with per-model cost breakdown.

```
ticker> history

═══ HISTORY (12 turns, page 1/2) ═══
  Opus $0.03  Sonnet $0.01  Haiku $0.00  5.4 $0.00  5.4m $0.00  Gemini $0.02
  ~8.2K tokens  ~$0.06

  1  Q: what do you think about AMZN's setup going into earnings?
     sonnet ▸ AMZN looks solid — RSI 52 is neutral, sitting right on the 50d...

  2  Q: compare it to GOOGL
     sonnet ▸ Both are mega-cap tech but with different growth drivers...

history 2 · peek N · search <term> · delete N[-M] · compact · clear
```

`history search <term>` finds old conversations. `history delete 3` removes a single exchange. `history delete 6-10` removes a range.

**Compaction** — `history compact` sends the full history through Gemini Flash, which produces a concise summary. The original messages are replaced with a single summary entry. The AI retains the knowledge without burning tokens re-reading 50 old exchanges.

### Chain-of-Thought, Search & Token Usage

Models with thinking budgets (Gemini Pro, Claude Sonnet, Claude Opus) stream their internal reasoning before responding — buffered by paragraph for readability. Toggle visibility with `Ctrl+O`. When hidden, thinking still runs and improves the answer.

**Native web search** fires automatically when a model needs current information. Each provider uses its own search: Claude uses `web_search`, Gemini uses Google Search with grounding metadata, GPT uses the Responses API `web_search`. Search indicators appear inline during thinking. Citation markers are stripped from final output.

**Token usage** displays input/output token counts, elapsed time, and per-model cost breakdown. Toggle with `Ctrl+T`. Anthropic calls use prompt caching on the system block — the static persona/context block is written once and read at 90% discount on subsequent turns. Cache hits are tracked per call.

A random thinking placeholder (from a pool of ~80 words including "Catfishing", "Moonwalking", "Glazing") shows while waiting for the first token.

```
Q: what's happening with the TSMC arizona fab timeline?
Moonwalking...
Analyzing the Request
    The user is asking about TSMC's Arizona fab construction
    timeline. I should search for the latest updates on the
    N4P process node ramp and any delays or policy changes.
Searching the Web
    🔍 Searching: "TSMC Arizona fab timeline 2026"
    🔍 Searching: "TSMC N4P production ramp update"
Synthesizing
    The latest reports confirm first silicon on the N4P node
    ahead of schedule. This is significant for the domestic
    semiconductor supply chain buildout...

A:
TSMC's Arizona Fab 1 hit first silicon on the N4P node two weeks
ahead of the revised schedule. Yield rates are tracking at 92%,
which is competitive with Tainan. Fab 2 (N3E) broke ground last
month with a 2028 target. The CHIPS Act second tranche disbursed
$2.1B in March, removing the funding overhang.

┌───────────────────────────┐
│ ↑ 3,156t  ↓ 809t  ⏱ 10.4s │
└───────────────────────────┘
```

### Journal & Resume

`journal save [N]` captures the last N chat exchanges losslessly to a persistent journal — useful for archiving analysis or trade rationale. `journal` lists saved entries with timestamps.

`resume` replays the tail of chat history (~3000 chars) with full markdown rendering, so you can pick up where you left off after restarting.

### Auto News Context

Mentioning a watchlist ticker in chat automatically fetches its recent headlines and injects them into the AI's context. No manual `/news` command needed — just say "what do you think about AAPL" and the model sees the latest headlines alongside its market data.

### AI Tool Use

The AI can call ticker-tape functions directly when you ask about specific data — no slash commands needed. Ask "what's MSFT's RSI?" and the model calls `get_technicals(MSFT)` to fetch real-time indicators, then analyzes the result. Ask "show my positions" and it calls `ibkr_get_positions()` to pull live portfolio data.

| Tool | What It Fetches |
|------|----------------|
| `get_technicals` | RSI, SMA, MACD, Bollinger, ATR, volume ratio |
| `get_news` | Recent headlines with age |
| `get_fundamentals` | P/E, margins, growth, market cap |
| `get_chart` | Price history (configurable period/interval) |
| `get_intraday` | Today's 5-min bars with VWAP |
| `ibkr_get_positions` | Portfolio positions with P&L |
| `ibkr_get_account_summary` | Account NLV, margin, buying power |
| `ibkr_briefing` | Risk dashboard with margin metrics |
| `ibkr_stress_test` | Stress test under configurable scenarios |
| `ibkr_what_if` | Simulate trade: margin impact, cushion change |
| `get_earnings_surprises` | Historical EPS beat/miss across watchlist |
| `get_briefing` | Morning briefing: portfolio, macro, movers, earnings |

Tools are defined once in a provider-agnostic registry and translated to each provider's native format (Anthropic tool_use, Gemini function_declarations, OpenAI function tools). Execution is local — the model requests a tool, ticker-tape runs it in Python, feeds the result back, and the model continues with analysis. Works across all seven models.

### Clipboard

`copy` captures the current screen content to your system clipboard (macOS `pbcopy`, Linux `xclip`, WSL `clip.exe`). `copy 20` for just the last 20 lines. Falls back to file export if no clipboard tool is available.

### Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| `Ctrl+O` | Chat | Toggle chain-of-thought display |
| `Ctrl+T` | Chat | Toggle token usage box |
| `Ctrl+P` | Chat | Paste clipboard image (macOS) |
| `Ctrl+N` | Chat | New line in input |
| `c` | Ticker | Toggle compact mode |
| `l` | Ticker | Switch language (en/zh) |

## Status Bar

Scrolling top bar with global indices (S&P, Nasdaq, HSI, VIX, WTI, Brent, Gold, Silver, Natgas) and local ET clock. Character-level scroll using Rich `Text` object slicing to preserve per-segment coloring. Off-hours swaps to futures tickers (ES=F, NQ=F). VIX and natgas color-coded by absolute level (green/yellow/red thresholds). NYSE holiday detection with orange "HOLIDAY" tag.

## Sidebar

Real-time watchlist quotes with refresh flash indicator. IBKR P&L section (daily/unrealized/realized aggregated across accounts, 30s refresh). Risk section (cushion, leverage, NLV from primary account, 60s refresh) with color-coded thresholds. Earnings calendar in the Pulse section.

## IBKR Integration

Multi-account MCP client over streamable HTTP. Two accounts on the same or separate IB Gateways, configured via environment variables. Per-account labels with gateway-down detection. Compact column formatting with automatic currency filtering (USD/CAD).

| Command | What it shows |
|---------|--------------|
| `/pos` | Positions table with cost basis, P&L, weights, daily P&L summary |
| `/acct` | Account health: NLV, margin, buying power, cushion, leverage |
| `/pnl` | Daily/unrealized/realized P&L with daily return % |
| `/risk` | Full risk dashboard: health, concentration, alerts, VaR |
| `/trades` | Today's executions grouped by symbol, VWAP, commissions |
| `/orders` | Open/pending orders |
| `/detail SYM` | Single position deep dive: cost, weight, P&L, margin, performance |
| `/margin` | Margin summary, `/margin SYM` headroom, `/margin SYM QTY` what-if |
| `/stress` | Stress test: preflight, drawdown curve, overnight gap risk |
| `/beta` | Portfolio beta vs SPY with per-position breakdown |
| `/corr` | Pairwise correlation matrix |
| `/sector` | Sector exposure breakdown with HHI |
| `/ibkr` | Consolidated cross-account view with FX conversion |

```
═══ POSITIONS (U12345678) ═══  14:32:05 ET

         Shares      Cost     Price         Value            P&L    Wt%
─────────────────────────────────────────────────────────────────────
 AAPL       500 $  185.20 $  198.45   $99,225.00   +$6,625.00   28.3%
 MSFT       300 $  378.50 $  412.30  $123,690.00  +$10,140.00   25.1%
 GOOGL      400 $  155.80 $  168.92   $67,568.00   +$5,248.00   19.4%
 AMZN       250 $  178.40 $  192.15   $48,037.50   +$3,437.50   13.7%

 Total Market Value             $350,482.50 USD
 Daily P&L                      +$2,841.20
 Daily P&L %                    +0.82%
```

## Stack

- `textual` — TUI framework, reactive properties, CSS styling
- `rich` — Markup rendering, Text objects for scrolling tape
- `yfinance` — Market data, technicals, earnings, insider transactions
- `google-genai` — Gemini chat with streaming, chain-of-thought, Google Search, code execution
- `anthropic` — Claude chat with extended thinking, web search, code execution, prompt caching
- `openai` — GPT chat with Responses API web search
- `tavily` — LLM-optimized web search fallback (DDG as secondary fallback)
- `mcp` — IBKR MCP client (streamable HTTP, multi-account)
- `httpx` — Async HTTP transport
- `peewee` — SQLite ORM for NLV history and earnings persistence (WAL mode)
- `pytest` — 584 tests covering data layer, formatters, screens, chat, tool registry, journal, memory tags, MCP pipeline, smart alerts, db persistence

## Demo

Multi-model AI chat with web search, chain-of-thought, and model switching mid-conversation.

<img src="assets/chat-multi-model.png" width="75%" />

Fully integrated Chinese language support with CJK-aware column alignment.

<p>
  <img src="assets/dashboard-en.png" width="49%" />
  <img src="assets/dashboard-zh.png" width="49%" />
</p>
