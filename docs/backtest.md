# Backtest / thesis-replay (`bt`)

Replays your book's realized fills against a benchmark since your first trade —
a factual "what did this actually do vs buy-and-hold" view, not a strategy
optimizer. The gap between the two curves is your alpha.

## Commands

```
backtest          replay vs the default benchmark (config.RS_BENCHMARK)
bt                same
bt SPY            replay vs a specific benchmark symbol
replay QQQ        same
```

## The fills ledger — why it's a local file

IBKR's execution API (`ibkr_trades`) only reaches back **~7 days**, so it
cannot reconstruct a multi-year book. The replay therefore reads fills from a
local CSV you seed once from your IBKR statements — those go back years.

Put it at `data/fills.csv` (gitignored — your trade history never leaves the
box). Format:

```csv
date,symbol,side,qty,price
2022-03-15,MSFT,BUY,100,250.50
2024-06-01,MSFT,SELL,40,420.00
2024-09-04,AAPL,BUY,50,220.00
```

- `date` ISO (`YYYY-MM-DD`), `side` is `BUY`/`SELL` (case-insensitive),
  `qty`/`price` positive numbers.
- Malformed rows are skipped with a warning, not fatal — hand-editing is fine.
- Prices for the equity curve come from yfinance (reaches back years); only the
  *fills* need seeding.

## Exporting fills from IBKR

The cleanest source is a **Flex Query** (Account Management → Reports → Flex
Queries → Trade Confirmations / Executions), which exports your full history as
CSV. Map its columns to `date,symbol,side,qty,price` and drop the result at
`data/fills.csv`. A one-time transform; after that the replay is live.

## How the equity curve is computed

`equity(day) = cost basis of all buys + realized gains + unrealized gains`,
using average-cost accounting so partial exits are exact. The deployed
principal stays "in the book" across a round-trip, so the curve reads as
account value, not just P&L. The benchmark is normalized to the same starting
capital, so both curves share a y-axis. Missing data never fabricates a value —
a feed gap carries the last-known price; an absent benchmark reads `N/A`, never
a fake `0%`.

## Demo

`ticker-tape --demo` seeds a deterministic sample ledger on the demo universe,
so `bt` shows the full shape with no network and no real data.
