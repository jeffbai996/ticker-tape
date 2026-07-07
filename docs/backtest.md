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
bt QQQ ccy CAD    report in CAD (default: config.BACKTEST_CCY)
bt ccy USD        report currency without changing the benchmark
```

## Currencies — mixed CAD/USD books

A book that mixes USD tickers and Canadian listings (CDRs on NEO, TSX ETFs)
is normalized to ONE report currency before the curve is built — summing CAD
and USD in one equity number would be silently wrong. Each fill and each
daily close converts at **its own date's** USDCAD rate (from yfinance), so FX
moves are part of the return; a missing FX day carries the last-known rate.
Ledger rows carry an optional `currency` column (defaults to `USD`):

```csv
date,symbol,side,qty,price,currency
2022-03-15,MSFT,BUY,100,250.50,USD
2023-01-10,AAPL.NE,BUY,500,25.10,CAD
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
CSV. The importer does the column mapping for you:

```
python flex_import.py my_flex_export.csv            # writes data/fills.csv
python flex_import.py my_flex_export.csv --force    # overwrite an existing ledger
```

It maps the usual Flex column aliases (`TradeDate`/`Date/Time`,
`TradePrice`/`Price`, `CurrencyPrimary`, ...), infers side from signed
quantity when there's no Buy/Sell column, skips section headers and subtotal
rows, and suffixes CAD listings to their yfinance tickers by listing exchange
(NEO → `.NE`, TSX → `.TO`). A one-time transform; after that the replay is
live.

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
