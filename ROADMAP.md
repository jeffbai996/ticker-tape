# ticker-tape v-next roadmap

Drafted 2026-07-04 alongside the ticker-tape-web build. Items marked **[web-proven]**
were built and verified on the web version first — the mechanism is known-good,
this is a port, not an experiment.

## 1. Batch quote first paint — kill the cold-start trickle **[web-proven]**

The sidebar prices symbols through `data.fetch_quotes` + per-symbol `fast_info`
patching; cold start renders rows as fetches land. The web feed replaced this
with ONE `v7/finance/quote?symbols=A,B,C…` request (chunks of 40) that prices
the entire watchlist in a single round trip (~2s for 37 symbols), with the
slower per-symbol chart fetches running behind it for sparklines/technicals.

CLI port: a `bulk_quote_v7(symbols)` in data.py using a shared requests.Session
with Yahoo cookie+crumb (same dance the web worker does; yfinance's session can
be reused). Sidebar paints from the batch, then patches from the existing paths.
Expected win: cold start 2-3s → sub-second, and the 15s refresh becomes one
request instead of N.

Gotcha ported from web: re-derive change/pct from `price - regularMarketPreviousClose`
— the v7 change fields are garbage for ^TNX and friends.

## 2. Econ calendar 2027 — hard deadline

`econ_calendar.py` EVENTS is hardcoded 2026 and runs dry in December. Add 2027:
FOMC schedule is already published; CPI/NFP/GDP/PCE dates publish in Q4. Add a
"calendar exhausted" warning when today > last event − 30d so this never
silently goes stale again.

## 3. Sidebar layout customization

Web now has add/remove/reorder widgets on the dashboard rail (pulse, earnings,
calendar, movers, mini-charts), persisted per browser. CLI equivalent: a
`layout` command + JSON store listing sidebar sections in order — pulse,
earnings, P&L, risk, calendar-countdown — so the sidebar composition is
user-defined instead of fixed.

## 4. Macro calendar countdown in sidebar **[web-proven]**

The merged catalyst/econ calendar exists behind `cal`, but nothing surfaces
countdowns ambiently. Web added a compact right-rail panel (type-colored, next
6-8 events). CLI: optional sidebar section reusing `catalyst.get_merged_calendar`.

## 5. Cross-platform alert sync (design sketch)

Alerts armed in the CLI are invisible to the web and vice versa. Sketch: tiny
authed KV endpoint on the existing yf-proxy worker (`/alerts` + bearer token,
worker secret). Both clients sync on a slow poll; conflict rule = last-writer-
wins per alert id. Worth doing only if web alerts get real use — revisit after
a few weeks of web telemetry (i.e. do you actually arm alerts there?).

## 6. Web → CLI parity checks (small)

- **Earnings reaction stat**: web's `ei` computes close-to-close reaction after
  each print + beat-rate/streak summary. Verify CLI's earnings view shows the
  same aggregates; port if not.
- **Volume-histogram sparks**: web rows render 40-bar volume histograms
  (green/red by close direction) — arguably more informative than the intraday
  price sparkline for scan-reading. Consider a config toggle.

## 7. CLI → web parity (tracked in web repo, listed for completeness)

- Chat tool-calling (get_quotes / set_alert / add_catalyst from chat)
- Catalyst store port (user-maintained forward catalysts, localStorage)
- Memo/briefing archive (CLI has archive.py; web AI reports are one-shot)

## Non-goals for v-next

- IBKR features on web (public repo — never)
- Realtime websockets (Yahoo's stream is unofficial + flaky; 10-15s polling is
  fine for the use case)
