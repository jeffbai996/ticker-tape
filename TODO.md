# TODO

## Demo polish (paused 2026-06-24, token-heavy)

`app.py --demo` is shippable. Done so far: window sized (~912w, thesis dashboard
fits without wrapping), $50K / no-leverage demo account, diversified positions +
watchlist (Tech / Financials / Healthcare / Consumer & Energy / Ballast — GOOGL
removed) so the demo shows a generic balanced book not modeled on any real
portfolio, command-bar input border fix (tall → round), pre-market PM/PRE
indicator, AI chat + memory-save demo (shot in README's Memory section), and the
bare-ticker → info-page fix.

Remaining:

- [ ] Capture more feature screens for the README / demo reel: price chart,
      heatmap (`hm`), technicals (`ta SYM`), intraday (`i SYM`)
- [ ] Optional: GIF of the tape animating (demo prices drift per minute)
- [ ] Decide whether to pause launch on the hero banner — it scrolls off in demo
      because the data is instant; live mode shows it during the network fetch
- [ ] Assemble the final demo reel / material

Intermediate captures live in `~/Desktop/ticker tape demo tmp/`.
Launch: `./venv/bin/python app.py --demo` (from the repo root)
