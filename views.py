"""Display functions for the terminal ticker."""

import os
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    NAMES, THESIS_BUCKETS,
    GREEN, RED, YELLOW, CYAN, WHITE, DIM, BOLD, RESET, MAGENTA,
)


# ── Formatters ──────────────────────────────────────────────

def _color_val(change: float, pct: float) -> str:
    color = GREEN if change >= 0 else RED
    arrow = "▲" if change >= 0 else "▼"
    return f"{color}{arrow} {change:+.2f} ({pct:+.2f}%){RESET}"


def _fmt_num(n) -> str:
    return f"{n:,.0f}" if n is not None else "—"


def _fmt_ratio(v) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _fmt_pct(v) -> str:
    return f"{v:.2f}%" if v is not None else "—"


def _fmt_cap(cap) -> str:
    if cap is None:
        return "—"
    if cap >= 1e12:
        return f"${cap / 1e12:.2f}T"
    if cap >= 1e9:
        return f"${cap / 1e9:.2f}B"
    if cap >= 1e6:
        return f"${cap / 1e6:.2f}M"
    return f"${cap:,.0f}"


def _row(label: str, value: str) -> str:
    return f"  {label:<20} {value:>16}"


def _section(title: str) -> None:
    print(f"\n  {BOLD}{CYAN}{title}{RESET}")
    print(f"  {DIM}{'─' * 36}{RESET}")


def _sparkline(prices: list[float], width: int = 40) -> str:
    """Render a Unicode sparkline from price data."""
    if not prices or len(prices) < 2:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(prices), max(prices)
    spread = hi - lo if hi != lo else 1

    # Resample to fit width
    if len(prices) > width:
        step = len(prices) / width
        sampled = [prices[int(i * step)] for i in range(width)]
    else:
        sampled = prices

    line = ""
    for p in sampled:
        idx = int(((p - lo) / spread) * (len(blocks) - 1))
        line += blocks[idx]

    # Color based on start vs end
    color = GREEN if sampled[-1] >= sampled[0] else RED
    return f"{color}{line}{RESET}"


def _rsi_color(rsi: float) -> str:
    """Color RSI based on overbought/oversold levels."""
    if rsi >= 70:
        return RED
    if rsi <= 30:
        return GREEN
    return WHITE


def _bar(pct: float, width: int = 20) -> str:
    """Horizontal bar chart segment for sector heatmap."""
    color = GREEN if pct >= 0 else RED
    filled = int(min(abs(pct) * 4, width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{RESET}"


# ── Ticker tape ─────────────────────────────────────────────

def scrolling_tape(quotes: list[dict], timestamp: str) -> None:
    """Horizontal scrolling ticker tape with extended hours data."""
    chars = []
    for q in quotes:
        if q.get("error") or q["price"] == 0.0:
            continue
        color = GREEN if q["change"] >= 0 else RED
        arrow = "▲" if q["change"] >= 0 else "▼"
        for c in f" {q['symbol']} ":
            chars.append((c, RESET + BOLD + YELLOW))
        for c in f"{q['price']:.2f} ":
            chars.append((c, RESET + WHITE))
        for c in f"{arrow}{q['pct']:+.2f}%":
            chars.append((c, RESET + color))

        # Show extended hours data inline
        if "ext_price" in q:
            ext_color = GREEN if q["ext_change"] >= 0 else RED
            ext_arrow = "▲" if q["ext_change"] >= 0 else "▼"
            for c in f" {q['ext_label']}:":
                chars.append((c, RESET + DIM + MAGENTA))
            for c in f"{q['ext_price']:.2f}":
                chars.append((c, RESET + MAGENTA))
            for c in f" {ext_arrow}{q['ext_pct']:+.2f}%":
                chars.append((c, RESET + ext_color))

        for c in "  │":
            chars.append((c, RESET + DIM))

    print(f"\n  {BOLD}{YELLOW}═══ TICKER TAPE ═══{RESET}")
    print(f"  {DIM}{timestamp} │ Press any key to continue{RESET}\n")

    if not chars:
        print(f"  {RED}No quote data available{RESET}\n")
        return

    # Try raw mode; fall back to static display
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError, AttributeError):
        _static_tape(quotes)
        return

    tty.setcbreak(fd)
    sys.stdout.write("\033[?25l")  # hide cursor

    # Temporarily capture Ctrl+C to stop tape instead of exiting
    stop = [False]
    prev_handler = signal.signal(signal.SIGINT, lambda s, f: stop.__setitem__(0, True))

    try:
        total = len(chars)
        width = os.get_terminal_size().columns - 1
        offset = 0
        while not stop[0]:
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
                break
            frame = ""
            last_ansi = ""
            for i in range(width):
                ch, ansi = chars[(offset + i) % total]
                if ansi != last_ansi:
                    frame += ansi
                    last_ansi = ansi
                frame += ch
            sys.stdout.write(f"\r{frame}{RESET}")
            sys.stdout.flush()
            offset = (offset + 1) % total
            time.sleep(0.1)
    finally:
        sys.stdout.write(f"\033[?25h\n")  # show cursor
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        signal.signal(signal.SIGINT, prev_handler)


def _static_tape(quotes: list[dict]) -> None:
    """Fallback static display when raw mode is unavailable."""
    for q in quotes:
        color = GREEN if q["change"] >= 0 else RED
        arrow = "▲" if q["change"] >= 0 else "▼"
        name = NAMES.get(q["symbol"], q["symbol"])
        ext = ""
        if "ext_price" in q:
            ext_c = GREEN if q["ext_change"] >= 0 else RED
            ext_a = "▲" if q["ext_change"] >= 0 else "▼"
            ext = f"  {MAGENTA}{q['ext_label']}:{q['ext_price']:.2f}{RESET} {ext_c}{ext_a}{q['ext_pct']:+.2f}%{RESET}"
        print(
            f"  {BOLD}{YELLOW}{q['symbol']:<5}{RESET} "
            f"{DIM}{name:<14}{RESET} "
            f"{WHITE}{q['price']:>8.2f}{RESET}  "
            f"{color}{arrow} {q['pct']:+.2f}%{RESET}"
            f"{ext}"
        )
    print()


# ── Stock lookup ────────────────────────────────────────────

def display_lookup(info: dict, symbol: str) -> None:
    """Medium-level stock metrics with extended hours."""
    price = info.get("regularMarketPrice", 0)
    prev = info.get("regularMarketPreviousClose", 0)
    change = price - prev if prev else 0
    pct = (change / prev * 100) if prev else 0
    name = info.get("shortName", symbol)

    # Last updated timestamp from market data
    mkt_time = info.get("regularMarketTime")
    if mkt_time:
        et = ZoneInfo("America/New_York")
        dt = datetime.fromtimestamp(mkt_time, tz=et)
        updated_str = f"  {DIM}Last updated: {dt.strftime('%b %d %H:%M ET')}{RESET}"
    else:
        updated_str = f"  {DIM}Last updated: unknown{RESET}"

    print(f"\n  {BOLD}{YELLOW}{symbol}{RESET}  {DIM}{name}{RESET}")
    print(f"  {BOLD}{WHITE}{price:.2f}{RESET}  {_color_val(change, pct)}")
    print(updated_str)

    # Extended hours
    pre = info.get("preMarketPrice")
    post = info.get("postMarketPrice")
    if pre:
        pre_chg = pre - prev if prev else 0
        pre_pct = (pre_chg / prev * 100) if prev else 0
        pre_c = GREEN if pre_chg >= 0 else RED
        print(f"  {MAGENTA}Pre-Market:{RESET} {WHITE}{pre:.2f}{RESET}  {pre_c}{pre_chg:+.2f} ({pre_pct:+.2f}%){RESET}")
    if post:
        post_chg = post - price
        post_pct = (post_chg / price * 100) if price else 0
        post_c = GREEN if post_chg >= 0 else RED
        print(f"  {MAGENTA}After Hours:{RESET} {WHITE}{post:.2f}{RESET}  {post_c}{post_chg:+.2f} ({post_pct:+.2f}%){RESET}")

    _section("Price & Volume")
    dl, dh = info.get("dayLow"), info.get("dayHigh")
    if dl and dh:
        print(_row("Day Range", f"{dl:.2f} — {dh:.2f}"))
    h52, l52 = info.get("fiftyTwoWeekHigh"), info.get("fiftyTwoWeekLow")
    if h52 and l52:
        print(_row("52w Range", f"{l52:.2f} — {h52:.2f}"))
        if h52 > 0:
            print(_row("Off 52w High", f"{((price - h52) / h52) * 100:.1f}%"))
    print(_row("Volume", _fmt_num(info.get("volume"))))
    print(_row("Avg Volume", _fmt_num(info.get("averageVolume"))))

    _section("Valuation")
    print(_row("Market Cap", _fmt_cap(info.get("marketCap"))))
    print(_row("P/E (TTM)", _fmt_ratio(info.get("trailingPE"))))
    print(_row("P/E (Fwd)", _fmt_ratio(info.get("forwardPE"))))
    print(_row("EV/EBITDA", _fmt_ratio(info.get("enterpriseToEbitda"))))

    _section("Earnings")
    print(_row("EPS (TTM)", _fmt_ratio(info.get("trailingEps"))))
    print(_row("EPS (Fwd)", _fmt_ratio(info.get("forwardEps"))))

    _section("Margins")
    for label, key in [("Gross", "grossMargins"), ("Net", "profitMargins")]:
        v = info.get(key)
        print(_row(f"{label} Margin", _fmt_pct(v * 100 if v is not None else None)))

    _section("Financials")
    print(_row("Revenue", _fmt_cap(info.get("totalRevenue"))))
    rg = info.get("revenueGrowth")
    print(_row("Revenue Growth", _fmt_pct(rg * 100 if rg is not None else None)))
    print(_row("Free Cash Flow", _fmt_cap(info.get("freeCashflow"))))
    print(_row("Debt/Equity", _fmt_ratio(info.get("debtToEquity"))))

    _section("Other")
    print(_row("Beta", _fmt_ratio(info.get("beta"))))
    dy = info.get("dividendYield")
    print(_row("Dividend Yield", _fmt_pct(dy * 100 if dy is not None else None)))
    target = info.get("targetMeanPrice")
    if target and price:
        upside = ((target - price) / price) * 100
        print(_row("Analyst Target", f"${target:.2f} ({upside:+.1f}%)"))
    print()


# ── Dashboards ──────────────────────────────────────────────

def display_thesis(quotes: list[dict]) -> None:
    """Holdings grouped by thesis bucket."""
    print(f"\n  {BOLD}{YELLOW}═══ THESIS DASHBOARD ═══{RESET}\n")
    quote_map = {q["symbol"]: q for q in quotes}

    for bucket, symbols in THESIS_BUCKETS.items():
        bucket_q = [quote_map[s] for s in symbols if s in quote_map]
        if not bucket_q:
            continue
        avg = sum(q["pct"] for q in bucket_q) / len(bucket_q)
        avg_c = GREEN if avg >= 0 else RED
        print(f"  {BOLD}{CYAN}{bucket}{RESET}  {avg_c}avg {avg:+.2f}%{RESET}")
        print(f"  {DIM}{'─' * 36}{RESET}")
        for q in bucket_q:
            name = NAMES.get(q["symbol"], q["symbol"])
            color = GREEN if q["change"] >= 0 else RED
            arrow = "▲" if q["change"] >= 0 else "▼"
            ext = ""
            if "ext_price" in q:
                ext_c = GREEN if q["ext_change"] >= 0 else RED
                ext = f"  {MAGENTA}{q['ext_label']}:{RESET}{ext_c}{q['ext_pct']:+.2f}%{RESET}"
            print(
                f"  {BOLD}{YELLOW}{q['symbol']:<5}{RESET} "
                f"{DIM}{name:<14}{RESET} "
                f"{WHITE}{q['price']:>8.2f}{RESET}  "
                f"{color}{arrow} {q['pct']:+.2f}%{RESET}"
                f"{ext}"
            )
        print()

    # Watchlist symbols not in any thesis bucket
    bucketed = {s for syms in THESIS_BUCKETS.values() for s in syms}
    extras = [q for q in quotes if q["symbol"] not in bucketed]
    if extras:
        print(f"  {BOLD}{CYAN}Watchlist{RESET}")
        print(f"  {DIM}{'─' * 36}{RESET}")
        for q in extras:
            color = GREEN if q["change"] >= 0 else RED
            arrow = "▲" if q["change"] >= 0 else "▼"
            print(
                f"  {BOLD}{YELLOW}{q['symbol']:<5}{RESET} "
                f"{WHITE}{q['price']:>8.2f}{RESET}  "
                f"{color}{arrow} {q['pct']:+.2f}%{RESET}"
            )
        print()


def display_earnings(earnings_data: list[dict]) -> None:
    """Upcoming earnings dates sorted by soonest."""
    print(f"\n  {BOLD}{YELLOW}═══ EARNINGS CALENDAR ═══{RESET}\n")
    for e in sorted(earnings_data, key=lambda x: x["days_until"] if x["days_until"] is not None else 9999):
        name = NAMES.get(e["symbol"], e["symbol"])
        d = e["days_until"]
        if d is not None and d <= 7:
            urg = RED + BOLD
        elif d is not None and d <= 30:
            urg = YELLOW
        else:
            urg = DIM
        days_str = f"{d}d" if d is not None else "?"
        print(
            f"  {BOLD}{YELLOW}{e['symbol']:<5}{RESET} "
            f"{DIM}{name:<14}{RESET} "
            f"{WHITE}{e['date']:>12}{RESET}  "
            f"{urg}{days_str:>5}{RESET}"
        )
    print()


def display_market(market_data: dict[str, list[dict]]) -> None:
    """Macro market overview with grouped categories."""
    print(f"\n  {BOLD}{YELLOW}═══ MARKET OVERVIEW ═══{RESET}")
    for group_name, items in market_data.items():
        print(f"\n  {BOLD}{CYAN}{group_name}{RESET}")
        print(f"  {DIM}{'─' * 44}{RESET}")
        for m in items:
            color = GREEN if m["change"] >= 0 else RED
            arrow = "▲" if m["change"] >= 0 else "▼"
            print(
                f"  {BOLD}{YELLOW}{m['symbol']:<12}{RESET} "
                f"{DIM}{m['name']:<14}{RESET} "
                f"{WHITE}{m['price']:>10.2f}{RESET}  "
                f"{color}{arrow} {m['pct']:+.2f}%{RESET}"
            )
    print()


# ── New features ────────────────────────────────────────────

def display_news(news_data: list[dict], symbol: str) -> None:
    """Recent news headlines."""
    print(f"\n  {BOLD}{YELLOW}═══ NEWS: {symbol} ═══{RESET}\n")
    if not news_data:
        print(f"  {DIM}No recent news found{RESET}\n")
        return
    for item in news_data:
        age = f" {DIM}({item['age']}){RESET}" if item["age"] else ""
        print(f"  {WHITE}{item['title']}{RESET}{age}")
        print(f"  {DIM}{item['publisher']}{RESET}")
        print()


def display_technicals(ta: dict, symbol: str) -> None:
    """Technical indicators dashboard."""
    print(f"\n  {BOLD}{YELLOW}═══ TECHNICALS: {symbol} ═══{RESET}\n")

    _section("Moving Averages")
    current = ta["current"]
    for label, key in [("SMA 20", "sma_20"), ("SMA 50", "sma_50"), ("SMA 200", "sma_200")]:
        val = ta[key]
        if val is None:
            print(_row(label, "—"))
            continue
        dist = ((current - val) / val) * 100
        color = GREEN if current > val else RED
        position = "above" if current > val else "below"
        print(_row(label, f"{val:.2f}  {color}{dist:+.1f}% {position}{RESET}"))

    _section("Momentum")
    rsi = ta["rsi"]
    if rsi is not None:
        rsi_c = _rsi_color(rsi)
        rsi_label = ""
        if rsi >= 70:
            rsi_label = " OVERBOUGHT"
        elif rsi <= 30:
            rsi_label = " OVERSOLD"
        print(_row("RSI (14)", f"{rsi_c}{rsi:.1f}{rsi_label}{RESET}"))
    else:
        print(_row("RSI (14)", "—"))

    _section("Volume")
    print(_row("Current Vol", _fmt_num(ta["current_vol"])))
    print(_row("Avg Vol (20d)", _fmt_num(ta["avg_vol_20"])))
    if ta["vol_ratio"] is not None:
        vr = ta["vol_ratio"]
        vr_color = YELLOW if vr > 1.5 else (RED if vr > 2.0 else DIM)
        print(_row("Vol Ratio", f"{vr_color}{vr:.2f}x{RESET}"))

    _section("Range")
    print(_row("52w High", f"{ta['high_52w']:.2f}  {RED}{ta['off_high']:.1f}%{RESET}"))
    print(_row("52w Low", f"{ta['low_52w']:.2f}  {GREEN}+{ta['off_low']:.1f}%{RESET}"))

    _section("Signals")
    for sig in ta["trend_signals"]:
        if "Golden" in sig or "Above" in sig:
            print(f"  {GREEN}● {sig}{RESET}")
        elif "Death" in sig or "Below" in sig:
            print(f"  {RED}● {sig}{RESET}")
        else:
            print(f"  {DIM}● {sig}{RESET}")
    print()


def display_chart(prices: list[float], symbol: str, period: str) -> None:
    """ASCII sparkline chart."""
    print(f"\n  {BOLD}{YELLOW}═══ CHART: {symbol} ({period}) ═══{RESET}\n")
    if not prices:
        print(f"  {DIM}No data available{RESET}\n")
        return

    try:
        width = min(os.get_terminal_size().columns - 8, 60)
    except OSError:
        width = 60
    spark = _sparkline(prices, width)
    print(f"  {spark}")

    lo, hi = min(prices), max(prices)
    first, last = prices[0], prices[-1]
    total_chg = ((last - first) / first) * 100 if first else 0
    color = GREEN if total_chg >= 0 else RED

    print(f"\n  {DIM}Low: {lo:.2f}  High: {hi:.2f}{RESET}")
    print(f"  {DIM}Start: {first:.2f}  End: {last:.2f}{RESET}  {color}{total_chg:+.2f}%{RESET}")
    print()


def display_sectors(sector_data: list[dict]) -> None:
    """Sector performance heatmap."""
    print(f"\n  {BOLD}{YELLOW}═══ SECTOR HEATMAP ═══{RESET}\n")
    if not sector_data:
        print(f"  {DIM}No data available{RESET}\n")
        return
    for s in sector_data:
        color = GREEN if s["pct"] >= 0 else RED
        arrow = "▲" if s["pct"] >= 0 else "▼"
        bar = _bar(s["pct"], 15)
        print(
            f"  {BOLD}{YELLOW}{s['symbol']:<5}{RESET} "
            f"{DIM}{s['name']:<14}{RESET} "
            f"{color}{arrow} {s['pct']:+.2f}%{RESET}  "
            f"{bar}"
        )
    print()


def auto_refresh_tape(fetch_fn, interval: int = 15) -> list[dict]:
    """Auto-refreshing static tape. Returns final quotes on exit."""
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (termios.error, ValueError, AttributeError):
        # No raw mode — do a single static refresh and return
        quotes, timestamp = fetch_fn()
        print(f"\n  {BOLD}{YELLOW}═══ TICKER TAPE ═══{RESET}")
        print(f"  {DIM}{timestamp}{RESET}\n")
        _static_tape(quotes)
        return quotes

    tty.setcbreak(fd)
    sys.stdout.write("\033[?25l")  # hide cursor

    stop = [False]
    prev_handler = signal.signal(signal.SIGINT, lambda s, f: stop.__setitem__(0, True))
    quotes = []

    try:
        while not stop[0]:
            # Fetch and render
            sys.stdout.write("\033[2J\033[H")  # clear screen, cursor to top
            print(f"  {BOLD}{YELLOW}═══ TICKER TAPE ═══{RESET}  {DIM}(auto-refresh {interval}s │ any key to stop){RESET}")
            print(f"  {DIM}Fetching...{RESET}", end="", flush=True)
            quotes, timestamp = fetch_fn()
            sys.stdout.write(f"\r\033[2K")
            print(f"  {DIM}{timestamp}{RESET}\n")
            _static_tape(quotes)

            # Wait for interval, checking for keypress every 0.1s
            elapsed = 0.0
            while elapsed < interval and not stop[0]:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.read(1)
                    stop[0] = True
                    break
                elapsed += 0.1
                remaining = max(0, int(interval - elapsed))
                sys.stdout.write(f"\r  {DIM}Next refresh in {remaining}s...{RESET}\033[K")
                sys.stdout.flush()
    finally:
        sys.stdout.write("\033[?25h")  # show cursor
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        signal.signal(signal.SIGINT, prev_handler)
        print(f"\n  {DIM}Auto-refresh stopped{RESET}\n")

    return quotes


def display_help() -> None:
    """Available commands."""
    print(f"\n  {BOLD}{YELLOW}═══ COMMANDS ═══{RESET}\n")
    cmds = [
        ("<ticker>", "Look up detailed stock metrics"),
        ("r, refresh", "Re-fetch and display ticker tape"),
        ("auto, a", "Auto-refresh tape every 15s"),
        ("thesis, t", "Thesis bucket dashboard"),
        ("earnings, er", "Upcoming earnings dates"),
        ("market, m", "Macro market overview"),
        ("news, n <SYM>", "Recent news headlines"),
        ("ta <SYM>", "Technical analysis (SMA/RSI)"),
        ("chart, c <SYM>", "ASCII sparkline price chart"),
        ("sectors, s", "Sector performance heatmap"),
        ("watch <SYM>", "Add symbol to watchlist"),
        ("unwatch <SYM>", "Remove from watchlist"),
        ("watchlist, wl", "Show watchlist"),
        ("help, h, ?", "Show this help"),
        ("q, quit, exit", "Exit"),
    ]
    for cmd, desc in cmds:
        print(f"  {BOLD}{CYAN}{cmd:<18}{RESET} {DIM}{desc}{RESET}")
    print()
