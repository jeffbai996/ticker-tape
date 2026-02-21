#!/usr/bin/env python3
"""Terminal stock ticker tape with interactive stock lookup."""

import sys
import signal

from config import CYAN, DIM, BOLD, RESET, GREEN, RED, YELLOW
from config import NAMES
from data import (
    get_all_symbols, fetch_quotes, fetch_stock_info, fetch_earnings,
    fetch_market_overview, fetch_news, fetch_technicals, fetch_chart_data,
    fetch_sector_performance, load_watchlist, add_to_watchlist, remove_from_watchlist,
)
from views import (
    scrolling_tape, display_lookup, display_thesis, display_earnings,
    display_market, display_news, display_technicals, display_chart,
    display_sectors, display_help, auto_refresh_tape,
)


def cleanup(sig=None, frame=None) -> None:
    print(f"\n  {DIM}Bye.{RESET}")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def show_tape() -> list[dict]:
    """Fetch quotes and run the scrolling tape. Returns quotes for reuse."""
    print(f"\n  {DIM}Fetching quotes...{RESET}", end="", flush=True)
    quotes, timestamp = fetch_quotes(get_all_symbols())
    print(f"\r\033[2K", end="")
    scrolling_tape(quotes, timestamp)
    return quotes


def handle_command(cmd: str, quotes: list[dict]) -> list[dict] | None:
    """Process a command. Returns updated quotes if refreshed, else None."""
    parts = cmd.split(maxsplit=1)
    action = parts[0]
    arg = parts[1].strip().upper() if len(parts) > 1 else ""

    if action in ("A", "AUTO"):
        return auto_refresh_tape(lambda: fetch_quotes(get_all_symbols()))

    if action in ("R", "REFRESH"):
        return show_tape()

    if action in ("T", "THESIS"):
        display_thesis(quotes)
        return None

    if action in ("ER", "EARNINGS"):
        print(f"\n  {DIM}Fetching earnings dates...{RESET}")
        display_earnings(fetch_earnings(get_all_symbols()))
        return None

    if action in ("M", "MARKET"):
        print(f"\n  {DIM}Fetching market data...{RESET}")
        display_market(fetch_market_overview())
        return None

    if action in ("N", "NEWS"):
        if not arg:
            print(f"\n  {DIM}Usage: news <SYMBOL>{RESET}\n")
        else:
            print(f"\n  {DIM}Fetching news for {arg}...{RESET}")
            display_news(fetch_news(arg), arg)
        return None

    if action == "TA":
        if not arg:
            print(f"\n  {DIM}Usage: ta <SYMBOL>{RESET}\n")
        else:
            print(f"\n  {DIM}Calculating technicals for {arg}...{RESET}")
            ta = fetch_technicals(arg)
            if ta:
                display_technicals(ta, arg)
            else:
                print(f"  {RED}No data for '{arg}'{RESET}\n")
        return None

    if action in ("C", "CHART"):
        if not arg:
            print(f"\n  {DIM}Usage: chart <SYMBOL> [period]{RESET}")
            print(f"  {DIM}Periods: 1w, 1mo, 3mo, 6mo, 1y, ytd{RESET}\n")
        else:
            # Allow optional period: "chart NVDA 3mo"
            chart_parts = arg.split()
            sym = chart_parts[0]
            period = chart_parts[1].lower() if len(chart_parts) > 1 else "1mo"
            valid_periods = ("1w", "1mo", "3mo", "6mo", "1y", "2y", "ytd")
            if period not in valid_periods:
                period = "1mo"
            print(f"\n  {DIM}Fetching chart data for {sym}...{RESET}")
            prices = fetch_chart_data(sym, period=period)
            if prices:
                display_chart(prices, sym, period)
            else:
                print(f"  {RED}No chart data for '{sym}'{RESET}\n")
        return None

    if action in ("S", "SECTORS"):
        print(f"\n  {DIM}Fetching sector data...{RESET}")
        display_sectors(fetch_sector_performance())
        return None

    if action == "WATCH":
        if not arg:
            print(f"\n  {DIM}Usage: watch <SYMBOL>{RESET}\n")
        elif add_to_watchlist(arg):
            print(f"\n  {GREEN}Added {arg} to watchlist — refreshing tape...{RESET}")
            return show_tape()
        else:
            print(f"\n  {DIM}{arg} already tracked{RESET}\n")
        return None

    if action == "UNWATCH":
        if not arg:
            print(f"\n  {DIM}Usage: unwatch <SYMBOL>{RESET}\n")
        elif remove_from_watchlist(arg):
            print(f"\n  {GREEN}Removed {arg} from watchlist — refreshing tape...{RESET}")
            return show_tape()
        else:
            print(f"\n  {RED}{arg} not on watchlist{RESET}\n")
        return None

    if action in ("WL", "WATCHLIST"):
        wl = load_watchlist()
        if wl:
            print(f"\n  {BOLD}{YELLOW}Watchlist:{RESET} {', '.join(wl)}\n")
        else:
            print(f"\n  {DIM}Watchlist empty. Use 'watch <SYM>' to add.{RESET}\n")
        return None

    if action in ("H", "HELP", "?"):
        display_help()
        return None

    # Default: treat as ticker lookup
    print(f"\n  {DIM}Fetching {action}...{RESET}")
    info = fetch_stock_info(action)
    if info:
        display_lookup(info, action)
    else:
        print(f"  {RED}No data for '{action}'{RESET}\n")
    return None


def check_upcoming_earnings() -> None:
    """Show an alert if any tracked symbols report earnings within 7 days."""
    symbols = get_all_symbols()
    print(f"\n  {DIM}Checking earnings...{RESET}", end="", flush=True)
    earnings = fetch_earnings(symbols)
    print(f"\r\033[2K", end="")
    upcoming = [e for e in earnings if e["days_until"] is not None and 0 <= e["days_until"] <= 7]
    if not upcoming:
        return
    upcoming.sort(key=lambda x: x["days_until"])
    print(f"  {RED}{BOLD}⚠ EARNINGS ALERT{RESET}")
    for e in upcoming:
        name = NAMES.get(e["symbol"], e["symbol"])
        if e["days_until"] == 0:
            when = f"{RED}{BOLD}TODAY{RESET}"
        elif e["days_until"] == 1:
            when = f"{RED}TOMORROW{RESET}"
        else:
            when = f"{YELLOW}{e['days_until']}d{RESET}"
        print(f"    {BOLD}{YELLOW}{e['symbol']:<5}{RESET} {DIM}{name:<14}{RESET} {e['date']}  ({when})")
    print()


def main() -> None:
    quotes = show_tape()
    check_upcoming_earnings()
    print(f"  {DIM}Type 'help' for commands, or enter a ticker{RESET}\n")

    while True:
        try:
            cmd = input(f"  {CYAN}ticker>{RESET} ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            cleanup()

        if not cmd:
            continue
        if cmd in ("Q", "QUIT", "EXIT"):
            cleanup()

        result = handle_command(cmd, quotes)
        if result is not None:
            quotes = result


if __name__ == "__main__":
    main()
