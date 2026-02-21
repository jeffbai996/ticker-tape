#!/usr/bin/env python3
"""Terminal stock ticker tape with interactive stock lookup."""

import sys
import signal

from config import CYAN, DIM, BOLD, RESET, GREEN, RED, YELLOW
from data import (
    get_all_symbols, fetch_quotes, fetch_stock_info, fetch_earnings,
    fetch_market_overview, fetch_news, fetch_technicals, fetch_chart_data,
    fetch_sector_performance, load_watchlist, add_to_watchlist, remove_from_watchlist,
    fetch_comparison_data, fetch_intraday_data, get_all_names,
    fetch_earnings_impact, fetch_batch_info,
    load_alerts, add_alert, remove_alert, evaluate_alerts,
)
from views import (
    scrolling_tape, display_lookup, display_thesis, display_earnings,
    display_market, display_news, display_technicals, display_chart,
    display_sectors, display_help, auto_refresh_tape,
    display_comparison, display_intraday,
    display_earnings_impact, display_screen,
    display_alerts, display_triggered_alerts,
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

    if action == "VS":
        if not arg or len(arg.split()) < 2:
            print(f"\n  {DIM}Usage: vs <SYM> <SYM> [SYM...] [period]{RESET}")
            print(f"  {DIM}Example: vs NVDA MU AVGO 3mo{RESET}\n")
        else:
            vs_parts = arg.split()
            valid_periods = ("1w", "1mo", "3mo", "6mo", "1y", "2y", "ytd")
            # Check if last arg is a period
            if vs_parts[-1].lower() in valid_periods:
                vs_syms = vs_parts[:-1]
                vs_period = vs_parts[-1].lower()
            else:
                vs_syms = vs_parts
                vs_period = "1mo"
            print(f"\n  {DIM}Fetching comparison data...{RESET}")
            comp_data = fetch_comparison_data(vs_syms, vs_period)
            if comp_data:
                display_comparison(comp_data, vs_syms, vs_period)
            else:
                print(f"  {RED}No data available{RESET}\n")
        return None

    if action in ("I", "INTRA"):
        if not arg:
            print(f"\n  {DIM}Usage: intra <SYMBOL>{RESET}\n")
        else:
            sym = arg.split()[0]
            print(f"\n  {DIM}Fetching intraday data for {sym}...{RESET}")
            intra = fetch_intraday_data(sym)
            if intra:
                display_intraday(intra, sym)
            else:
                print(f"  {RED}No intraday data for '{sym}'{RESET}\n")
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

    if action in ("W", "WATCH"):
        if not arg:
            print(f"\n  {DIM}Usage: watch <SYMBOL>{RESET}\n")
        else:
            print(f"\n  {DIM}Adding {arg}...{RESET}", end="", flush=True)
            if add_to_watchlist(arg):
                print(f"\r\033[2K\n  {GREEN}Added {arg} to watchlist — refreshing tape...{RESET}")
                return show_tape()
            else:
                print(f"\r\033[2K\n  {DIM}{arg} already tracked{RESET}\n")
        return None

    if action in ("UW", "UNWATCH"):
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
            names = get_all_names()
            print(f"\n  {BOLD}{YELLOW}═══ WATCHLIST ═══{RESET}\n")
            for sym in wl:
                name = names.get(sym, sym)
                print(f"  {BOLD}{YELLOW}{sym:<6}{RESET} {DIM}{name}{RESET}")
            print()
        else:
            print(f"\n  {DIM}Watchlist empty. Use 'watch <SYM>' to add.{RESET}\n")
        return None

    if action in ("IMPACT", "EI"):
        if not arg:
            print(f"\n  {DIM}Usage: impact <SYMBOL>{RESET}\n")
        else:
            sym = arg.split()[0]
            print(f"\n  {DIM}Fetching earnings impact for {sym}...{RESET}")
            impact = fetch_earnings_impact(sym)
            if impact:
                display_earnings_impact(impact, sym)
            else:
                print(f"  {RED}No earnings data for '{sym}'{RESET}\n")
        return None

    if action == "SCREEN":
        if not arg or len(arg.split()) < 2:
            print(f"\n  {DIM}Usage: screen <SYM> <SYM> [SYM...]{RESET}")
            print(f"  {DIM}Example: screen NVDA MU AVGO LRCX{RESET}\n")
        else:
            syms = arg.split()
            print(f"\n  {DIM}Fetching data for {', '.join(syms)}...{RESET}")
            infos = fetch_batch_info(syms)
            if infos:
                display_screen(infos, syms)
            else:
                print(f"  {RED}No data available{RESET}\n")
        return None

    if action in ("ALERT", "AL"):
        # Subcommands: alert (list), alert rm N (remove), alert SYM >N (add)
        if not arg:
            display_alerts(load_alerts())
            return None
        alert_parts = arg.split()
        if alert_parts[0] == "RM" and len(alert_parts) >= 2:
            try:
                aid = int(alert_parts[1])
                if remove_alert(aid):
                    print(f"\n  {GREEN}Removed alert #{aid}{RESET}\n")
                else:
                    print(f"\n  {RED}Alert #{aid} not found{RESET}\n")
            except ValueError:
                print(f"\n  {RED}Invalid alert ID{RESET}\n")
            return None
        # Parse: SYM >N or SYM <N  (also handles "SYM > N" with spaces)
        if len(alert_parts) >= 2:
            sym = alert_parts[0]
            cond = alert_parts[1]
            op = None
            val = None
            if cond.startswith(">"):
                op = ">"
                try:
                    val = float(cond[1:])
                except ValueError:
                    pass
            elif cond.startswith("<"):
                op = "<"
                try:
                    val = float(cond[1:])
                except ValueError:
                    pass
            # Handle space between operator and value: "NVDA > 150"
            if op and val is None and len(alert_parts) >= 3:
                try:
                    val = float(alert_parts[2])
                except ValueError:
                    pass
            # Handle bare operator as separate token: "NVDA > 150"
            if op is None and cond in (">", "<") and len(alert_parts) >= 3:
                op = cond
                try:
                    val = float(alert_parts[2])
                except ValueError:
                    pass
            if op and val is not None:
                a = add_alert(sym, op, val)
                print(f"\n  {GREEN}Alert #{a['id']}: {a['symbol']} {a['operator']}{a['value']:.2f}{RESET}\n")
                return None
        print(f"\n  {DIM}Usage: alert NVDA >150 | alert | alert rm 1{RESET}\n")
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
    names = get_all_names()
    print(f"  {RED}{BOLD}⚠ EARNINGS ALERT{RESET}")
    for e in upcoming:
        name = names.get(e["symbol"], e["symbol"])
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
    # Check price alerts against fresh quotes
    triggered = evaluate_alerts(quotes)
    display_triggered_alerts(triggered)
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
