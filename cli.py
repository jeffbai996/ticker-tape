#!/usr/bin/env python3
"""cli.py — headless ticker-tape: Rich-formatted market data to stdout.

Reuses ticker-tape's data layer and screen formatters without the Textual TUI.
Supports both yfinance market data and IBKR MCP portfolio data.

Usage:
    python cli.py <command> [args...]

Market Data (yfinance):
    thesis                  Full thesis dashboard
    market                  Market overview
    sectors                 Sector performance
    earnings                Earnings data
    lookup <SYM>            Stock info
    ta <SYM>                Technical analysis
    news <SYM>              News headlines
    chart <SYM> [period]    Price chart (1d,5d,1mo,3mo,6mo,1y,2y)
    compare <S1,S2> [period] Performance comparison
    intraday <SYM>          Intraday data
    impact <SYM>            Earnings impact analysis
    valuation <S1,S2,...>   Valuation comparison
    heatmap                 Portfolio heatmap
    calendar                Economic calendar
    insider <SYM>           Insider transactions
    commodities             Commodity futures prices

IBKR (needs MCP server):
    positions               Portfolio positions
    account                 Account summary
    pnl                     Daily P&L
    trades [N]              Trade history (N = days back, max 7)
    ibkr                    All IBKR data (positions + account + P&L)
    margin <SYM> <QTY>      Margin simulation
    whatif <SYM> <QTY>      What-if order analysis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

console = Console()


def _title(text: str) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S ET")
    return f"[bold #ffc800]═══ {text} ═══[/]  [dim]{ts}[/]"


# ── Market data commands (yfinance) ─────────────────────────────


def cmd_thesis() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import data
    from screens.thesis import format_thesis

    symbols = data.get_all_symbols()
    quotes, _ = data.fetch_quotes(symbols)
    valid = [q["symbol"] for q in quotes if not q.get("error") and q["price"] != 0.0]

    pool = ThreadPoolExecutor(max_workers=4)
    chart_f = pool.submit(data.fetch_chart_data_batch, valid, "1mo", "1d")
    ta_f = pool.submit(data.fetch_technicals_batch, valid)
    earn_f = pool.submit(data.fetch_earnings, symbols)

    def _market_ctx():
        syms_keys = [("^GSPC", "idx.sp"), ("^VIX", "idx.vix"),
                     ("DX-Y.NYB", "idx.dxy"), ("^TNX", "idx.10y"),
                     ("CL=F", "idx.wti"), ("GC=F", "idx.gold"),
                     ("HG=F", "idx.copper"), ("NG=F", "idx.natgas"),
                     ("BTC-USD", "idx.btc"), ("^SOX", "idx.sox")]
        bp = data.bulk_prices([s for s, _ in syms_keys])
        ctx = {}
        for sym, key in syms_keys:
            pm = bp.get(sym)
            if pm and pm[0] and pm[1]:
                ctx[key] = {"price": pm[0], "pct": ((pm[0] - pm[1]) / pm[1]) * 100, "show_pct": True}
        return ctx

    mkt_f = pool.submit(_market_ctx)

    charts = _safe_result(chart_f, {})
    ta = _safe_result(ta_f, {})
    earnings = _safe_result(earn_f, None)
    mkt = _safe_result(mkt_f, {})
    pool.shutdown(wait=False)

    console.print(_title("Thesis"))
    console.print(format_thesis(quotes, charts, earnings, ta, market_ctx=mkt))


def cmd_market() -> None:
    import data
    from screens.market import format_market
    console.print(_title("Market"))
    console.print(format_market(data.fetch_market_overview()))


def cmd_sectors() -> None:
    import data
    from screens.sectors import format_sectors
    console.print(_title("Sectors"))
    console.print(format_sectors(data.fetch_sector_performance()))


def cmd_earnings() -> None:
    import data
    from screens.earnings import format_earnings
    console.print(_title("Earnings"))
    console.print(format_earnings(data.fetch_earnings(data.get_all_symbols())))


def cmd_lookup(sym: str) -> None:
    import data
    from screens.lookup import format_lookup
    info = data.fetch_stock_info(sym)
    if info:
        console.print(_title(sym))
        console.print(format_lookup(info, sym))
    else:
        console.print(f"[#ff3232]No data for {sym}[/]")


def cmd_technicals(sym: str) -> None:
    import data
    from screens.technicals import format_technicals
    console.print(_title(f"Technicals: {sym}"))
    console.print(format_technicals(data.fetch_technicals(sym), sym))


def cmd_news(sym: str) -> None:
    import data
    from screens.news import format_news
    console.print(_title(f"News: {sym}"))
    console.print(format_news(data.fetch_news(sym), sym))


def cmd_chart(sym: str, period: str = "1mo") -> None:
    import data
    from screens.chart import format_chart
    console.print(_title(f"Chart: {sym} ({period})"))
    console.print(format_chart(data.fetch_chart_data(sym, period=period), sym, period))


def cmd_compare(symbols: list[str], period: str = "1mo") -> None:
    import data
    from screens.comparison import format_comparison
    result = data.fetch_comparison_data(symbols, period)
    console.print(_title("Compare"))
    console.print(format_comparison(result, symbols, period))


def cmd_intraday(sym: str) -> None:
    import data
    from screens.intraday import format_intraday
    console.print(_title(f"Intraday: {sym}"))
    console.print(format_intraday(data.fetch_intraday_data(sym), sym))


def cmd_impact(sym: str) -> None:
    import data
    from screens.impact import format_impact
    console.print(_title(f"Impact: {sym}"))
    console.print(format_impact(data.fetch_earnings_impact(sym), sym))


def cmd_valuation(symbols: list[str]) -> None:
    import data
    from screens.valuation import format_valuation
    console.print(_title("Valuation"))
    console.print(format_valuation(data.fetch_batch_info(symbols), symbols))


def cmd_heatmap() -> None:
    import data
    from screens.heatmap import format_heatmap
    quotes, _ = data.fetch_quotes(data.get_all_symbols())
    console.print(_title("Heatmap"))
    console.print(format_heatmap(quotes))


def cmd_calendar() -> None:
    from screens.calendar_screen import format_econ_calendar
    console.print(_title("Economic Calendar"))
    console.print(format_econ_calendar())


def cmd_insider(sym: str) -> None:
    import data
    from screens.insider import format_insider
    console.print(_title(f"Insider: {sym}"))
    console.print(format_insider(data.fetch_insider(sym), sym))


def cmd_commodities() -> None:
    import data
    from screens.commodities import format_commodities
    console.print(_title("Commodities"))
    console.print(format_commodities(data.fetch_commodities()))


# ── IBKR commands (MCP) ────────────────────────────────────────


def _ibkr_call(tool: str, fmt, title: str, arguments: dict | None = None) -> None:
    from ibkr_client import call_ibkr_tool
    from config import IBKR_ACCOUNTS
    for url, label, acct in IBKR_ACCOUNTS:
        tag = f" ({label})" if label else ""
        raw = call_ibkr_tool(tool, arguments, url=url, account=acct)
        console.print(_title(f"{title}{tag}"))
        if raw:
            console.print(fmt(raw))
        else:
            console.print("[#ff3232]IBKR gateway unavailable[/]")


def cmd_positions() -> None:
    from screens.ibkr_screen import format_positions
    _ibkr_call("ibkr_get_positions", format_positions, "Positions")


def cmd_account() -> None:
    from screens.ibkr_screen import format_account
    _ibkr_call("ibkr_get_account_summary", format_account, "Account")


def cmd_pnl() -> None:
    from screens.ibkr_screen import format_pnl
    _ibkr_call("ibkr_get_account_pnl", format_pnl, "P&L")


def cmd_trades(days: int = 0) -> None:
    from screens.ibkr_screen import format_trades
    args = {"params": {"view": "journal", "days_back": days}} if days else None
    title = f"Trades ({days}d)" if days else "Trades"
    _ibkr_call("ibkr_trades", format_trades, title, arguments=args)


def cmd_margin(sym: str, qty: int) -> None:
    from screens.ibkr_screen import format_margin
    _ibkr_call("ibkr_margin", format_margin, "Margin",
               arguments={"params": {"symbol": sym, "quantity": qty, "action": "BUY"}})


def cmd_whatif(sym: str, qty: int) -> None:
    from screens.ibkr_screen import format_whatif
    _ibkr_call("ibkr_what_if", format_whatif, "What-If",
               arguments={"params": {"symbol": sym, "quantity": qty, "action": "BUY"}})


def cmd_ibkr_all() -> None:
    from ibkr_client import call_ibkr_tool
    from config import IBKR_ACCOUNTS
    from screens.ibkr_screen import format_positions, format_account, format_pnl
    for url, label, acct in IBKR_ACCOUNTS:
        tag = f" ({label})" if label else ""
        for tool, fmt, name in [
            ("ibkr_get_positions", format_positions, "Positions"),
            ("ibkr_get_account_summary", format_account, "Account"),
            ("ibkr_get_account_pnl", format_pnl, "P&L"),
        ]:
            raw = call_ibkr_tool(tool, url=url, account=acct)
            console.print(_title(f"{name}{tag}"))
            if raw:
                console.print(fmt(raw))
            else:
                console.print("[dim]unavailable[/]")
            console.print()


# ── Helpers ─────────────────────────────────────────────────────


def _safe_result(future, default, timeout: int = 30):
    try:
        return future.result(timeout=timeout)
    except Exception:
        return default


def _usage() -> None:
    console.print(__doc__)


# ── Dispatch ────────────────────────────────────────────────────

COMMANDS = {
    # yfinance
    "thesis": lambda a: cmd_thesis(),
    "market": lambda a: cmd_market(),
    "sectors": lambda a: cmd_sectors(),
    "earnings": lambda a: cmd_earnings(),
    "lookup": lambda a: cmd_lookup(a[0].upper()) if a else _usage(),
    "ta": lambda a: cmd_technicals(a[0].upper()) if a else _usage(),
    "news": lambda a: cmd_news(a[0].upper()) if a else _usage(),
    "chart": lambda a: cmd_chart(a[0].upper(), a[1] if len(a) > 1 else "1mo") if a else _usage(),
    "compare": lambda a: cmd_compare([s.upper() for s in a[0].split(",")], a[1] if len(a) > 1 else "1mo") if a else _usage(),
    "intraday": lambda a: cmd_intraday(a[0].upper()) if a else _usage(),
    "impact": lambda a: cmd_impact(a[0].upper()) if a else _usage(),
    "valuation": lambda a: cmd_valuation([s.upper() for s in a[0].split(",")]) if a else _usage(),
    "heatmap": lambda a: cmd_heatmap(),
    "calendar": lambda a: cmd_calendar(),
    "insider": lambda a: cmd_insider(a[0].upper()) if a else _usage(),
    "commodities": lambda a: cmd_commodities(),
    # ibkr
    "positions": lambda a: cmd_positions(),
    "account": lambda a: cmd_account(),
    "pnl": lambda a: cmd_pnl(),
    "trades": lambda a: cmd_trades(int(a[0]) if a else 0),
    "ibkr": lambda a: cmd_ibkr_all(),
    "margin": lambda a: cmd_margin(a[0].upper(), int(a[1])) if len(a) >= 2 else _usage(),
    "whatif": lambda a: cmd_whatif(a[0].upper(), int(a[1])) if len(a) >= 2 else _usage(),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _usage()
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd not in COMMANDS:
        console.print(f"[#ff3232]Unknown command: {cmd}[/]\n")
        _usage()
        return

    try:
        COMMANDS[cmd](args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[#ff3232]Error: {e}[/]")
        raise


if __name__ == "__main__":
    main()
