"""Data fetching and persistence for the terminal ticker."""

import json
import logging
import os
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf

from config import SYMBOLS, NAMES, SECTOR_ETFS, WATCHLIST_FILE

ET = ZoneInfo("America/New_York")


def get_all_symbols() -> list[str]:
    """Combined portfolio + watchlist symbols, deduplicated."""
    seen = set()
    result = []
    for s in SYMBOLS + load_watchlist():
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def market_state() -> str:
    """Return 'pre', 'open', 'post', or 'closed' based on US market hours."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if t < time(4, 0):
        return "closed"
    if t < time(9, 30):
        return "pre"
    if t < time(16, 0):
        return "open"
    if t < time(20, 0):
        return "post"
    return "closed"


def fetch_quotes(symbols: list[str]) -> tuple[list[dict], str]:
    """Pull quotes via info dict — includes pre/post market data."""
    quotes = []
    state = market_state()
    tickers = yf.Tickers(" ".join(symbols))

    for sym in symbols:
        try:
            t = tickers.tickers[sym]
            info = t.info

            price = info.get("regularMarketPrice", 0) or 0
            prev = info.get("regularMarketPreviousClose", 0) or 0
            change = price - prev if prev else 0
            pct = (change / prev) * 100 if prev else 0

            q = {"symbol": sym, "price": price, "change": change, "pct": pct}

            # Extended hours data
            pre_price = info.get("preMarketPrice")
            post_price = info.get("postMarketPrice")

            if state == "pre" and pre_price:
                q["ext_price"] = pre_price
                q["ext_change"] = pre_price - prev if prev else 0
                q["ext_pct"] = (q["ext_change"] / prev) * 100 if prev else 0
                q["ext_label"] = "PRE"
            elif state in ("post", "closed") and post_price:
                q["ext_price"] = post_price
                q["ext_change"] = post_price - price
                q["ext_pct"] = (q["ext_change"] / price) * 100 if price else 0
                q["ext_label"] = "AH"

            quotes.append(q)
        except Exception as e:
            quotes.append({"symbol": sym, "price": 0.0, "change": 0.0, "pct": 0.0, "error": str(e)})

    state_labels = {"pre": "Pre-Market", "open": "Market Open", "post": "After Hours", "closed": "Closed"}
    now = datetime.now(ET)
    timestamp = f"{now.strftime('%H:%M ET')} | {state_labels[state]}"

    return quotes, timestamp


def fetch_stock_info(symbol: str) -> dict | None:
    """Fetch comprehensive stock info for lookup."""
    try:
        info = yf.Ticker(symbol).info
        if not info or not info.get("regularMarketPrice"):
            return None
        return info
    except Exception:
        return None


def _to_date(raw) -> date | None:
    """Coerce a date-like value from yfinance to date."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if hasattr(raw, "to_pydatetime"):
        return raw.to_pydatetime().date()
    if hasattr(raw, "date"):
        return raw.date()
    return None


def fetch_earnings(symbols: list[str]) -> list[dict]:
    """Fetch next earnings date for each symbol."""
    results = []
    today = date.today()
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            dates = []
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
            elif hasattr(cal, "loc"):
                try:
                    dates = list(cal.loc["Earnings Date"])
                except (KeyError, TypeError):
                    pass

            if dates:
                d = _to_date(dates[0])
                if d:
                    results.append({"symbol": sym, "date": str(d), "days_until": (d - today).days})
                    continue

            results.append({"symbol": sym, "date": "Unknown", "days_until": None})
        except Exception:
            results.append({"symbol": sym, "date": "Error", "days_until": None})
    return results


MARKET_GROUPS = {
    "US Equity": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "DIA": "Dow 30",
        "IWM": "Russell 2000",
        "SOXX": "Semis (SOX)",
    },
    "World Equity": {
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "000001.SS": "Shanghai Comp",
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX",
        "^GSPTSE": "TSX Comp",
    },
    "Vol & Rates": {
        "^VIX": "VIX",
        "^TNX": "10Y Yield",
        "^TYX": "30Y Yield",
    },
    "Commodities & FX": {
        "GLD": "Gold",
        "CL=F": "Crude Oil WTI",
        "DX-Y.NYB": "US Dollar Idx",
    },
}


def fetch_market_overview() -> dict[str, list[dict]]:
    """Fetch macro market indicators, grouped by category."""
    grouped = {}
    for group_name, symbols in MARKET_GROUPS.items():
        results = []
        for sym, name in symbols.items():
            try:
                fi = yf.Ticker(sym).fast_info
                price = fi.last_price
                prev = fi.previous_close
                change = price - prev
                pct = (change / prev) * 100
                results.append({"symbol": sym, "name": name, "price": price, "change": change, "pct": pct})
            except Exception:
                results.append({"symbol": sym, "name": name, "price": 0.0, "change": 0.0, "pct": 0.0})
        grouped[group_name] = results
    return grouped


def fetch_news(symbol: str, count: int = 8) -> list[dict]:
    """Fetch recent news headlines for a symbol."""
    try:
        t = yf.Ticker(symbol)
        raw = t.news or []
        results = []
        for item in raw[:count]:
            # yfinance >= 1.0 nests everything under "content"
            content = item.get("content", item)

            title = content.get("title", "")
            publisher = ""
            provider = content.get("provider")
            if isinstance(provider, dict):
                publisher = provider.get("displayName", "")
            else:
                publisher = content.get("publisher", "")

            link = ""
            canonical = content.get("canonicalUrl")
            if isinstance(canonical, dict):
                link = canonical.get("url", "")
            else:
                link = content.get("link", "")

            # pubDate is ISO string in new format, Unix timestamp in old
            age_str = ""
            dt = None
            pub_date = content.get("pubDate")
            pub_time = content.get("providerPublishTime")
            if pub_date and isinstance(pub_date, str):
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif pub_time:
                dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)

            if dt:
                age = datetime.now(timezone.utc) - dt
                if age.days > 0:
                    age_str = f"{age.days}d ago"
                elif age.seconds >= 3600:
                    age_str = f"{age.seconds // 3600}h ago"
                else:
                    age_str = f"{age.seconds // 60}m ago"

            results.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "age": age_str,
                "timestamp": dt,
            })
        return results
    except Exception:
        return []


def fetch_technicals(symbol: str) -> dict | None:
    """Calculate technical indicators from historical data."""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 20:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        current = close.iloc[-1]

        sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 15 else None

        # Volume
        avg_vol_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else None
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else None

        # 52w range
        high_52w = close.max()
        low_52w = close.min()
        off_high = ((current - high_52w) / high_52w) * 100
        off_low = ((current - low_52w) / low_52w) * 100

        # Trend signals
        trend_signals = []
        if sma_50 and sma_200:
            trend_signals.append("Golden Cross" if sma_50 > sma_200 else "Death Cross")
        if sma_20:
            trend_signals.append("Above 20d" if current > sma_20 else "Below 20d")
        if sma_50:
            trend_signals.append("Above 50d" if current > sma_50 else "Below 50d")

        return {
            "current": current,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi": rsi,
            "avg_vol_20": avg_vol_20,
            "current_vol": current_vol,
            "vol_ratio": vol_ratio,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "off_high": off_high,
            "off_low": off_low,
            "trend_signals": trend_signals,
        }
    except Exception:
        return None


def fetch_chart_data(symbol: str, period: str = "1mo", interval: str = "1d") -> list[float] | None:
    """Fetch closing prices for sparkline chart."""
    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist["Close"].tolist()
    except Exception:
        return None


def fetch_sector_performance() -> list[dict]:
    """Fetch sector ETF performance for heatmap."""
    results = []
    for sym, name in SECTOR_ETFS.items():
        try:
            fi = yf.Ticker(sym).fast_info
            price = fi.last_price
            prev = fi.previous_close
            change = price - prev
            pct = (change / prev) * 100
            results.append({"symbol": sym, "name": name, "price": price, "change": change, "pct": pct})
        except Exception:
            results.append({"symbol": sym, "name": name, "price": 0.0, "change": 0.0, "pct": 0.0})
    results.sort(key=lambda x: x["pct"], reverse=True)
    return results


log = logging.getLogger(__name__)


def fetch_comparison_data(symbols: list[str], period: str = "1mo") -> dict[str, list[float]] | None:
    """Fetch aligned closing prices for multiple symbols."""
    try:
        df = yf.download(symbols, period=period, interval="1d", progress=False)
        if df.empty:
            return None
        close = df["Close"]
        # Single symbol returns a Series, not DataFrame
        if len(symbols) == 1:
            return {symbols[0]: close.dropna().tolist()}
        result = {}
        for sym in symbols:
            if sym in close.columns:
                series = close[sym].dropna()
                if len(series) >= 2:
                    result[sym] = series.tolist()
        return result if result else None
    except Exception as e:
        log.warning("fetch_comparison_data failed: %s", e)
        return None


def fetch_intraday_data(symbol: str) -> dict | None:
    """Fetch today's 5-min bars with VWAP calculation."""
    try:
        hist = yf.Ticker(symbol).history(period="1d", interval="5m")
        if hist.empty or len(hist) < 2:
            return None

        closes = hist["Close"].values
        volumes = hist["Volume"].values
        highs = hist["High"].values
        lows = hist["Low"].values

        # VWAP: cumulative(typical_price * volume) / cumulative(volume)
        typical = (closes + highs + lows) / 3
        cum_vol = np.cumsum(volumes)
        cum_tp_vol = np.cumsum(typical * volumes)
        # Avoid division by zero for pre-market bars with no volume
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)

        return {
            "prices": closes.tolist(),
            "vwap": vwap.tolist(),
            "current": float(closes[-1]),
            "vwap_current": float(vwap[-1]),
            "high": float(np.max(highs)),
            "low": float(np.min(lows)),
            "volume": int(np.sum(volumes)),
        }
    except Exception as e:
        log.warning("fetch_intraday_data failed for %s: %s", symbol, e)
        return None


def _load_watchlist_data() -> dict:
    """Load raw watchlist JSON."""
    if not os.path.exists(WATCHLIST_FILE):
        return {"symbols": [], "names": {}}
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
            # Migrate old format (just symbols list) to new format
            if "names" not in data:
                data["names"] = {}
            return data
    except Exception:
        return {"symbols": [], "names": {}}


def _save_watchlist_data(data: dict) -> None:
    """Save raw watchlist JSON."""
    data["symbols"] = sorted(set(data["symbols"]))
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_watchlist() -> list[str]:
    """Load watchlist symbols from disk."""
    return _load_watchlist_data().get("symbols", [])


def get_all_names() -> dict[str, str]:
    """Merged name map: portfolio NAMES + watchlist names."""
    wl_data = _load_watchlist_data()
    return {**NAMES, **wl_data.get("names", {})}


def add_to_watchlist(symbol: str) -> bool:
    """Add symbol with name lookup. Returns False if already tracked."""
    sym = symbol.upper()
    if sym in SYMBOLS:
        return False
    data = _load_watchlist_data()
    if sym in data["symbols"]:
        return False
    data["symbols"].append(sym)
    # Fetch display name from yfinance
    try:
        info = yf.Ticker(sym).info
        name = info.get("shortName", sym)
        data["names"][sym] = name
    except Exception:
        data["names"][sym] = sym
    _save_watchlist_data(data)
    return True


def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol. Returns False if not on watchlist."""
    data = _load_watchlist_data()
    sym = symbol.upper()
    if sym not in data["symbols"]:
        return False
    data["symbols"].remove(sym)
    data["names"].pop(sym, None)
    _save_watchlist_data(data)
    return True
