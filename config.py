"""Configuration constants for the terminal ticker."""

import os

# Portfolio holdings
SYMBOLS = ["MU", "AVGO", "NVDA", "GOOG", "LRCX", "RKLB"]
NAMES = {
    "MU": "Micron",
    "AVGO": "Broadcom",
    "NVDA": "NVIDIA",
    "GOOG": "Alphabet",
    "LRCX": "Lam Research",
    "RKLB": "Rocket Lab",
}

# Thesis groupings
THESIS_BUCKETS = {
    "Infrastructure": ["NVDA", "MU", "AVGO", "LRCX"],
    "Applications": ["GOOG"],
    "Uncorrelated": ["RKLB"],
}

# Sector ETFs for heatmap
SECTOR_ETFS = {
    "XLK": "Technology",
    "SMH": "Semiconductors",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Comm. Svcs",
}

# ANSI colors
GREEN = "\033[38;2;0;255;0m"
RED = "\033[38;2;255;50;50m"
YELLOW = "\033[38;2;255;200;0m"
CYAN = "\033[38;2;0;200;255m"
WHITE = "\033[38;2;255;255;255m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MAGENTA = "\033[38;2;200;100;255m"

# Watchlist persistence
WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")
