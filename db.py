"""SQLite persistence layer for time-series data (NLV history, earnings surprises).

Uses peewee ORM with WAL mode for thread-safe writes from sidebar worker threads.
Single database at data/ticker.db — two tables, one file.
"""

import logging
from datetime import date, datetime, timedelta, timezone

from peewee import (
    AutoField,
    CharField,
    DateField,
    DateTimeField,
    FloatField,
    Model,
    SqliteDatabase,
    fn,
)

from config import DB_FILE

log = logging.getLogger(__name__)

db = SqliteDatabase(None)  # deferred — opened in init_db()

# Counter for periodic pruning (avoid checking every insert)
_insert_count = 0


class BaseModel(Model):
    class Meta:
        database = db


class NLVSnapshot(BaseModel):
    id = AutoField()
    timestamp = DateTimeField(index=True)
    nlv = FloatField()
    cushion = FloatField(null=True)
    leverage = FloatField(null=True)
    daily_pnl = FloatField(null=True)

    class Meta:
        table_name = "nlv_snapshots"


class EarningsSurprise(BaseModel):
    id = AutoField()
    symbol = CharField(index=True)
    date = DateField()
    eps_est = FloatField(null=True)
    eps_actual = FloatField()
    surprise_pct = FloatField(null=True)
    price_move = FloatField(null=True)
    fetched_at = DateTimeField()

    class Meta:
        table_name = "earnings_surprises"
        indexes = ((("symbol", "date"), True),)  # unique together


# ── Public API ────────────────────────────────────────────


def init_db() -> None:
    """Open the database and create tables if they don't exist."""
    db.init(DB_FILE, pragmas={
        "journal_mode": "wal",
        "cache_size": -8000,       # 8 MB
        "synchronous": "normal",   # safe with WAL
    })
    db.connect(reuse_if_open=True)
    db.create_tables([NLVSnapshot, EarningsSurprise])
    log.info("Database initialized at %s", DB_FILE)


# ── NLV snapshots ─────────────────────────────────────────


def record_nlv(
    nlv: float,
    cushion: float | None = None,
    leverage: float | None = None,
    daily_pnl: float | None = None,
) -> None:
    """Insert an NLV snapshot. Auto-prunes every ~100 inserts."""
    global _insert_count
    try:
        NLVSnapshot.create(
            timestamp=datetime.now(timezone.utc),
            nlv=nlv,
            cushion=cushion,
            leverage=leverage,
            daily_pnl=daily_pnl,
        )
        _insert_count += 1
        if _insert_count % 100 == 0:
            prune_nlv()
    except Exception as e:
        log.warning("Failed to record NLV snapshot: %s", e)


def get_nlv_history(days: int = 90) -> list[dict]:
    """Return NLV snapshots within the given day range, oldest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        NLVSnapshot.select()
        .where(NLVSnapshot.timestamp >= cutoff)
        .order_by(NLVSnapshot.timestamp)
    )
    return [
        {
            "timestamp": r.timestamp,
            "nlv": r.nlv,
            "cushion": r.cushion,
            "leverage": r.leverage,
            "daily_pnl": r.daily_pnl,
        }
        for r in rows
    ]


def get_nlv_peak(days: int = 90) -> float | None:
    """Return the max NLV in the given window, or None if no data."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = (
        NLVSnapshot.select(fn.MAX(NLVSnapshot.nlv))
        .where(NLVSnapshot.timestamp >= cutoff)
        .scalar()
    )
    return result


def prune_nlv(days: int = 90) -> int:
    """Delete snapshots older than the retention window. Returns count deleted."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = NLVSnapshot.delete().where(NLVSnapshot.timestamp < cutoff).execute()
    if count:
        log.info("Pruned %d NLV snapshots older than %d days", count, days)
    return count


# ── Earnings surprises ────────────────────────────────────


def upsert_earnings(
    symbol: str,
    earn_date: date,
    eps_est: float | None,
    eps_actual: float,
    surprise_pct: float | None = None,
    price_move: float | None = None,
) -> bool:
    """Insert an earnings record, skip if (symbol, date) already exists.

    Returns True if inserted, False if duplicate skipped.
    """
    try:
        EarningsSurprise.insert(
            symbol=symbol.upper(),
            date=earn_date,
            eps_est=eps_est,
            eps_actual=eps_actual,
            surprise_pct=surprise_pct,
            price_move=price_move,
            fetched_at=datetime.now(timezone.utc),
        ).on_conflict_ignore().execute()
        return True
    except Exception as e:
        log.warning("Failed to upsert earnings for %s: %s", symbol, e)
        return False


def get_earnings_history(
    symbols: list[str] | None = None,
) -> list[dict]:
    """Return all persisted earnings records, optionally filtered by symbols."""
    query = EarningsSurprise.select().order_by(
        EarningsSurprise.symbol, EarningsSurprise.date.desc()
    )
    if symbols:
        query = query.where(
            EarningsSurprise.symbol << [s.upper() for s in symbols]
        )
    return [
        {
            "symbol": r.symbol,
            "date": r.date,
            "eps_est": r.eps_est,
            "eps_actual": r.eps_actual,
            "surprise_pct": r.surprise_pct,
            "price_move": r.price_move,
            "fetched_at": r.fetched_at,
        }
        for r in query
    ]


def get_earnings_summary(symbol: str) -> dict | None:
    """Compute beat/miss stats for a single symbol from stored data."""
    rows = (
        EarningsSurprise.select()
        .where(EarningsSurprise.symbol == symbol.upper())
        .order_by(EarningsSurprise.date.desc())
    )
    records = list(rows)
    if not records:
        return None

    surprises = [r.surprise_pct for r in records if r.surprise_pct is not None]
    moves = [r.price_move for r in records if r.price_move is not None]
    beats = sum(1 for s in surprises if s > 0)
    total = len(surprises)

    # Beat streak — consecutive beats from most recent
    beat_streak = 0
    for r in records:
        if r.surprise_pct is not None and r.surprise_pct > 0:
            beat_streak += 1
        else:
            break

    return {
        "symbol": symbol.upper(),
        "total": total,
        "beats": beats,
        "beat_rate": beats / total if total else None,
        "beat_streak": beat_streak,
        "avg_surprise": sum(surprises) / len(surprises) if surprises else None,
        "avg_move": sum(moves) / len(moves) if moves else None,
        "last_eps": records[0].eps_actual if records else None,
        "last_surprise": records[0].surprise_pct if records else None,
        "last_move": records[0].price_move if records else None,
    }
