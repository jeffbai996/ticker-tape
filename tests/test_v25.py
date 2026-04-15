"""Tests for v2.5 features — db persistence, smart alerts, earnings surprises,
timeline/surprises/sizing/briefing screen formatters."""

import os
from datetime import date, datetime, timedelta, timezone

import pytest


# ── db.py persistence layer ───────────────────────────────


class TestDBInit:
    def test_init_creates_tables(self, tmp_db):
        import db
        # Tables should exist after init
        tables = db.db.get_tables()
        assert "nlv_snapshots" in tables
        assert "earnings_surprises" in tables

    def test_init_idempotent(self, tmp_db):
        import db
        db.init_db()  # calling again should not error
        tables = db.db.get_tables()
        assert "nlv_snapshots" in tables


class TestNLVSnapshots:
    def test_record_and_query(self, tmp_db):
        import db
        db.record_nlv(500000.0, cushion=18.5, leverage=1.8, daily_pnl=1200.0)
        db.record_nlv(501000.0, cushion=18.7, leverage=1.78, daily_pnl=1500.0)
        history = db.get_nlv_history(days=1)
        assert len(history) == 2
        assert history[0]["nlv"] == 500000.0
        assert history[1]["nlv"] == 501000.0

    def test_peak_query(self, tmp_db):
        import db
        db.record_nlv(500000.0)
        db.record_nlv(510000.0)
        db.record_nlv(505000.0)
        assert db.get_nlv_peak(days=1) == 510000.0

    def test_peak_no_data(self, tmp_db):
        import db
        assert db.get_nlv_peak(days=1) is None

    def test_prune_deletes_old(self, tmp_db):
        import db
        # Insert a snapshot with an old timestamp
        db.NLVSnapshot.create(
            timestamp=datetime.now(timezone.utc) - timedelta(days=100),
            nlv=400000.0,
        )
        db.record_nlv(500000.0)  # recent
        count = db.prune_nlv(days=90)
        assert count == 1
        history = db.get_nlv_history(days=365)
        assert len(history) == 1
        assert history[0]["nlv"] == 500000.0

    def test_nullable_fields(self, tmp_db):
        import db
        db.record_nlv(500000.0)  # no cushion/leverage/pnl
        history = db.get_nlv_history(days=1)
        assert history[0]["cushion"] is None
        assert history[0]["leverage"] is None
        assert history[0]["daily_pnl"] is None


class TestEarningsSurprises:
    def test_upsert_and_query(self, tmp_db):
        import db
        db.upsert_earnings("NVDA", date(2026, 2, 26), 0.89, 0.92, 3.37, 2.1)
        records = db.get_earnings_history(["NVDA"])
        assert len(records) == 1
        assert records[0]["symbol"] == "NVDA"
        assert records[0]["eps_actual"] == 0.92

    def test_upsert_dedup(self, tmp_db):
        import db
        db.upsert_earnings("NVDA", date(2026, 2, 26), 0.89, 0.92)
        db.upsert_earnings("NVDA", date(2026, 2, 26), 0.89, 0.92)  # same date
        records = db.get_earnings_history(["NVDA"])
        assert len(records) == 1  # not duplicated

    def test_earnings_summary(self, tmp_db):
        import db
        db.upsert_earnings("NVDA", date(2026, 2, 26), 0.89, 0.92, 3.37, 2.1)
        db.upsert_earnings("NVDA", date(2025, 11, 20), 0.80, 0.85, 6.25, -1.5)
        summary = db.get_earnings_summary("NVDA")
        assert summary["total"] == 2
        assert summary["beats"] == 2
        assert summary["beat_streak"] == 2
        assert summary["beat_rate"] == 1.0

    def test_summary_no_data(self, tmp_db):
        import db
        assert db.get_earnings_summary("FAKE") is None

    def test_filter_by_symbol(self, tmp_db):
        import db
        db.upsert_earnings("NVDA", date(2026, 2, 26), 0.89, 0.92)
        db.upsert_earnings("AVGO", date(2026, 3, 10), 1.50, 1.55)
        nvda = db.get_earnings_history(["NVDA"])
        assert len(nvda) == 1
        all_recs = db.get_earnings_history()
        assert len(all_recs) == 2


# ── Smart alerts ──────────────────────────────────────────


class TestSmartAlerts:
    def test_add_typed_alert(self, alerts_file):
        from data import add_alert, load_alerts
        a = add_alert("NVDA", ">", 70, alert_type="rsi")
        assert a["type"] == "rsi"
        alerts = load_alerts()
        assert alerts[0]["type"] == "rsi"

    def test_price_alert_default_type(self, alerts_file):
        from data import add_alert
        a = add_alert("NVDA", ">", 150)
        assert a["type"] == "price"

    def test_backward_compat_no_type_field(self, alerts_file):
        """Alerts without type field should be treated as price alerts."""
        from data import save_alerts, evaluate_alerts
        # Simulate old-format alert without 'type' key
        save_alerts([{"id": 1, "symbol": "NVDA", "operator": ">", "value": 100}])
        quotes = [{"symbol": "NVDA", "price": 150.0}]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 1

    def test_evaluate_alerts_skips_non_price(self, alerts_file):
        from data import add_alert, evaluate_alerts
        add_alert("NVDA", ">", 70, alert_type="rsi")
        add_alert("NVDA", ">", 100, alert_type="price")
        quotes = [{"symbol": "NVDA", "price": 150.0}]
        triggered = evaluate_alerts(quotes)
        assert len(triggered) == 1  # only the price alert

    def test_evaluate_technical_rsi(self, alerts_file):
        from data import add_alert, evaluate_technical_alerts
        add_alert("NVDA", ">", 70, alert_type="rsi")
        batch = {"NVDA": {"rsi": 75.0, "current": 160.0}}
        triggered = evaluate_technical_alerts(batch)
        assert len(triggered) == 1
        assert triggered[0]["current_value"] == 75.0

    def test_evaluate_technical_rsi_not_triggered(self, alerts_file):
        from data import add_alert, evaluate_technical_alerts
        add_alert("NVDA", ">", 70, alert_type="rsi")
        batch = {"NVDA": {"rsi": 55.0}}
        triggered = evaluate_technical_alerts(batch)
        assert len(triggered) == 0

    def test_evaluate_technical_sma_cross(self, alerts_file):
        from data import add_alert, evaluate_technical_alerts
        add_alert("NVDA", ">", 200, alert_type="sma_cross")
        batch = {"NVDA": {"rsi": 55.0, "current": 160.0, "sma_200": 155.0}}
        triggered = evaluate_technical_alerts(batch)
        assert len(triggered) == 1
        assert triggered[0]["current_value"] == 160.0

    def test_evaluate_technical_volume(self, alerts_file):
        from data import add_alert, evaluate_technical_alerts
        add_alert("NVDA", ">", 2.0, alert_type="volume")
        batch = {"NVDA": {"vol_ratio": 2.5}}
        triggered = evaluate_technical_alerts(batch)
        assert len(triggered) == 1

    def test_evaluate_cushion(self, alerts_file):
        from data import add_alert, evaluate_cushion_alerts
        add_alert("_ACCOUNT", "<", 15, alert_type="cushion")
        triggered = evaluate_cushion_alerts(12.5)
        assert len(triggered) == 1
        assert triggered[0]["current_value"] == 12.5

    def test_evaluate_cushion_not_triggered(self, alerts_file):
        from data import add_alert, evaluate_cushion_alerts
        add_alert("_ACCOUNT", "<", 10, alert_type="cushion")
        triggered = evaluate_cushion_alerts(15.0)
        assert len(triggered) == 0

    def test_evaluate_technical_missing_symbol(self, alerts_file):
        from data import add_alert, evaluate_technical_alerts
        add_alert("NVDA", ">", 70, alert_type="rsi")
        batch = {"AVGO": {"rsi": 75.0}}  # NVDA not in batch
        triggered = evaluate_technical_alerts(batch)
        assert len(triggered) == 0


# ── Screen formatters ────────────────────────────────────


class TestTimelineFormatter:
    def test_empty_data(self):
        from screens.timeline import format_timeline
        result = format_timeline([], None)
        assert "timeline" in result.lower() or "no" in result.lower() or "empty" in result.lower()

    def test_single_day_insufficient(self):
        from screens.timeline import format_timeline
        snapshots = [{
            "timestamp": datetime.now(timezone.utc),
            "nlv": 500000, "cushion": 18.0, "leverage": 1.8, "daily_pnl": 1000,
        }]
        result = format_timeline(snapshots, 500000)
        assert "500,000" in result

    def test_multi_day_chart(self):
        from screens.timeline import format_timeline
        now = datetime.now(timezone.utc)
        snapshots = []
        for i in range(10):
            snapshots.append({
                "timestamp": now - timedelta(days=10-i),
                "nlv": 500000 + i * 1000,
                "cushion": 18.0, "leverage": 1.8, "daily_pnl": 1000,
            })
        result = format_timeline(snapshots, 510000)
        assert "509,000" in result  # latest NLV
        assert "510,000" in result  # peak
        assert "days" in result


class TestAlertsFormatter:
    def test_empty_shows_all_types(self):
        from screens.alerts import format_alerts
        result = format_alerts([])
        assert "rsi" in result.lower()
        assert "sma" in result.lower()
        assert "vol" in result.lower()
        assert "cushion" in result.lower()

    def test_typed_alert_display(self):
        from screens.alerts import format_alerts
        alerts = [
            {"id": 1, "symbol": "NVDA", "type": "rsi", "operator": ">", "value": 70, "created": "2026-04-13"},
            {"id": 2, "symbol": "AVGO", "type": "sma_cross", "operator": ">", "value": 200, "created": "2026-04-13"},
            {"id": 3, "symbol": "_ACCOUNT", "type": "cushion", "operator": "<", "value": 15, "created": "2026-04-13"},
        ]
        result = format_alerts(alerts)
        assert "RSI" in result
        assert "SMA" in result
        assert "Cushion" in result
        assert "ACCT" in result  # cushion displays as ACCT


class TestSurprisesFormatter:
    def test_empty_data(self):
        from screens.surprises import format_surprises
        result = format_surprises({"symbols": {}, "watchlist_summary": {}})
        assert "no" in result.lower() or "empty" in result.lower()

    def test_with_data(self):
        from screens.surprises import format_surprises
        data = {
            "symbols": {
                "NVDA": {
                    "events": [],
                    "summary": {
                        "total": 4, "beats": 3, "beat_rate": 0.75,
                        "beat_streak": 2, "avg_surprise": 5.0, "avg_move": 3.0,
                        "last_eps": 0.92, "last_surprise": 3.37, "last_move": 2.1,
                    },
                },
            },
            "watchlist_summary": {
                "total_beats": 3, "total_total": 4,
                "avg_beat_rate": 0.75, "avg_surprise": 5.0, "avg_move": 3.0,
            },
        }
        result = format_surprises(data)
        assert "NVDA" in result
        assert "75%" in result  # beat rate


class TestSizingFormatter:
    def test_no_ibkr_data(self):
        from screens.sizing import format_sizing
        result = format_sizing(None, None, None, "NVDA", 100)
        assert "NVDA" in result
        assert "100" in result
        assert "unavail" in result.lower()

    def test_with_whatif_data(self):
        from screens.sizing import format_sizing
        whatif = "Init Margin: $5,000\nMaint Margin: $3,500\nCushion: 18.5%"
        result = format_sizing(whatif, None, None, "NVDA", 100)
        assert "NVDA" in result
        assert "5,000" in result


class TestBriefingFormatter:
    def test_empty_briefing(self):
        from screens.briefing import format_briefing
        data = {
            "portfolio": None,
            "movers": {"gainers": [], "losers": []},
            "earnings_week": [],
            "macro": None,
            "nlv_drawdown": None,
        }
        result = format_briefing(data)
        assert "offline" in result.lower() or "unavail" in result.lower()

    def test_with_portfolio(self):
        from screens.briefing import format_briefing
        data = {
            "portfolio": {"nlv": 500000, "cushion": 18.0, "leverage": 1.8, "daily_pnl": 1200},
            "movers": {
                "gainers": [{"symbol": "NVDA", "pct": 3.5, "price": 160.0}],
                "losers": [{"symbol": "MU", "pct": -2.1, "price": 95.0}],
            },
            "earnings_week": [{"symbol": "AVGO", "date": "2026-04-15", "days_until": 2}],
            "macro": {"sp500_pct": 0.5, "nasdaq_pct": 0.8, "vix": 15.2, "gold": 2350},
            "nlv_drawdown": -2.5,
        }
        result = format_briefing(data)
        assert "500,000" in result
        assert "NVDA" in result
        assert "MU" in result
        assert "AVGO" in result
        assert "15.2" in result  # VIX


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Provide a temp SQLite database for testing."""
    db_path = str(tmp_path / "test_ticker.db")
    monkeypatch.setattr("config.DB_FILE", db_path)
    import db
    # Re-init with the temp path
    if not db.db.is_closed():
        db.db.close()
    db.db.init(db_path, pragmas={"journal_mode": "wal"})
    db.db.connect()
    db.db.create_tables([db.NLVSnapshot, db.EarningsSurprise])
    yield db_path
    db.db.close()
