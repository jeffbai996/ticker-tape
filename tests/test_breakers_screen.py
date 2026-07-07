"""Breakers screen formatter — pure snapshot dict → Rich markup."""
import time

from screens.breakers import _age, format_breakers


def _snap(**kw):
    base = {"available": True, "last_run": 1751900000,
            "breakers": [
                {"id": "rates_high", "verdict": "CLEAR", "category": "macro",
                 "severity": "trim", "reason": "", "auto": True, "swept": False},
                {"id": "capex_cut", "verdict": "FIRED", "category": "capex",
                 "severity": "reunderwrite", "reason": "guided down",
                 "auto": False, "swept": True},
            ],
            "candidates": [{"id": 3, "breaker_id": "capex_cut",
                            "summary": "vendor guided down", "url": "https://x"}]}
    base.update(kw)
    return base


def test_fired_breaker_rendered_red_with_reason():
    out = format_breakers(_snap())
    assert "capex_cut" in out and "FIRED" in out and "guided down" in out
    assert "red" in out.lower() or "#f" in out.lower()  # NEG style applied


def test_candidates_section_with_review_hint():
    out = format_breakers(_snap())
    assert "#3" in out and "vendor guided down" in out
    assert "candidates confirm 3" in out


def test_unavailable_state_is_honest():
    out = format_breakers({"available": False, "breakers": [],
                           "candidates": [], "last_run": None})
    assert "no watcher state" in out.lower() or "unavailable" in out.lower()


# ── _age() timezone correctness ─────────────────────────────
# House discipline (data.py:29 ZoneInfo("America/New_York"), db.py's
# datetime.now(timezone.utc) pairing) does all wall-clock-adjacent math with
# explicit tz-aware datetimes. _age() previously subtracted two NAIVE
# datetimes (datetime.now() and datetime.fromtimestamp(epoch)), which only
# happens to cancel out because both sides silently assume the same process
# local tz — a subtler, less auditable version of the same discipline every
# other module follows explicitly. This locks in the correct elapsed-time
# math using unambiguous UTC epoch arithmetic.

def test_age_one_hour_ago_reads_1h():
    epoch = time.time() - 3600
    out = _age(epoch)
    assert out == "1h ago"


def test_age_recent_reads_minutes():
    epoch = time.time() - 300  # 5 minutes ago
    out = _age(epoch)
    assert out.endswith("m ago")
    assert "5" in out


def test_age_days_reads_days():
    epoch = time.time() - 3 * 86400  # 3 days ago
    out = _age(epoch)
    assert out == "3d ago"


def test_age_none_epoch_returns_empty():
    assert _age(None) == ""


def test_age_zero_epoch_returns_empty():
    # falsy epoch (0) must short-circuit same as None, not render "0m ago"
    assert _age(0) == ""


def test_age_stable_across_process_local_timezone(monkeypatch):
    """The age computation must not depend on the process's local TZ setting.

    Regression guard for the naive datetime.now() vs. datetime.fromtimestamp()
    subtraction: switch the process to a non-US timezone and confirm the
    computed age of a fixed real-world epoch is unchanged.
    """
    epoch = time.time() - 7200  # 2 hours ago, real wall-clock time

    import os
    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Shanghai"
        time.tzset()
        shanghai_result = _age(epoch)

        os.environ["TZ"] = "America/New_York"
        time.tzset()
        ny_result = _age(epoch)
    finally:
        if original_tz is not None:
            os.environ["TZ"] = original_tz
        else:
            os.environ.pop("TZ", None)
        time.tzset()

    assert shanghai_result == ny_result == "2h ago"


def test_age_uses_timezone_aware_now(monkeypatch):
    """Regression lock: _age() must call datetime.now(timezone.utc), not the
    naive datetime.now(). Patches screens.breakers.datetime.now to assert it
    is always invoked with a tz argument."""
    import screens.breakers as breakers_mod
    from datetime import timezone as _tz

    real_datetime = breakers_mod.datetime

    class _SpyDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is not None, "_age() must pass timezone.utc to datetime.now()"
            assert tz == _tz.utc
            return real_datetime.now(tz)

    monkeypatch.setattr(breakers_mod, "datetime", _SpyDatetime)
    epoch = time.time() - 3600
    assert _age(epoch) == "1h ago"
