"""Breakers screen formatter — pure snapshot dict → Rich markup."""
from screens.breakers import format_breakers


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
