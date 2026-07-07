"""Tests for decision_cards — trim-ladder math and card builders."""

from decimal import Decimal

import decision_cards as dc


# ── trim_ladder math ──────────────────────────────────────────────────────

def test_ladder_frees_needed_excess_from_largest_first():
    # Cushion 10% → 15% on NLV 1,000,000 needs (0.05 * 1M) = 50,000 excess.
    # At 25% maint, each $ trimmed frees $0.25, so 50,000 / 0.25 = 200,000
    # of position value to trim. Largest position (300k) covers it alone.
    positions = [
        {"symbol": "AAA", "mkt_value": 300000},
        {"symbol": "BBB", "mkt_value": 100000},
    ]
    r = dc.trim_ladder(positions, 1000000, 10, 15)
    assert r["feasible"] is True
    assert r["needed_excess"] == Decimal("50000")
    assert len(r["rungs"]) == 1
    assert r["rungs"][0]["symbol"] == "AAA"
    assert r["rungs"][0]["trim_value"] == Decimal("200000")
    assert r["rungs"][0]["freed_excess"] == Decimal("50000")
    assert r["shortfall"] == Decimal("0")


def test_ladder_spills_to_second_position_when_first_insufficient():
    # Need 200,000 of trim value; largest position only 120,000 → spill to next.
    positions = [
        {"symbol": "AAA", "mkt_value": 120000},
        {"symbol": "BBB", "mkt_value": 90000},
    ]
    r = dc.trim_ladder(positions, 1000000, 10, 15)
    assert r["feasible"] is True
    assert [x["symbol"] for x in r["rungs"]] == ["AAA", "BBB"]  # largest first
    total_trim = sum(x["trim_value"] for x in r["rungs"])
    assert total_trim == Decimal("200000")
    assert r["rungs"][0]["trim_value"] == Decimal("120000")  # AAA fully used
    assert r["rungs"][1]["trim_value"] == Decimal("80000")   # BBB covers rest


def test_ladder_never_trims_more_than_held():
    positions = [{"symbol": "AAA", "mkt_value": 10000}]
    r = dc.trim_ladder(positions, 1000000, 10, 15)  # needs 200k trim, only 10k exists
    assert r["feasible"] is False
    assert r["rungs"][0]["trim_value"] == Decimal("10000")  # capped at holding
    assert r["shortfall"] > 0


def test_ladder_no_op_when_cushion_already_above_target():
    r = dc.trim_ladder([{"symbol": "AAA", "mkt_value": 100000}], 1000000, 20, 15)
    assert r["feasible"] is True
    assert r["rungs"] == []
    assert "already" in r["reason"]


def test_ladder_bad_inputs_return_infeasible_with_reason():
    r = dc.trim_ladder([], None, 10, 15)
    assert r["feasible"] is False
    assert r["reason"] == "insufficient account data"


def test_ladder_ignores_short_and_zero_positions():
    positions = [
        {"symbol": "SHORT", "mkt_value": -50000},
        {"symbol": "ZERO", "mkt_value": 0},
        {"symbol": "LONG", "mkt_value": 300000},
    ]
    r = dc.trim_ladder(positions, 1000000, 10, 15)
    assert [x["symbol"] for x in r["rungs"]] == ["LONG"]


# ── card builders ─────────────────────────────────────────────────────────

def test_cushion_card_no_ibkr_gives_static_note():
    card = dc.build_cushion_card(None, None, 10, 15)
    assert "Connect IBKR" in card
    assert "trim" in card.lower()


def test_cushion_card_with_data_lists_ladder():
    positions = [{"symbol": "AAA", "mkt_value": 300000}]
    card = dc.build_cushion_card(positions, 1000000, 10, 15)
    assert "AAA" in card
    assert "trim" in card.lower()
    assert "%" in card  # states the maint-rate assumption


def test_cushion_card_flags_infeasible():
    positions = [{"symbol": "AAA", "mkt_value": 10000}]
    card = dc.build_cushion_card(positions, 1000000, 10, 15)
    assert "short" in card.lower()


def test_price_card_held_shows_position():
    alert = {"symbol": "MSFT", "type": "price"}
    pos = {"symbol": "MSFT", "qty": 100, "avg_cost": 300, "unrealized": 5000}
    card = dc.build_price_card(alert, pos)
    assert "MSFT" in card
    assert "100" in card
    assert "Held" in card


def test_price_card_not_held():
    alert = {"symbol": "GOOGL", "type": "rsi"}
    card = dc.build_price_card(alert, None)
    assert "GOOGL" in card
    assert "Not currently held" in card


def test_parse_positions_markdown():
    text = (
        "| Symbol | Shares | Avg Cost | Mkt Price | Mkt Value | Unreal P&L | Weight | Currency |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| AAPL | 340 | $150.00 | $54.18 | $18,414 USD | $-952 | 38.1% | USD |\n"
        "| MSFT |  86 | $80.00  | $94.32 | $8,112 USD  | $-892 | 22.3% | USD |\n"
    )
    rows = dc.parse_positions_markdown(text)
    assert len(rows) == 2
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["mkt_value"] == Decimal("18414")
    assert rows[0]["qty"] == Decimal("340")
    assert rows[1]["symbol"] == "MSFT"


def test_parse_positions_markdown_empty_and_junk():
    assert dc.parse_positions_markdown(None) == []
    assert dc.parse_positions_markdown("no table here") == []
    # Header/separator only → no data rows.
    assert dc.parse_positions_markdown("| Symbol | Mkt Value |\n|---|---|") == []


def test_money_formatting():
    assert dc._money(Decimal("1500000")) == "$1.5M"
    assert dc._money(Decimal("50000")) == "$50K"
    assert dc._money(Decimal("999")) == "$999"
    assert dc._money(Decimal("-50000")) == "-$50K"
    assert dc._money(None) == "n/a"
