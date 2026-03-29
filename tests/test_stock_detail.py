"""Tests for stock detail screens — dividends, short interest, ratings."""

from screens.stock_detail import format_dividends, format_short_interest, format_ratings


class TestFormatDividends:
    def test_no_data(self):
        result = format_dividends(None, "AAPL")
        assert "No data" in result

    def test_no_dividend(self):
        result = format_dividends({"regularMarketPrice": 150}, "TSLA")
        assert "does not pay" in result

    def test_basic_dividend(self):
        info = {
            "dividendYield": 0.025,
            "dividendRate": 1.00,
            "payoutRatio": 0.30,
        }
        result = format_dividends(info, "AAPL")
        assert "2.50%" in result
        assert "$1.00" in result
        assert "30.0%" in result

    def test_high_payout_ratio_red(self):
        info = {"dividendYield": 0.05, "payoutRatio": 0.9}
        result = format_dividends(info, "T")
        assert "ff3232" in result  # red for high payout


class TestFormatShortInterest:
    def test_no_data(self):
        result = format_short_interest(None, "AAPL")
        assert "No data" in result

    def test_no_short_data(self):
        result = format_short_interest({"regularMarketPrice": 150}, "AAPL")
        assert "No short interest" in result

    def test_basic_short(self):
        info = {
            "shortPercentOfFloat": 0.08,
            "shortRatio": 2.5,
            "sharesShort": 15_000_000,
            "sharesShortPriorMonth": 12_000_000,
        }
        result = format_short_interest(info, "GME")
        assert "8.00%" in result
        assert "2.5" in result
        assert "15,000,000" in result

    def test_high_short_red(self):
        info = {"shortPercentOfFloat": 0.15, "shortRatio": 6.0}
        result = format_short_interest(info, "GME")
        assert "ff3232" in result  # red for high short interest


class TestFormatRatings:
    def test_no_data(self):
        result = format_ratings(None, "AAPL")
        assert "No data" in result

    def test_no_analyst_data(self):
        result = format_ratings({"regularMarketPrice": 150}, "AAPL")
        assert "No analyst data" in result

    def test_buy_rating_green(self):
        info = {
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 30,
            "targetMeanPrice": 200.0,
            "regularMarketPrice": 150.0,
        }
        result = format_ratings(info, "AAPL")
        assert "BUY" in result
        assert "30 analysts" in result
        assert "+33.3%" in result
        assert "green" in result

    def test_sell_rating_red(self):
        info = {"recommendationKey": "sell", "targetMeanPrice": 100.0, "regularMarketPrice": 150.0}
        result = format_ratings(info, "TEST")
        assert "SELL" in result
        assert "ff3232" in result

    def test_target_range(self):
        info = {
            "recommendationKey": "hold",
            "targetMeanPrice": 200.0,
            "targetLowPrice": 150.0,
            "targetHighPrice": 250.0,
            "regularMarketPrice": 180.0,
        }
        result = format_ratings(info, "AAPL")
        assert "$150.00" in result
        assert "$250.00" in result

    def test_with_recommendations(self):
        info = {"recommendationKey": "buy", "targetMeanPrice": 200.0, "regularMarketPrice": 150.0}
        recs = [
            {"firm": "Goldman Sachs", "toGrade": "Buy", "fromGrade": "Hold", "action": "upgrade"},
            {"firm": "Morgan Stanley", "toGrade": "Sell", "fromGrade": "Buy", "action": "downgrade"},
        ]
        result = format_ratings(info, "AAPL", recs)
        assert "Goldman" in result
        assert "Morgan" in result
        assert "Hold" in result and "Buy" in result
