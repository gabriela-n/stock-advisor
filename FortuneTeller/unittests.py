import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from api.technical_analysis import calculate_indicators, find_patterns, find_support_resistance
from api.news_sentiment import get_sentiment_score, get_sentiment_label, validate_news_article, get_news_sentiment
from api.stock_data import get_df

class TestTechnicalAnalysis(unittest.TestCase):
    def setUp(self):
        data = {"Open": [225.02000427246094, 226.39999389648438, 225.25, 226.97999572753906, 228.05999755859375, 228.8800048828125, 228.05999755859375, 231.4600067138672, 233.3300018310547, 234.47000122070312, 234.80999755859375, 237.27000427246094, 239.80999755859375, 242.8699951171875, 243.99000549316406, 242.91000366210938, 241.8300018310547, 246.88999938964844, 247.9600067138672, 246.88999938964844, 247.82000732421875, 247.99000549316406, 250.0800018310547, 252.16000366210938, 247.5, 248.0399932861328, 254.77000427246094, 255.49000549316406, 258.19000244140625, 257.8299865722656, 252.22999572753906, 252.44000244140625, 248.92999267578125, 243.36000061035156, 244.30999755859375, 242.97999572753906, 241.9199981689453, 240.00999450683594, 233.52999877929688, 234.75, 234.63999938964844, 237.35000610351562, 232.1199951171875, 224.0, 219.7899932861328, 224.74000549316406, 224.77999877929688, 224.02000427246094, 230.85000610351562, 234.1199951171875, 238.6699981689453, 247.19000244140625, 229.99000549316406, 227.25, 228.52999877929688, 231.2899932861328, 232.60000610351562, 229.57000732421875, 228.1999969482422, 231.1999969482422, 236.91000366210938, 241.07000732421875],
                "High": [228.8699951171875, 226.9199981689453, 229.74000549316406, 230.16000366210938, 229.92999267578125, 230.16000366210938, 230.72000122070312, 233.25, 235.57000732421875, 235.69000244140625, 237.80999755859375, 240.7899932861328, 242.75999450683594, 244.11000061035156, 244.5399932861328, 244.6300048828125, 247.24000549316406, 248.2100067138672, 250.8000030517578, 248.74000549316406, 249.2899932861328, 251.3800048828125, 253.8300018310547, 254.27999877929688, 252.0, 255.0, 255.64999389648438, 258.2099914550781, 260.1000061035156, 258.70001220703125, 253.5, 253.27999877929688, 249.10000610351562, 244.17999267578125, 247.3300018310547, 245.5500030517578, 243.7100067138672, 240.16000366210938, 234.6699981689453, 236.1199951171875, 238.9600067138672, 238.00999450683594, 232.2899932861328, 224.4199981689453, 224.1199951171875, 227.02999877929688, 225.6300048828125, 232.14999389648438, 240.19000244140625, 239.86000061035156, 240.7899932861328, 247.19000244140625, 231.8300018310547, 233.1300048828125, 232.6699981689453, 233.8000030517578, 234.0, 230.58999633789062, 235.22999572753906, 236.9600067138672, 242.33999633789062, 245.0500030517578],
                "Low": [225.0, 224.27000427246094, 225.1699981689453, 226.66000366210938, 225.88999938964844, 225.7100067138672, 228.05999755859375, 229.74000549316406, 233.3300018310547, 233.80999755859375, 233.97000122070312, 237.16000366210938, 238.89999389648438, 241.25, 242.1300048828125, 242.0800018310547, 241.75, 245.33999633789062, 246.25999450683594, 245.67999267578125, 246.24000549316406, 247.64999389648438, 249.77999877929688, 247.74000549316406, 247.08999633789062, 245.69000244140625, 253.4499969482422, 255.2899932861328, 257.6300048828125, 253.05999755859375, 250.75, 249.42999267578125, 241.82000732421875, 241.88999938964844, 243.1999969482422, 241.35000610351562, 240.0500030517578, 233.0, 229.72000122070312, 232.47000122070312, 234.42999267578125, 228.02999877929688, 228.47999572753906, 219.3800048828125, 219.7899932861328, 222.3000030517578, 221.41000366210938, 223.97999572753906, 230.80999755859375, 234.00999450683594, 237.2100067138672, 233.44000244140625, 225.6999969482422, 226.64999389648438, 228.27000427246094, 230.42999267578125, 227.25999450683594, 227.1999969482422, 228.1300048828125, 230.67999267578125, 235.57000732421875, 241.0],
                "Close": [228.22000122070312, 225.0, 228.02000427246094, 228.27999877929688, 229.0, 228.52000427246094, 229.8699951171875, 232.8699951171875, 235.05999755859375, 234.92999267578125, 237.3300018310547, 239.58999633789062, 242.64999389648438, 243.00999450683594, 243.0399932861328, 242.83999633789062, 246.75, 247.77000427246094, 246.49000549316406, 247.9600067138672, 248.1300048828125, 251.0399932861328, 253.47999572753906, 248.0500030517578, 249.7899932861328, 254.49000549316406, 255.27000427246094, 258.20001220703125, 259.0199890136719, 255.58999633789062, 252.1999969482422, 250.4199981689453, 243.85000610351562, 243.36000061035156, 245.0, 242.2100067138672, 242.6999969482422, 236.85000610351562, 234.39999389648438, 233.27999877929688, 237.8699951171875, 228.25999450683594, 229.97999572753906, 222.63999938964844, 223.8300018310547, 223.66000366210938, 222.77999877929688, 229.86000061035156, 238.25999450683594, 239.36000061035156, 237.58999633789062, 236.0, 228.00999450683594, 232.8000030517578, 232.47000122070312, 233.22000122070312, 227.6300048828125, 227.64999389648438, 232.6199951171875, 236.8699951171875, 241.52999877929688, 244.69000244140625],
                "Volume": [44923900, 47923700, 44686000, 36211800, 35169600, 42108300, 38168300, 90152800, 45986200, 33498400, 28481400, 48137100, 38861000, 44383900, 40033900, 36870600, 44649200, 36914800, 45205800, 32777500, 33155300, 51694800, 51356400, 56774100, 60882300, 147495300, 40858800, 23234700, 27237100, 42355300, 35557500, 39480700, 55740700, 40244100, 45045600, 40856000, 37628900, 61710900, 49630700, 39435300, 39832000, 71759100, 68488300, 98070400, 64126500, 60234800, 54697900, 94863400, 75707600, 45486100, 55658300, 101075100, 73063300, 45067300, 39620300, 29925300, 39707200, 33115600, 53718400, 45243300, 53543300, 9636717]
                }
        date = pd.to_datetime(["2024-11-14T00:00:00", "2024-11-15T00:00:00", "2024-11-18T00:00:00", "2024-11-19T00:00:00", "2024-11-20T00:00:00", "2024-11-21T00:00:00", "2024-11-22T00:00:00", "2024-11-25T00:00:00", "2024-11-26T00:00:00", "2024-11-27T00:00:00", "2024-11-29T00:00:00", "2024-12-02T00:00:00", "2024-12-03T00:00:00", "2024-12-04T00:00:00", "2024-12-05T00:00:00", "2024-12-06T00:00:00", "2024-12-09T00:00:00", "2024-12-10T00:00:00", "2024-12-11T00:00:00", "2024-12-12T00:00:00", "2024-12-13T00:00:00", "2024-12-16T00:00:00", "2024-12-17T00:00:00", "2024-12-18T00:00:00", "2024-12-19T00:00:00", "2024-12-20T00:00:00", "2024-12-23T00:00:00", "2024-12-24T00:00:00", "2024-12-26T00:00:00", "2024-12-27T00:00:00", "2024-12-30T00:00:00", "2024-12-31T00:00:00", "2025-01-02T00:00:00", "2025-01-03T00:00:00", "2025-01-06T00:00:00", "2025-01-07T00:00:00", "2025-01-08T00:00:00", "2025-01-10T00:00:00", "2025-01-13T00:00:00", "2025-01-14T00:00:00", "2025-01-15T00:00:00", "2025-01-16T00:00:00", "2025-01-17T00:00:00", "2025-01-21T00:00:00", "2025-01-22T00:00:00", "2025-01-23T00:00:00", "2025-01-24T00:00:00", "2025-01-27T00:00:00", "2025-01-28T00:00:00", "2025-01-29T00:00:00", "2025-01-30T00:00:00", "2025-01-31T00:00:00", "2025-02-03T00:00:00", "2025-02-04T00:00:00", "2025-02-05T00:00:00", "2025-02-06T00:00:00", "2025-02-07T00:00:00", "2025-02-10T00:00:00", "2025-02-11T00:00:00", "2025-02-12T00:00:00", "2025-02-13T00:00:00", "2025-02-14T00:00:00"])
        self.df = pd.DataFrame(data, index=date)

    def test_calsulate_indicators_moving_average(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("MA20", result.columns)
        self.assertIn("MA50", result.columns)
        self.assertIn("MA200", result.columns)

        self.assertTrue(result["MA20"].iloc[:19].isna().all())
        self.assertTrue(result["MA50"].iloc[:49].isna().all())

    def test_calculate_indicators_rsi(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("RSI", result.columns)
        
        self.assertTrue(np.isnan(result["RSI"].iloc[:13]).all())  #NaN for the first 13 values

    def test_calculate_indicators_macd(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("MACD", result.columns)
        self.assertIn("MACD_signal", result.columns)
        self.assertIn("MACD_hist", result.columns)

    def test_calculate_indicators_bollinger(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("BB_upper", result.columns)
        self.assertIn("BB_middle", result.columns)
        self.assertIn("BB_lower", result.columns)

        self.assertTrue(result["BB_upper"].iloc[:19].isna().all())

    def test_calculate_indicators_stochastic(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("%K", result.columns)
        self.assertIn("%D", result.columns)
        
        self.assertTrue(result["%K"].iloc[:13].isna().all())
        self.assertTrue(result["%D"].iloc[:15].isna().all())
    
    def test_calculate_indicators_atr(self):
        result = calculate_indicators(self.df.copy())

        self.assertIn("ATR", result.columns)
        
        self.assertTrue(result["ATR"].iloc[:13].isna().all())

    @patch("api.technical_analysis.find_patterns")
    def test_find_patterns(self, mock_find_patterns):
        mock_find_patterns.return_value = ["Doji"]
        result = find_patterns(self.df)
        self.assertIn("Doji", result)
        # still figuring out how to test these

    def test_find_support_resistance(self):
        levels = find_support_resistance(self.df)

        self.assertIn(max(self.df["Close"]), levels["resistance"]) 
    

class TestNewsSentiment(unittest.TestCase):
    def test_sentiment_score_positive(self):
        text = "The AAPL stock has potential for growth."
        sentiment = get_sentiment_score(text)

        self.assertIn("compound", sentiment)
        self.assertIn("positive", sentiment)
        self.assertIn("negative", sentiment)
        self.assertIn("neutral", sentiment)

        self.assertGreaterEqual(sentiment["positive"], 0)
        self.assertGreaterEqual(sentiment["compound"], 0)

    def test_sentiment_score_negative(self):
        text = "AAPL CEO is going to jail."
        sentiment = get_sentiment_score(text)

        self.assertIn("compound", sentiment)
        self.assertIn("positive", sentiment)
        self.assertIn("negative", sentiment)
        self.assertIn("neutral", sentiment)

        self.assertLessEqual(sentiment["negative"], 0)
        self.assertLessEqual(sentiment["compound"], 0)

    def test_sentiment_label(self):
        self.assertEqual(get_sentiment_label(0.6), "Very Positive")
        self.assertEqual(get_sentiment_label(0.3), "Positive")
        self.assertEqual(get_sentiment_label(-0.3), "Negative")
        self.assertEqual(get_sentiment_label(-0.6), "Very Negative")
        self.assertEqual(get_sentiment_label(0.0), "Neutral")

    def test_valid_news_article(self):
        valid_article = {"title": "SHOCK", "description": "Shocking news.", "source": "reliable", "publishedAt": "2025-02-15T12:30:00Z"}
        self.assertTrue(validate_news_article(valid_article))

    def test_invalid_news_article(self):
        invalid_article = {"title": "SHOCK", "description": "Shocking news.", "source": "reliable"}
        self.assertFalse(validate_news_article(invalid_article))

    """
    @patch("api.news_sentiment.NewsApiClient")
    @patch("api.news_sentiment.validate_news_article", return_value=True)
    @patch("api.news_sentiment.get_sentiment_score", return_value={"compound": 0.5})
    @patch("api.news_sentiment.get_sentiment_label", return_value="Positive")
    def test_get_news_sentiment(self, mock_news_api, mock_validate, mock_s_score, mock_s_label):
        mock_news_api.get_everything.return_value = {"totalResults": 2, "articles": [
            {"title": "Stock rises", "description": "Company reports growth", "source": {"name": "News Source"}, "publishedAt": "2025-02-15T12:00:00Z"},
            {"title": "Market update", "description": "Index sees gains", "source": {"name": "Another Source"}, "publishedAt": "2025-02-14T10:00:00Z"}]}

        result = get_news_sentiment("")
        
        self.assertEqual(result["sentiment_summary"]["total_articles"], 2)
        self.assertEqual(result["sentiment_summary"]["sentiment_trend"], "Positive")
        self.assertEqual(result["sentiment_summary"]["sentiment_distribution"]["Positive"], 2)
    """
  

if __name__ == "__main__":
    unittest.main()