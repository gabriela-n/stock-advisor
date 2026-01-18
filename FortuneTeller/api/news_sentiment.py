from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from datetime import datetime, timedelta


try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    return {
        "compound": sentiment["compound"],
        "positive": sentiment["pos"],
        "negative": sentiment["neg"],
        "neutral": sentiment["neu"]
    }

def get_sentiment_label(compound_score):
    if compound_score >= 0.5:
        return "Very Positive"
    elif compound_score >= 0.1:
        return "Positive"
    elif compound_score <= -0.5:
        return "Very Negative"
    elif compound_score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

def validate_news_article(article):
    required_fields = ["title", "description", "source", "publishedAt"]
    return all(field in article and article[field] is not None for field in required_fields)

def get_news_sentiment(symbol):
    newsapi = NewsApiClient(api_key="e2ebff5042cf4c6e9d5af4e6e4e6a26b")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    news = newsapi.get_everything(q=f"'{symbol}' AND (stock OR market OR trading OR finance)", language="en", sort_by="publishedAt", 
                                  from_param=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))

    if not news or news["totalResults"] == 0:
        return {"news_items": [], "sentiment_summary": {"average_sentiment": 0, "sentiment_distribution": {"Very Positive": 0, "Positive": 0,
                                                                                                           "Neutral": 0, "Negative": 0, "Very Negative": 0},
                                                        "total_articles": 0, "sentiment_trend": "Neutral"}}

    news_data = []
    sentiment_scores = {"Very Positive": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Very Negative": 0}
    total_sentiment = 0
    valid_articles = 0

    for article in news["articles"]:
        if not validate_news_article(article):
            continue

        try:
            text = f"{article['title']} {article['description']}"
            sentiment = get_sentiment_score(text)
            label = get_sentiment_label(sentiment["compound"])

            sentiment_scores[label] += 1
            total_sentiment += sentiment["compound"]
            valid_articles += 1

            news_item = {
                "title": article["title"],
                "summary": article["description"],
                "source": article["source"]["name"],
                "date": pd.Timestamp(article["publishedAt"]).strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": label,
                "sentiment_scores": sentiment
            }
            news_data.append(news_item)

        except Exception as e:
            continue

    if valid_articles == 0:
        raise ValueError("No valid news articles found")

    sentiment_summary = {
        "average_sentiment": total_sentiment / valid_articles,
        "sentiment_distribution": sentiment_scores,
        "total_articles": valid_articles,
        "sentiment_trend": get_sentiment_label(total_sentiment / valid_articles)
    }

    return {
        "news_items": news_data,
        "sentiment_summary": sentiment_summary
    }
