import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

from api.utils import load_css, initialize_session_state
from api.stock_data import analyze
from api.news_sentiment import get_news_sentiment


def setup_ui():
    st.set_page_config(
        page_title="FortuneTeller",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_css()
    initialize_session_state()

    st.title("FortuneTeller")
    st.markdown("The Wealth Whisperer ✨")
    st.sidebar.title("The Profit Prophet")


def stock_search():
    input_col1, input_col2, input_col3 = st.columns([2, 1, 1])

    with input_col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL").upper()

        if symbol:
            is_favorite = symbol in st.session_state.favorites
            button_label = ("❌ Remove from Favorites" if is_favorite else "❤️ Add to Favorites")

        if st.button(button_label, key=f"fav_{symbol}", type="primary"):
            if is_favorite:
                st.session_state.favorites.remove(symbol)
            else:
                st.session_state.favorites.append(symbol)

    with input_col2:
        valid_timeframes = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        timeframe = st.selectbox("Select Timeframe", valid_timeframes, index=valid_timeframes.index("3mo"))

    with input_col3:
        valid_intervals = {
            "1d": ["1m", "2m", "5m", "15m", "30m", "1h"],
            "5d": ["1m", "2m", "5m", "15m", "30m", "1h", "1d"],
            "1mo": ["2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk"],
            "3mo": ["1h", "1d", "5d", "1wk", "1mo"],
            "6mo": ["1h", "1d", "5d", "1wk", "1mo", "3mo"],
            "1y": ["1h", "1d", "5d", "1wk", "1mo", "3mo"],
            "2y": ["1h", "1d", "5d", "1wk", "1mo", "3mo"],
            "5y": ["1d", "5d", "1wk", "1mo", "3mo"],
            "10y": ["1d", "5d", "1wk", "1mo", "3mo"],
            "ytd": ["1d", "5d", "1wk", "1mo", "3mo"],
            "max": ["1d", "5d", "1wk", "1mo", "3mo"],
        }

        interval = st.selectbox("Select Interval", valid_intervals[timeframe], index=valid_intervals["3mo"].index("1d"))
    
    return symbol, timeframe, interval


def main_chart(df, levels):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC")])
    #Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200", line=dict(color="red", width=1)))
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(color="gray", dash="dash")))
    #Support and resistance levels
    for level in levels["support"]:
        fig.add_hline(y=level, line_color="green", line_dash="dash", annotation_text="Support")
    for level in levels["resistance"]:
        fig.add_hline(y=level, line_color="red", line_dash="dash", annotation_text="Resistance")
    fig.update_layout(title="Stock Price and Technical Analysis", yaxis_title="Price", template="plotly_dark", height=600, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def tech_chart(df, patterns):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Moving Averages", "RSI & Stochastic", "MACD", "Bollinger Bands", "Patterns"])

    with tab1:
        st.line_chart(df[["MA20", "MA50", "MA200"]])
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df["RSI"])
        with col2:
            st.line_chart(df[["%K", "%D"]])
    with tab3:
        st.line_chart(df[["MACD", "MACD_signal"]])
        st.bar_chart(df["MACD_hist"])
    with tab4:
        st.line_chart(df[["BB_upper", "BB_middle", "BB_lower"]])
    with tab5:
        st.subheader("Detected Patterns")
        for pattern, indices in patterns.items():
            if indices:
                dates = df.index[indices].strftime("%Y-%m-%d").tolist()
                st.write(f"**{pattern}** pattern detected on: {", ".join(dates[-5:])}")


def news_sentiment(symbol):
    st.markdown("---")
    st.header("News Sentiment Analysis")
    with st.spinner("Analyzing news sentiment..."):
        news_data = get_news_sentiment(symbol)
        if news_data and news_data["news_items"]:
            summary = news_data["sentiment_summary"]

            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_value = summary["average_sentiment"]
                st.metric("Overall Sentiment", f"{sentiment_value:.2f}", delta=("Positive" if sentiment_value > 0 else "Negative"))
            with col2:
                st.metric("Total Articles", summary["total_articles"])
            with col3:
                st.metric("Sentiment Trend", summary["sentiment_trend"])

            st.subheader("Sentiment Distribution")
            dist_data = pd.DataFrame([{"Sentiment": k, "Count": v} for k, v in summary["sentiment_distribution"].items() if v > 0])

            if not dist_data.empty:
                st.bar_chart(dist_data.set_index("Sentiment"))

            st.subheader("Latest News Articles")
            for article in news_data["news_items"]:
                with st.expander(article["title"]):
                    st.write(f"**Source:** {article["source"]}")
                    st.write(f"**Date:** {article["date"]}")

                    scores = article["sentiment_scores"]
                    score_data = pd.DataFrame([{"Type": "Positive", "Score": scores["positive"]},
                                               {"Type": "Neutral", "Score": scores["neutral"]},
                                               {"Type": "Negative", "Score": scores["negative"]}])

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Sentiment Breakdown:**")
                        st.bar_chart(score_data.set_index("Type"))
                    with col2:
                        sentiment_color = {"Very Positive": "green", "Positive": "lightgreen", "Neutral": "gray", "Negative": "pink", "Very Negative": "red"}
                        st.markdown(f"**Overall Sentiment:**\n\n"
                                    f"<span style='color:{sentiment_color[article['sentiment']]};'>"
                                    f"{article['sentiment']}</span> "
                                    f"(Score: {scores['compound']:.2f})",
                                    unsafe_allow_html=True
                                )

                    st.write("**Summary:**")
                    st.write(article["summary"])
            else:
                st.warning(f"No articles found for {symbol}")


def analyze_page(symbol, timeframe, interval):
    if st.button("Analyze Stock", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            df, patterns, levels = analyze(symbol, timeframe, interval)

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            stock = yf.Ticker(symbol)
            info = stock.info

            with info_col1:
                st.metric("Current Price", f"${info.get("currentPrice", "N/A")}")
            with info_col2:
                st.metric("Market Cap", f"${info.get("marketCap", "N/A"):,}")
            with info_col3:
                st.metric("PE ratio", f"${info.get("trailingPE", "N/A"):,}")
            with info_col4:
                st.metric("Dividend Yield", f"${info.get("dividendYield", "N/A"):,}")
                
            main_chart(df, levels)
            tech_chart(df, patterns)
            news_sentiment(symbol)


def favorites_page():
    st.title("Favorite Stocks")

    if not st.session_state.favorites:
        st.info("No favorite stocks added yet")
    else:
        cols = st.columns(3)
        for idx, symbol in enumerate(st.session_state.favorites):
            with cols[idx % 3]:
                st.write(f"### {symbol}")
                stock = yf.Ticker(symbol)
                info = stock.info
                st.metric("Price", f"${info.get("currentPrice", "N/A")}")
                if st.button("Remove", key=f"remove_{symbol}", type="primary"):
                    st.session_state.favorites.remove(symbol)

    