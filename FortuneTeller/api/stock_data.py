import yfinance as yf
from .technical_analysis import calculate_indicators, find_patterns, find_support_resistance


def get_df(symbol, timeframe, interval):
    stock = yf.Ticker(symbol)
    df = stock.history(period=timeframe, interval=interval)

    if df.empty:
        return None
    return df


def analyze(symbol, timeframe, interval):
    df = get_df(symbol, timeframe, interval)
    if df is not None and not df.empty:
        df = calculate_indicators(df)
        patterns = find_patterns(df)
        levels = find_support_resistance(df)

    return df, patterns, levels