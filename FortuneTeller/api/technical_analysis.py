import pandas as pd
import numpy as np


def calculate_indicators(df: pd.DataFrame):
    # Basic Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    # Bollinger Bands
    middle_band = df["Close"].rolling(window=20).mean()
    std_dev = df["Close"].rolling(window=20).std()
    df["BB_upper"] = middle_band + (std_dev * 2)
    df["BB_middle"] = middle_band
    df["BB_lower"] = middle_band - (std_dev * 2)
    # Stochastic Oscillator
    period = 14
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    df["%K"] = ((df["Close"] - low_min) / (high_max - low_min)) * 100
    df["%D"] = df["%K"].rolling(window=3).mean()
    # Average True Range
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    
    return df


def find_patterns(df: pd.DataFrame):
    patterns = {"Doji": [], "Hammer": [], "Shooting Star": [], "Engulfing Bullish": [], "Engulfing Bearish": []}
    df["Body"] = df["Close"] - df["Open"]
    df["Upper_Shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower_Shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    for i in range(1, len(df)):
        # Doji
        body_size = abs(df["Body"].iloc[i])
        avg_shadow = (df["Upper_Shadow"].iloc[i] + df["Lower_Shadow"].iloc[i]) / 2
        if body_size < 0.1 * avg_shadow:
            patterns["Doji"].append(i)
        # Hammer
        if (df["Lower_Shadow"].iloc[i] > 2 * body_size and 
            df["Upper_Shadow"].iloc[i] < 0.1 * df["Lower_Shadow"].iloc[i]):
            patterns["Hammer"].append(i)
        # Shooting star
        if (df["Upper_Shadow"].iloc[i] > 2 * body_size and 
            df["Lower_Shadow"].iloc[i] < 0.1 * df["Upper_Shadow"].iloc[i]):
            patterns["Shooting Star"].append(i)
        # Engulfing
        if i > 0:
            prev_body = df["Body"].iloc[i-1]
            curr_body = df["Body"].iloc[i]
            # Bullish
            if (prev_body < 0 and curr_body > 0 and 
                abs(curr_body) > abs(prev_body)):
                patterns["Engulfing Bullish"].append(i)
            # Bearish
            if (prev_body > 0 and curr_body < 0 and 
                abs(curr_body) > abs(prev_body)):
                patterns["Engulfing Bearish"].append(i)

    return patterns

def find_support_resistance(df: pd.DataFrame, window=20, threshold=0.02):
    levels = {"support": [], "resistance": []}

    for i in range(window, len(df) - window):
        window_prices = df["Close"].iloc[i-window:i+window]
        current_price = df["Close"].iloc[i]

        if all(current_price <= price for price in window_prices):
            if not any(abs(level - current_price) / current_price < threshold 
                      for level in levels["support"]):
                levels["support"].append(current_price)

        if all(current_price >= price for price in window_prices):
            if not any(abs(level - current_price) / current_price < threshold 
                      for level in levels["resistance"]):
                levels["resistance"].append(current_price)

    return levels
