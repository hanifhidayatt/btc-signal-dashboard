import pandas as pd
import pandas_ta as ta
import yfinance as yf


def fetch_coinbase_premium(start_date=None):
    """
    Coinbase Premium = BTC-USD (Coinbase) - BTC-USDT (Binance)
    Positive = US buyers paying premium = bullish signal
    Negative = selling pressure dominating
    """
    btc_coinbase = yf.download(
        "BTC-USD", period="10y", interval="1d", progress=False)
    btc_binance = yf.download("BTC-USDT", period="10y",
                              interval="1d", progress=False)

    # Flatten multi-level columns if present (yfinance v0.2+ quirk)
    if isinstance(btc_coinbase.columns, pd.MultiIndex):
        btc_coinbase.columns = btc_coinbase.columns.get_level_values(0)
    if isinstance(btc_binance.columns, pd.MultiIndex):
        btc_binance.columns = btc_binance.columns.get_level_values(0)

    premium = btc_coinbase['Close'] - btc_binance['Close']
    premium.name = 'Coinbase_premium'
    premium_pct = premium / btc_coinbase['Close']
    premium_pct.name = 'Coinbase_premium_pct'

    return pd.concat([premium, premium_pct], axis=1)


def add_indicators(df):
    # Flatten multi-level columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- CRITICAL FIX: Ensure all core columns are strictly numeric ---
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that failed conversion and became NaN
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # --- Trend Indicators ---
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)

    # MACD
    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd_df.iloc[:, 0] if macd_df is not None else 0

    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Bollinger Bands
    bb_df = ta.bbands(df['Close'], length=20, std=2)
    if bb_df is not None:
        df['BB_upper'] = bb_df.iloc[:, 2]
        df['BB_lower'] = bb_df.iloc[:, 0]
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
    else:
        df['BB_upper'], df['BB_lower'], df['BB_width'] = 0, 0, 0

    # ADX & ATR
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0] if adx_df is not None else 0
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_pct'] = df['ATR'] / df['Close']

    # --- Candlestick Features & Helpers ---
    df['Volume_change'] = df['Volume'].pct_change()
    df['Return_lag1'] = df['Close'].pct_change(1)
    df['Return_lag2'] = df['Close'].pct_change(2)
    df['Return_lag3'] = df['Close'].pct_change(3)
    df['Return_lag5'] = df['Close'].pct_change(5)

    df['Price_EMA20_ratio'] = df['Close'] / df['EMA_20']
    df['Price_EMA50_ratio'] = df['Close'] / df['EMA_50']
    df['EMA_cross'] = (df['EMA_20'] > df['EMA_50']).astype(int)
    df['Is_bull_market'] = (df['Close'] > df['EMA_200']).astype(int)

    df['Body_size'] = (df['Close'] - df['Open']).abs() / df['Open']
    df['Upper_wick'] = (
        df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
    df['Lower_wick'] = (df[['Open', 'Close']].min(
        axis=1) - df['Low']) / df['Open']
    df['Is_green'] = (df['Close'] > df['Open']).astype(int)

    # --- Target Label Calculation ---
    R = 0.02
    target_3r = []
    target_5r = []
    target_short_3r = []

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    for i in range(len(df)):
        if i >= len(df) - 10:
            target_3r.append(0)
            target_5r.append(0)
            target_short_3r.append(0)
            continue

        entry_price = closes[i]

        # 1. LONG TARGETS
        long_stop = entry_price * (1 - R)
        long_tp_3r = entry_price * (1 + R * 3.5)
        long_tp_5r = entry_price * (1 + R * 5.0)

        hit_3r = 0
        hit_5r = 0

        for j in range(i + 1, min(i + 11, len(df))):
            high = highs[j]
            low = lows[j]

            if low <= long_stop:
                break
            if high >= long_tp_5r:
                hit_5r = 1
                hit_3r = 1
                break
            if high >= long_tp_3r:
                hit_3r = 1

        target_3r.append(hit_3r)
        target_5r.append(hit_5r)

        # 2. SHORT TARGETS
        short_stop = entry_price * (1 + R)
        short_tp_3r = entry_price * (1 - R * 3.5)

        hit_short_3r = 0

        for j in range(i + 1, min(i + 11, len(df))):
            high = highs[j]
            low = lows[j]

            if high >= short_stop:
                break
            if low <= short_tp_3r:
                hit_short_3r = 1
                break

        target_short_3r.append(hit_short_3r)

    df['Target_3R'] = target_3r
    df['Target_5R'] = target_5r
    df['Target_Short_3R'] = target_short_3r

    # --- Coinbase Premium Feature Merge ---
    premium_df = fetch_coinbase_premium()
    premium_df.index = pd.to_datetime(premium_df.index)
    if isinstance(premium_df.index, pd.DatetimeIndex) and premium_df.index.tz is not None:
        premium_df.index = premium_df.index.tz_localize(None)

    df.index = pd.to_datetime(df.index)
    df = df.join(premium_df, how='left')

    df['Coinbase_premium'] = df['Coinbase_premium'].fillna(0)
    df['Coinbase_premium_pct'] = df['Coinbase_premium_pct'].fillna(0)

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    try:
        # Explicitly ignore header text conflicts during load
        raw_df = pd.read_csv("data/BTC_USD_10y.csv", index_col=0)
        processed_df = add_indicators(raw_df)
        processed_df.to_csv("data/BTC_USD_features.csv")
        print(f"📊 Features built successfully! Shape: {processed_df.shape}")
        print("Generated targets check:")
        print(
            processed_df[['Target_3R', 'Target_5R', 'Target_Short_3R']].sum())
    except Exception as e:
        print(f"Error building features: {e}")
