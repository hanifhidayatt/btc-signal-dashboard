import pandas as pd
import pandas_ta as ta
import yfinance as yf


def add_indicators(df):
    # Flatten multi-level columns if needed (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- Trend Indicators ---
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']

    # --- Market Regime Filter ---
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['Is_bull_market'] = (df['Close'] > df['EMA_200']).astype(int)

    # --- Trend Strength ---
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']

    # --- Momentum ---
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # --- Volatility & ATR ---
    bbands = ta.bbands(df['Close'], length=20)
    bb_upper_col = [c for c in bbands.columns if 'BBU' in c][0]
    bb_lower_col = [c for c in bbands.columns if 'BBL' in c][0]
    df['BB_upper'] = bbands[bb_upper_col]
    df['BB_lower'] = bbands[bb_lower_col]
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    # Calculate ATR (which was missing but required by models)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_pct'] = df['ATR'] / df['Close']

    # --- Volume ---
    df['Volume_change'] = df['Volume'].pct_change()

    # --- Lag features ---
    for lag in [1, 2, 3, 5]:
        df[f'Return_lag{lag}'] = df['Close'].pct_change(lag)

    # --- Price vs moving average ratios ---
    df['Price_EMA20_ratio'] = df['Close'] / df['EMA_20']
    df['Price_EMA50_ratio'] = df['Close'] / df['EMA_50']
    df['EMA_cross'] = df['EMA_20'] - df['EMA_50']

    # --- Candle features ---
    df['Body_size'] = abs(df['Close'] - df['Open'])
    df['Upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Is_green'] = (df['Close'] > df['Open']).astype(int)

    # --- Forward looking R-multiple targets ---
    R = 0.02  # 2% stop loss

    target_3r = []
    target_5r = []

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    for i in range(len(df)):
        entry = closes[i]
        stop = entry * (1 - R)
        tp_3r = entry * (1 + R * 3.5)
        tp_5r = entry * (1 + R * 5.0)

        hit_3r = 0
        hit_5r = 0

        for j in range(i + 1, min(i + 11, len(df))):
            high = highs[j]
            low = lows[j]

            if low <= stop:
                break

            if high >= tp_5r:
                hit_5r = 1
                hit_3r = 1
                break

            if high >= tp_3r:
                hit_3r = 1

        target_3r.append(hit_3r)
        target_5r.append(hit_5r)

    df['Target_3R'] = target_3r
    df['Target_5R'] = target_5r

    # Clean up dates and drop missing rows caused by indicator lookbacks
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    # Load raw 10y data and parse columns correctly
    df = pd.read_csv("data/BTC_USD_10y.csv", header=[0, 1], index_col=0)
    df = add_indicators(df)

    print(df[['Close', 'RSI', 'MACD', 'EMA_20',
          'ATR', 'Target_3R', 'Target_5R']].tail())
    df.to_csv("data/BTC_USD_features.csv")
    print(f"\nSaved features — {len(df)} rows, {len(df.columns)} columns")
    print(f"3.5R trades available: {df['Target_3R'].sum()}")
    print(f"5R trades available:   {df['Target_5R'].sum()}")
