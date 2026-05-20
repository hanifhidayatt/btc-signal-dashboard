import yfinance as yf
import pandas as pd
import os


def fetch_crypto_data(ticker="BTC-USD", period="2y", interval="1d"):
    """
    Fetch historical crypto data from Yahoo Finance.
    ticker: e.g. BTC-USD, ETH-USD, BNB-USD
    period: 1y, 2y, 5y
    interval: 1d, 1h (hourly has limits on free tier)
    """
    print(f"Fetching {ticker} data...")
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)

    # Save to data folder
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{ticker.replace('-', '_')}_{period}.csv"
    df.to_csv(filepath)
    print(f"Saved to {filepath} — {len(df)} rows")
    return df


if __name__ == "__main__":
    df = fetch_crypto_data("BTC-USD", period="10y")
    print(df.tail())
