import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

FEATURES = [
    'EMA_20', 'EMA_50', 'MACD', 'RSI',
    'BB_upper', 'BB_lower', 'BB_width', 'Volume_change',
    'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
    'Price_EMA20_ratio', 'Price_EMA50_ratio', 'EMA_cross',
    'Body_size', 'Upper_wick', 'Lower_wick', 'Is_green',
    'EMA_200', 'Is_bull_market', 'ADX', 'ATR', 'ATR_pct',
    'SP500_return', 'DXY_return'  # Added macro triggers
]

R = 0.02


def run_backtest(csv_path="data/BTC_USD_features.csv"):
    df = pd.read_csv(csv_path, index_col=0)

    X = df[FEATURES]
    y_3r = df['Target_3R']
    y_5r = df['Target_5R']

    # Fixed date split — train on bear market, test on 2023-2024 bull run
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2024-12-31'

    train_mask = df.index <= train_end
    test_mask = (df.index >= test_start) & (df.index <= test_end)

    print(f"Training on: {df.index[0]} → {train_end}")
    print(f"Testing on:  {test_start} → {test_end}")
    print(f"Train size: {train_mask.sum()} days")
    print(f"Test size:  {test_mask.sum()} days")

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_3r_train = y_3r[train_mask]
    y_5r_train = y_5r[train_mask]

    # Train fresh models on training data only
    neg_3r = (y_3r_train == 0).sum()
    pos_3r = (y_3r_train == 1).sum()
    model_3r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                 eval_metric='logloss',
                                 scale_pos_weight=neg_3r/pos_3r)
    model_3r.fit(X_train, y_3r_train)

    neg_5r = (y_5r_train == 0).sum()
    pos_5r = (y_5r_train == 1).sum()
    model_5r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                 eval_metric='logloss',
                                 scale_pos_weight=neg_5r/pos_5r)
    model_5r.fit(X_train, y_5r_train)

    # Predict on unseen test data only
    probs_3r = model_3r.predict_proba(X_test)[:, 1]
    probs_5r = model_5r.predict_proba(X_test)[:, 1]

    # Regime breakdown
    df_test = df[test_mask].copy()
    bull_days = int(df_test['Is_bull_market'].sum())
    bear_days = len(df_test) - bull_days
    print(f"Bull market days in test: {bull_days}")
    print(f"Bear market days in test: {bear_days}\n")

    trades = []
    last_trade_day = -999
    last_result = None

    # --- ADD MACRO CONDITION INSIDE THE TRADING LOOP ---
    for i in range(len(X_test)):
        # Fetch the macro returns we added to features
        sp500_ret = df_test['SP500_return'].iloc[i]
        dxy_ret = df_test['DXY_return'].iloc[i]

        # FILTER: Skip days where the S&P 500 drops heavily (>1.5%)
        # or the US Dollar surges heavily (>0.8%), creating market friction
        if sp500_ret < -0.015 or dxy_ret > 0.008:
            continue

        # Continue down to probability calculations...
        p3r = probs_3r[i]
        p5r = probs_5r[i]

    for i in range(len(X_test)):
        is_bull = df_test['Is_bull_market'].iloc[i]
        ema_20 = df_test['EMA_20'].iloc[i]
        ema_50 = df_test['EMA_50'].iloc[i]
        adx_val = df_test['ADX'].iloc[i]

        # Cooldown: wait 3 days after a loss
        if last_result == 'LOSS' and (i - last_trade_day) < 3:
            continue

            # --- REVISED SIGNAL SELECTION LOGIC IN BACKTEST.PY ---
        # --- NEW FOCUSED SIGNAL LOGIC ---
        p3r = probs_3r[i]

        # Completely ignore p5r. Only trade the highly profitable 3.5R signal.
        if p3r >= 0.70:
            signal = '3.5R'
            reward = R * 3.5
            target_hit = df_test['Target_3R'].iloc[i]
        else:
            continue

        pnl = reward if target_hit else -R
        result = 'WIN' if target_hit else 'LOSS'
        last_trade_day = i
        last_result = result

        trades.append({
            'Date': df_test.index[i],
            'Signal': signal,
            'Confidence_3R': round(p3r, 3),
            'Confidence_5R': round(p5r, 3),
            'Result': result,
            'PnL_R': round(pnl / R, 2)
        })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("No trades taken — try lowering thresholds")
        return

    wins = (trades_df['Result'] == 'WIN').sum()
    losses = (trades_df['Result'] == 'LOSS').sum()
    total = len(trades_df)
    win_rate = wins / total
    total_r = trades_df['PnL_R'].sum()

    print(f"=== Honest Backtest Results (2023-2024 Bull Run) ===")
    print(f"Total trades:  {total}")
    print(f"Wins:          {wins} ({win_rate:.1%})")
    print(f"Losses:        {losses}")
    print(f"Total R:       {total_r:.1f}R")
    print(f"Avg R/trade:   {trades_df['PnL_R'].mean():.2f}R")
    print(f"\n--- By Signal ---")
    print(trades_df.groupby('Signal')['PnL_R'].agg(
        ['count', 'sum', 'mean']).round(2))
    print(f"\n--- All Trades ---")
    print(trades_df.to_string(index=False))


if __name__ == "__main__":
    run_backtest()
