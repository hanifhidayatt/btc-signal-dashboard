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
    y_short_3r = df['Target_Short_3R']

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
    y_short_3r_train = y_short_3r[train_mask]

    # Train fresh models on training data only
    neg_3r = (y_3r_train == 0).sum()
    pos_3r = (y_3r_train == 1).sum()
    model_3r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                 eval_metric='logloss',
                                 scale_pos_weight=neg_3r/pos_3r)
    model_3r.fit(X_train, y_3r_train)

    # 🟢 NEW: Train Short Model
    neg_short = (y_short_3r_train == 0).sum()
    pos_short = (y_short_3r_train == 1).sum()
    model_short_3r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                       eval_metric='logloss',
                                       scale_pos_weight=neg_short/pos_short)
    model_short_3r.fit(X_train, y_short_3r_train)

    # Predict on unseen test data only
    probs_3r = model_3r.predict_proba(X_test)[:, 1]
    probs_short_3r = model_short_3r.predict_proba(X_test)[:, 1]

    # Regime breakdown
    df_test = df[test_mask].copy()
    bull_days = int(df_test['Is_bull_market'].sum())
    bear_days = len(df_test) - bull_days
    print(f"Bull market days in test: {bull_days}")
    print(f"Bear market days in test: {bear_days}\n")

    trades = []
    last_trade_day = -999
    last_result = None

    # === UNIFIED SIMULATION LOOP ===
    for i in range(len(X_test)):

        # 1. Macro Filter
        sp500_ret = df_test['SP500_return'].iloc[i]
        dxy_ret = df_test['DXY_return'].iloc[i]

        # Skip days with massive equity sell-offs or a spiking US Dollar index
        if sp500_ret < -0.015 or dxy_ret > 0.008:
            continue

        # 2. Cooldown Filter: wait 3 days after a loss
        if last_result == 'LOSS' and (i - last_trade_day) < 3:
            continue

        # 3. Model Probabilities
        p3r = probs_3r[i]
        pshort3r = probs_short_3r[i]

        # 4. Long vs Short Signal Logic
        if p3r >= 0.70 and p3r > pshort3r:
            signal = 'LONG_3.5R'
            reward = R * 3.5
            target_hit = df_test['Target_3R'].iloc[i]
        elif pshort3r >= 0.70 and pshort3r > p3r:
            signal = 'SHORT_3.5R'
            reward = R * 3.5
            target_hit = df_test['Target_Short_3R'].iloc[i]
        else:
            continue

        # 5. Calculate PnL and record trade
        pnl = reward if target_hit else -R
        result = 'WIN' if target_hit else 'LOSS'
        last_trade_day = i
        last_result = result

        # 🟢 FIX: Log the Short 3R confidence instead of the old 5R
        trades.append({
            'Date': df_test.index[i],
            'Signal': signal,
            'Confidence_Long_3R': round(p3r, 3),
            'Confidence_Short_3R': round(pshort3r, 3),
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
