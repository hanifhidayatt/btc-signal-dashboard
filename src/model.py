import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# ── Synchronized Feature Array ──────────────────────────────
# This matches both your features.py output and dashboard exactly
FEATURES = [
    'EMA_20', 'EMA_50', 'MACD', 'RSI',
    'BB_upper', 'BB_lower', 'BB_width', 'Volume_change',
    'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
    'Price_EMA20_ratio', 'Price_EMA50_ratio', 'EMA_cross',
    'Body_size', 'Upper_wick', 'Lower_wick', 'Is_green',
    'EMA_200', 'Is_bull_market', 'ADX', 'ATR', 'ATR_pct',
    'Coinbase_premium', 'Coinbase_premium_pct'
]


def train_and_save_models():
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load dataset
    print("💾 Loading feature dataset...")
    df = pd.read_csv("data/BTC_USD_features.csv", index_col=0)

    # Filter to training set cutoff (2022-12-31) to prevent data leakage from the test period
    train_mask = df.index <= '2022-12-31'
    df_train = df[train_mask]

    X_train = df_train[FEATURES]

    # ── 1. Train Long 3.5R Model ──────────────────────────────
    print("🚀 Training Long 3.5R Model...")
    y_long = df_train['Target_3R']

    neg_long = (y_long == 0).sum()
    pos_long = (y_long == 1).sum()
    scale_pos_weight_long = neg_long / pos_long if pos_long > 0 else 1.0

    model_long = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_long
    )
    model_long.fit(X_train, y_long)

    # Save Long Model
    with open("models/model_3r.pkl", "wb") as f:
        pickle.dump(model_long, f)
    print("✅ Saved: models/model_3r.pkl")

    # ── 2. Train Short 3.5R Model ─────────────────────────────
    print("📉 Training Short 3.5R Model...")
    y_short = df_train['Target_Short_3R']

    neg_short = (y_short == 0).sum()
    pos_short = (y_short == 1).sum()
    scale_pos_weight_short = neg_short / pos_short if pos_short > 0 else 1.0

    model_short = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_short
    )
    model_short.fit(X_train, y_short)

    # Save Short Model
    with open("models/model_short_3r.pkl", "wb") as f:
        pickle.dump(model_short, f)
    print("✅ Saved: models/model_short_3r.pkl")


if __name__ == "__main__":
    train_and_save_models()
    print("\n🎉 Model pipeline successfully synchronized!")
