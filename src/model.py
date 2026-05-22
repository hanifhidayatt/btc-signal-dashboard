import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle
import os

FEATURES = [
    'EMA_20', 'EMA_50', 'MACD', 'RSI',
    'BB_upper', 'BB_lower', 'BB_width', 'Volume_change',
    'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
    'Price_EMA20_ratio', 'Price_EMA50_ratio', 'EMA_cross',
    'Body_size', 'Upper_wick', 'Lower_wick', 'Is_green',
    'EMA_200', 'Is_bull_market', 'ADX', 'ATR', 'ATR_pct'
]


def train_single_model(X, y, label):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    # Handle class imbalance
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale = neg / pos

    print(f"\n=== {label} ===")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"scale_pos_weight: {scale:.1f}")

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric='logloss',
                                  scale_pos_weight=scale)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        scores.append(score)
        print(f"  Fold accuracy: {score:.2%}")

    print(f"Avg Accuracy: {sum(scores)/len(scores):.2%}")

    final = xgb.XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric='logloss',
                              scale_pos_weight=scale)
    final.fit(X, y)
    return final


def train_model(csv_path="data/BTC_USD_features.csv"):
    df = pd.read_csv(csv_path, index_col=0)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        return

    X = df[FEATURES]

    # Train 3.5R model
    model_3r = train_single_model(X, df['Target_3R'], "3.5R Model")

    # Train 5R model
    model_5r = train_single_model(X, df['Target_5R'], "5R Model")

    # Save both
    os.makedirs("models", exist_ok=True)
    with open("models/model_3r.pkl", "wb") as f:
        pickle.dump(model_3r, f)
    with open("models/model_5r.pkl", "wb") as f:
        pickle.dump(model_5r, f)

    print("\n✅ Both models saved!")

    # Show probability distribution on last 20 days
    print("\n=== Last 20 days confidence scores ===")
    last_20 = X.iloc[-20:]
    probs_3r = model_3r.predict_proba(last_20)[:, 1]
    probs_5r = model_5r.predict_proba(last_20)[:, 1]
    prob_df = pd.DataFrame({
        'Date': df.index[-20:],
        '3R_conf': probs_3r.round(3),
        '5R_conf': probs_5r.round(3),
        'Target_3R': df['Target_3R'].iloc[-20:].values,
        'Target_5R': df['Target_5R'].iloc[-20:].values
    })
    print(prob_df.to_string(index=False))

    # Show signal logic on latest data
    print("\n=== Latest Signal ===")
    latest = X.iloc[[-1]]
    prob_3r = model_3r.predict_proba(latest)[0][1]
    prob_5r = model_5r.predict_proba(latest)[0][1]

    print(f"3.5R confidence: {prob_3r:.2%}")
    print(f"5R confidence:   {prob_5r:.2%}")

    if prob_5r >= 0.25:
        print("🚀 Signal: STRONG LONG — target 5R")
    elif prob_3r >= 0.40:
        print("✅ Signal: LONG — target 3.5R")
    else:
        print("⏸️  Signal: FLAT — no trade today")

    print("\n=== Feature Importance (3.5R Model) ===")
    importance = pd.Series(
        model_3r.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)
    print(importance)


if __name__ == "__main__":
    train_model()
