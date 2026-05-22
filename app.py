import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Signal Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Dark terminal theme ───────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .signal-buy {
        color: #3fb950;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .signal-flat {
        color: #8b949e;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .signal-sell {
        color: #f85149;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .value {
        color: #e6edf3;
        font-size: 1.4rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { color: #e6edf3; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    .stDataFrame { background: #161b22; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'EMA_20', 'EMA_50', 'MACD', 'RSI',
    'BB_upper', 'BB_lower', 'BB_width', 'Volume_change',
    'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5',
    'Price_EMA20_ratio', 'Price_EMA50_ratio', 'EMA_cross',
    'Body_size', 'Upper_wick', 'Lower_wick', 'Is_green',
    'EMA_200', 'Is_bull_market', 'ADX', 'ATR', 'ATR_pct',
    'SP500_return', 'DXY_return'
]

R = 0.02


def get_model_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return model.get_booster().feature_names

# ── Load data & models ────────────────────────────────────────


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("data/BTC_USD_features.csv", index_col=0)
    return df


@st.cache_resource
def load_models():
    with open("models/model_3r.pkl", "rb") as f:
        model_3r = pickle.load(f)
    with open("models/model_short_3r.pkl", "rb") as f:
        model_short_3r = pickle.load(f)
    return model_3r, model_short_3r


@st.cache_data(ttl=3600)
def run_backtest(df, _model_3r, _model_short_3r):
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2024-12-31'

    train_mask = df.index <= train_end
    test_mask = (df.index >= test_start) & (df.index <= test_end)

    X = df[FEATURES]
    X_test = X[test_mask]
    df_test = df[test_mask].copy()

    X_train = X[train_mask]
    y_3r_train = df['Target_3R'][train_mask]
    y_short_3r_train = df['Target_Short_3R'][train_mask]

    neg_3r = (y_3r_train == 0).sum()
    pos_3r = (y_3r_train == 1).sum()
    bt_model_3r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                    eval_metric='logloss',
                                    scale_pos_weight=neg_3r/pos_3r)
    bt_model_3r.fit(X_train, y_3r_train)

    neg_short = (y_short_3r_train == 0).sum()
    pos_short = (y_short_3r_train == 1).sum()
    bt_model_short_3r = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric='logloss',
                                          scale_pos_weight=neg_short/pos_short)
    bt_model_short_3r.fit(X_train, y_short_3r_train)

    probs_3r = bt_model_3r.predict_proba(X_test)[:, 1]
    probs_short_3r = bt_model_short_3r.predict_proba(X_test)[:, 1]

    trades = []
    last_trade_day = -999
    last_result = None

    for i in range(len(X_test)):
        sp500_ret = df_test['SP500_return'].iloc[i]
        dxy_ret = df_test['DXY_return'].iloc[i]

        if sp500_ret < -0.015 or dxy_ret > 0.008:
            continue

        if last_result == 'LOSS' and (i - last_trade_day) < 3:
            continue

        p3r = probs_3r[i]
        pshort3r = probs_short_3r[i]

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

        pnl = reward if target_hit else -R
        result = 'WIN' if target_hit else 'LOSS'
        last_trade_day = i
        last_result = result

        trades.append({
            'Date': df_test.index[i],
            'Signal': signal,
            'Confidence_Long_3R': round(p3r, 3),
            'Confidence_Short_3R': round(pshort3r, 3),
            'Result': result,
            'PnL_R': round(pnl / R, 2)
        })

    return pd.DataFrame(trades)


# ── App ───────────────────────────────────────────────────────
df = load_data()
model_3r, model_short_3r = load_models()
MODEL_FEATURES = get_model_features(model_3r)

# Header
st.markdown("## ₿ BTC Signal Dashboard")
st.markdown("<p class='label'>Machine Learning Trading Signals • Daily</p>",
            unsafe_allow_html=True)
st.markdown("---")

# ── Today's Signal ────────────────────────────────────────────
latest = df[FEATURES].iloc[[-1]]
latest_row = df.iloc[-1]
p3r = model_3r.predict_proba(latest)[0][1]
pshort3r = model_short_3r.predict_proba(latest)[0][1]

btc_price = latest_row['Close']
rsi_val = latest_row['RSI']
adx_val = latest_row['ADX']

if p3r >= 0.70 and p3r > pshort3r:
    signal_label = "🚀 LONG"
    signal_class = "signal-buy"
    signal_detail = f"Target 3.5R (+{R*3.5*100:.0f}%) | Stop -{R*100:.0f}%"
elif pshort3r >= 0.70 and pshort3r > p3r:
    signal_label = "📉 SHORT"
    signal_class = "signal-sell"
    signal_detail = f"Target 3.5R (+{R*3.5*100:.0f}%) | Stop -{R*100:.0f}%"
else:
    signal_label = "⏸ FLAT"
    signal_class = "signal-flat"
    signal_detail = "No high-confidence setup today"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Today's Signal</div>
        <div class='{signal_class}'>{signal_label}</div>
        <div class='label' style='margin-top:8px'>{signal_detail}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>BTC Price</div>
        <div class='value'>${btc_price:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Long Conf. (3.5R)</div>
        <div class='value'>{p3r:.1%}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>Short Conf. (3.5R)</div>
        <div class='value'>{pshort3r:.1%}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    adx_display = f"{adx_val:.1f}"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='label'>RSI / ADX</div>
        <div class='value'>{rsi_val:.1f} / {adx_display}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── BTC Price Chart ───────────────────────────────────────────
st.markdown("### Price Chart with Indicators")

chart_df = df.tail(180).copy()

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.03
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df['Open'], high=chart_df['High'],
    low=chart_df['Low'],   close=chart_df['Close'],
    name='BTC', increasing_line_color='#3fb950',
    decreasing_line_color='#f85149'
), row=1, col=1)

# EMAs
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'],
                         line=dict(color='#58a6ff', width=1.5), name='EMA 20'), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_50'],
                         line=dict(color='#f0883e', width=1.5), name='EMA 50'), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_200'],
                         line=dict(color='#bc8cff', width=1.5, dash='dash'), name='EMA 200'), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_upper'],
                         line=dict(color='#8b949e', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_lower'],
                         line=dict(color='#8b949e', width=1, dash='dot'), name='BB Lower',
                         fill='tonexty', fillcolor='rgba(139,148,158,0.05)'), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'],
                         line=dict(color='#58a6ff', width=1.5), name='RSI'), row=2, col=1)
fig.add_hline(y=70, line_color='#f85149',
              line_dash='dash', line_width=1, row=2, col=1)
fig.add_hline(y=30, line_color='#3fb950',
              line_dash='dash', line_width=1, row=2, col=1)

# ADX
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['ADX'],
                         line=dict(color='#f0883e', width=1.5), name='ADX'), row=3, col=1)
fig.add_hline(y=25, line_color='#8b949e',
              line_dash='dash', line_width=1, row=3, col=1)

fig.update_layout(
    height=600,
    paper_bgcolor='#0d1117',
    plot_bgcolor='#161b22',
    font=dict(color='#8b949e'),
    xaxis_rangeslider_visible=False,
    legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
    margin=dict(l=0, r=0, t=20, b=0)
)
fig.update_xaxes(gridcolor='#21262d', showgrid=True)
fig.update_yaxes(gridcolor='#21262d', showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# ── Backtest Results ──────────────────────────────────────────
st.markdown("### Backtest Performance (2023–2024 Bull Run)")

trades_df = run_backtest(df, model_3r, model_short_3r)

if not trades_df.empty:
    wins = (trades_df['Result'] == 'WIN').sum()
    losses = (trades_df['Result'] == 'LOSS').sum()
    total = len(trades_df)
    win_rate = wins / total
    total_r = trades_df['PnL_R'].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trades", total)
    m2.metric("Win Rate", f"{win_rate:.1%}")
    m3.metric("Total R", f"+{total_r:.1f}R" if total_r >
              0 else f"{total_r:.1f}R")
    m4.metric("Avg R/Trade", f"{trades_df['PnL_R'].mean():.2f}R")

    # Equity curve
    trades_df['Cumulative_R'] = trades_df['PnL_R'].cumsum()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=trades_df['Date'],
        y=trades_df['Cumulative_R'],
        fill='tozeroy',
        line=dict(color='#3fb950', width=2),
        fillcolor='rgba(63,185,80,0.1)',
        name='Cumulative R'
    ))
    fig2.add_hline(y=0, line_color='#8b949e', line_dash='dash', line_width=1)
    fig2.update_layout(
        height=300,
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#8b949e'),
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_title='Cumulative R',
        showlegend=False
    )
    fig2.update_xaxes(gridcolor='#21262d')
    fig2.update_yaxes(gridcolor='#21262d')
    st.plotly_chart(fig2, use_container_width=True)

    # Trade history table
    st.markdown("### Trade History")
    styled = trades_df[['Date', 'Signal', 'Confidence_Long_3R',
                        'Confidence_Short_3R', 'Result', 'PnL_R']].copy()
    styled['Result'] = styled['Result'].apply(
        lambda x: f"✅ WIN" if x == 'WIN' else "❌ LOSS"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
