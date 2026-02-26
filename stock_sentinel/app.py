"""
app.py — Stock Sentinel Streamlit Dashboard.

Tabs:
  1. Prices & Forecast  — Plotly candlestick, Prophet/LSTM overlay, anomalies
  2. Sentiment          — VADER bar chart, word cloud, correlation heatmap
  3. Backtest           — Equity curve, risk metrics, PuLP portfolio weights

Chat query box parses keywords and navigates/highlights the relevant view.

Caching:
  @st.cache_data     → pipeline data (re-fetched per session or on refresh)
  @st.cache_resource → expensive ML models (Prophet / LSTM)
"""

import io
import sys
import os

# Ensure the stock_sentinel folder is on the path when run from repo root
# (needed for Streamlit Community Cloud deployment)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Stock Sentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (avoid hard crashes if a lib is missing) ───────────────────
def _try_import(*modules):
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            st.warning(f"Optional library '{mod}' not installed — some features disabled.")

_try_import("wordcloud", "prophet", "tensorflow", "pulp")

from utils import (
    NEWS_API_KEY, NEWSDATA_API_KEY, NEWS_SOURCE, STOCKS,
    DEFAULT_START, DEFAULT_END, INITIAL_CAPITAL, COMMISSION,
    plot_candlestick, plot_forecast, plot_equity_curve,
    plot_sentiment_bars, generate_report_text,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Sentinel")
    st.caption("AI-powered stock analysis · For educational use only")
    st.divider()

    symbol = st.text_input("Stock Symbol", value="AAPL",
                            placeholder="e.g. AAPL, TSLA, PSO.KA").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime(DEFAULT_START))
    with col2:
        end_date = st.date_input("To",   value=pd.to_datetime(DEFAULT_END))

    news_source = st.selectbox(
        "News Source", ["auto", "newsapi", "newsdata"],
        index=0, help="'auto' tries NewsAPI first, falls back to NewsData.io"
    )
    newsapi_key   = st.text_input("NewsAPI Key",   value=NEWS_API_KEY,
                                   type="password", placeholder="Enter NewsAPI key")
    newsdata_key  = st.text_input("NewsData Key",  value=NEWSDATA_API_KEY,
                                   type="password", placeholder="Enter NewsData.io key")

    st.divider()
    run_lstm    = st.toggle("Enable LSTM", value=False,
                             help="LSTM training is slow — enable only when needed")
    run_backtest_flag = st.toggle("Run Backtest", value=True)
    refresh     = st.button("🔄 Refresh Data", use_container_width=True)

    st.divider()
    st.markdown(
        "<small>⚠️ **Disclaimer**: This tool is for educational and simulation "
        "purposes only. It is NOT financial advice. Past simulated performance "
        "does not guarantee future results.</small>",
        unsafe_allow_html=True,
    )


# Top-level warning banner
st.warning(
    "⚠️ **Educational / Simulation Only** — Stock Sentinel is not financial advice. "
    "All forecasts and backtest results are simulations using historical data.",
    icon="⚠️",
)

st.title(f"📊 {symbol} — Stock Sentinel")


# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching price & news data…", ttl=3600)
def load_data(sym, start, end, na_key, nd_key, news_src, _force):
    """Cache-busted by `_force` token (changes on every Refresh click)."""
    from data_pipeline import build_dataset
    from sentiment import add_sentiment_to_df, clean_text, score_sentiment

    df, texts_by_date = build_dataset(
        sym, str(start), str(end),
        newsapi_key=na_key, newsdata_key=nd_key,
        force_refresh=bool(_force),
    )
    if df.empty:
        return df, {}, []

    # Clean and score sentiment
    cleaned_by_date = {
        date: [clean_text(t) for t in texts]
        for date, texts in texts_by_date.items()
    }
    df = add_sentiment_to_df(df, cleaned_by_date)

    all_cleaned = [t for texts in cleaned_by_date.values() for t in texts]
    return df, cleaned_by_date, all_cleaned


@st.cache_resource(show_spinner="Training Prophet model…")
def load_prophet(df_hash, df_json, horizon=30):
    """Cache Prophet model by data hash so it's not re-trained on each interaction."""
    from modeling import add_technical_indicators, add_lag_features, forecast_with_prophet
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    forecast = forecast_with_prophet(df, horizon=horizon)
    return forecast


@st.cache_resource(show_spinner="Training LSTM model (this may take a minute)…")
def load_lstm(df_hash, df_json):
    from modeling import add_technical_indicators, add_lag_features, train_lstm, predict_lstm
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    model, scaler, train_m, test_m = train_lstm(df)
    preds = None
    if model is not None:
        preds = predict_lstm(model, scaler, df)
    return preds, train_m, test_m


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
force_token = int(st.session_state.get("refresh_count", 0))
if refresh:
    st.session_state["refresh_count"] = force_token + 1
    force_token += 1

with st.spinner("Loading data…"):
    try:
        df, cleaned_by_date, all_texts = load_data(
            symbol, start_date, end_date,
            newsapi_key, newsdata_key, news_source, force_token,
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if df.empty:
    st.error(
        f"No data returned for **{symbol}** between {start_date} and {end_date}. "
        "Check the symbol and your network connection."
    )
    st.stop()

# ─── Technical indicators + lag features ─────────────────────────────────────
from modeling import add_technical_indicators, add_lag_features, naive_baseline, detect_anomalies

df = add_technical_indicators(df)
df = add_lag_features(df)
df = detect_anomalies(df)
df.dropna(inplace=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Prices & Forecast", "🧠 Sentiment", "⚙️ Backtest"])


# ════════════════════════════════════════════════════════
# TAB 1 — PRICES & FORECAST
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"{symbol} — Price Chart & Forecast")

    # Candlestick
    fig_candle = plot_candlestick(df, symbol)
    # Overlay anomalies
    anomaly_df = df[df["anomaly"]]
    if not anomaly_df.empty:
        import plotly.graph_objects as go
        fig_candle.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df["Close"],
            mode="markers",
            marker=dict(symbol="x", size=10, color="red"),
            name="Anomaly",
        ), row=1, col=1)
    st.plotly_chart(fig_candle, use_container_width=True)

    # Key stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close",  f"${df['Close'].iloc[-1]:.2f}")
    c2.metric("52-Week High", f"${df['Close'].max():.2f}")
    c3.metric("52-Week Low",  f"${df['Close'].min():.2f}")
    c4.metric("Anomalies",    f"{df['anomaly'].sum()}")

    st.divider()

    # Forecast (Prophet)
    horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
    df_json = df[["Open","High","Low","Close","Volume","sentiment"]].to_json()
    df_hash = hash(df_json[:500])   # cheap fingerprint

    with st.spinner("Running Prophet forecast…"):
        try:
            forecast = load_prophet(df_hash, df_json, horizon=horizon)
        except Exception as e:
            st.warning(f"Prophet forecast failed: {e}")
            forecast = None

    baseline = naive_baseline(df)

    # LSTM (optional)
    lstm_preds = None
    if run_lstm:
        with st.spinner("Training LSTM…"):
            try:
                lstm_preds, train_m, test_m = load_lstm(df_hash, df_json)
                if lstm_preds is not None:
                    st.success(
                        f"LSTM trained — Test RMSE: {test_m.get('rmse', 'N/A'):.4f} | "
                        f"R²: {test_m.get('r2', 'N/A'):.4f}"
                    )
            except Exception as e:
                st.warning(f"LSTM training failed: {e}")

    fig_fc = plot_forecast(df, forecast, symbol=symbol, baseline=baseline)
    if lstm_preds is not None:
        import plotly.graph_objects as go
        fig_fc.add_trace(go.Scatter(
            x=lstm_preds.index, y=lstm_preds.values,
            name="LSTM Prediction",
            line=dict(color="#e056fd", width=1.5, dash="dash"),
        ))
    st.plotly_chart(fig_fc, use_container_width=True)

    # Technical indicators accordion
    with st.expander("📐 Technical Indicators", expanded=False):
        ind_cols = [c for c in ["rsi_14","macd","boll_upper","boll_lower","atr_14"]
                    if c in df.columns]
        if ind_cols:
            st.dataframe(df[ind_cols].tail(20).style.highlight_max(color="#1a472a")
                                                    .highlight_min(color="#641220"),
                         use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — SENTIMENT
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"{symbol} — Sentiment Analysis")

    if "sentiment" not in df.columns or df["sentiment"].abs().max() == 0:
        st.info("No sentiment data available. Add valid API keys to enable news sentiment.")
    else:
        # Sentiment bar chart
        fig_sent = plot_sentiment_bars(df["sentiment"], symbol=symbol)
        st.plotly_chart(fig_sent, use_container_width=True)

        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Sentiment",  f"{df['sentiment'].mean():.4f}")
        c2.metric("Max Positive",    f"{df['sentiment'].max():.4f}")
        c3.metric("Max Negative",    f"{df['sentiment'].min():.4f}")

        st.divider()

        col_wc, col_kw = st.columns([2, 1])

        with col_wc:
            st.markdown("**Word Cloud**")
            if all_texts:
                from sentiment import generate_wordcloud
                wc_img = generate_wordcloud(all_texts)
                if wc_img is not None:
                    st.image(wc_img, use_column_width=True,
                             caption="Most frequent words in news/social posts")
                else:
                    st.info("Word cloud unavailable — install: pip install wordcloud")
            else:
                st.info("No text data collected for word cloud.")

        with col_kw:
            st.markdown("**Top Keywords**")
            if all_texts:
                from sentiment import get_top_keywords
                keywords = get_top_keywords(all_texts, n=15)
                kw_df = pd.DataFrame(keywords, columns=["Keyword", "Count"])
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

        st.divider()

        # Correlation heatmap
        st.markdown("**Sentiment ↔ Price Correlation**")
        from sentiment import sentiment_price_correlation
        corr = sentiment_price_correlation(df)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            font=dict(color="#c9d1d9"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"{symbol} — Strategy Backtest")

    if not run_backtest_flag:
        st.info("Enable **Run Backtest** in the sidebar to see results.")
    else:
        from backtesting import generate_signals, run_backtest, calc_all_metrics, generate_insights

        pred_col = "lstm_pred" if lstm_preds is not None else "log_return"
        if lstm_preds is not None:
            df["lstm_pred"] = lstm_preds.reindex(df.index)

        with st.spinner("Running backtest…"):
            signals = generate_signals(df, pred_col=pred_col)
            portfolio_df = run_backtest(df, signals, capital=INITIAL_CAPITAL,
                                        commission=COMMISSION)
            metrics = calc_all_metrics(portfolio_df, capital=INITIAL_CAPITAL)

        # Equity curve
        fig_eq = plot_equity_curve(portfolio_df, symbol=symbol)
        st.plotly_chart(fig_eq, use_container_width=True)

        # Metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        ret_color = "normal" if metrics["total_return_pct"] >= 0 else "inverse"
        c1.metric("Total Return",   f"{metrics['total_return_pct']:+.2f}%")
        c2.metric("Sharpe Ratio",   f"{metrics['sharpe']:.3f}")
        c3.metric("VaR (95%)",      f"{metrics['var_95']:.2%}")
        c4.metric("Max Drawdown",   f"{metrics['max_drawdown']:.2%}")
        c5.metric("No. of Trades",  f"{metrics['num_trades']}")

        # Signals overlay
        with st.expander("📋 Signal Details", expanded=False):
            sig_df = pd.DataFrame({"signal": signals, "close": df["Close"]})
            sig_df["signal_label"] = sig_df["signal"].map(
                {1: "BUY", -1: "SELL", 0: "HOLD"})
            st.dataframe(sig_df[sig_df["signal"] != 0].tail(30),
                         use_container_width=True)

        st.divider()

        # Narrative
        st.markdown("**📝 Strategy Narrative**")
        narrative = generate_insights(metrics, symbol=symbol)
        st.code(narrative, language="")

        # Portfolio optimizer
        st.divider()
        st.markdown("**🎯 Portfolio Optimization (Multi-Stock)**")
        selected_stocks = st.multiselect(
            "Select stocks to optimize (max 5)",
            options=STOCKS, default=["AAPL", "MSFT"],
            max_selections=5,
        )
        if st.button("Optimize Portfolio", key="opt_btn") and len(selected_stocks) >= 2:
            from backtesting import optimize_portfolio
            from data_pipeline import get_stock_data

            with st.spinner("Fetching returns for all selected stocks…"):
                all_returns = {}
                for s in selected_stocks:
                    s_df = get_stock_data(s, str(start_date), str(end_date))
                    if not s_df.empty:
                        all_returns[s] = s_df["Close"].pct_change().dropna()

                if len(all_returns) >= 2:
                    returns_df_multi = pd.DataFrame(all_returns).dropna()
                    weights = optimize_portfolio(returns_df_multi, target_return=0.0005)
                    wt_df = pd.DataFrame(list(weights.items()), columns=["Stock", "Weight"])
                    import plotly.graph_objects as go
                    fig_pie = px.pie(wt_df, names="Stock", values="Weight",
                                     title="Optimized Portfolio Weights",
                                     color_discrete_sequence=px.colors.qualitative.Safe)
                    fig_pie.update_layout(paper_bgcolor="#0e1117", font=dict(color="#c9d1d9"))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.dataframe(wt_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Not enough stock data to optimize.")


# ─────────────────────────────────────────────
# CHAT QUERY BOX (bottom of page)
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 💬 Ask Stock Sentinel")
user_query = st.text_input(
    "Type a question (e.g. 'predict price', 'show sentiment', 'backtest risk')",
    key="chat_input",
    placeholder="e.g. predict AAPL price for next 30 days",
)

if user_query:
    q = user_query.lower()
    if any(k in q for k in ["predict", "forecast", "price", "future"]):
        response = (
            f"📈 **Price Forecast for {symbol}**: See Tab 1 for the Prophet forecast "
            f"(orange dashed line) and confidence interval. The next {horizon}-day "
            f"outlook is shown beyond the historical data."
        )
    elif any(k in q for k in ["sentiment", "news", "social", "tweet", "feel"]):
        mean_sent = df.get("sentiment", pd.Series([0])).mean()
        label = "positive 🟢" if mean_sent > 0.05 else "negative 🔴" if mean_sent < -0.05 else "neutral ⚪"
        response = (
            f"🧠 **Sentiment for {symbol}**: Overall sentiment is **{label}** "
            f"(mean VADER score: {mean_sent:.4f}). See Tab 2 for daily breakdown."
        )
    elif any(k in q for k in ["risk", "backtest", "portfolio", "return", "sharpe", "var", "drawdown"]):
        m = metrics if run_backtest_flag else {}
        response = (
            f"⚙️ **Backtest Summary for {symbol}**: "
            f"Return {m.get('total_return_pct', 'N/A'):+.2f}% | "
            f"Sharpe {m.get('sharpe', 'N/A'):.3f} | "
            f"VaR {m.get('var_95', 'N/A'):.2%} | "
            f"Max Drawdown {m.get('max_drawdown', 'N/A'):.2%}. "
            f"See Tab 3 for full equity curve and metrics."
        ) if run_backtest_flag else "Enable **Run Backtest** in the sidebar first."
    elif any(k in q for k in ["anomaly", "crash", "spike", "unusual"]):
        n_anom = int(df["anomaly"].sum())
        response = (
            f"🚨 **Anomalies for {symbol}**: IsolationForest detected **{n_anom}** "
            f"unusual trading days. They are marked with ✕ on the candlestick chart in Tab 1."
        )
    else:
        response = (
            f"I can answer questions about: **price forecasts**, **sentiment analysis**, "
            f"**backtest results**, and **anomaly detection** for {symbol}. "
            f"Try: 'predict price', 'show sentiment', or 'what is the risk?'"
        )

    st.markdown(f"**🤖 Sentinel:** {response}")
