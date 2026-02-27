"""
app.py — Stock Sentinel Streamlit Dashboard.
Beginner-friendly version: plain-English labels, explanations, and full error handling.
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Optional library warnings ──────────────────────────────────────────────
def _try_import(*modules):
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            st.warning(f"Optional library '{mod}' not installed — some features disabled.")

_try_import("wordcloud", "prophet", "pulp")

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

    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, NBP.KA",
        help="Type any stock ticker. Pakistani stocks need .KA suffix (e.g. NBP.KA, MCB.KA)"
    ).upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.to_datetime(DEFAULT_START))
    with col2:
        end_date = st.date_input("To", value=pd.to_datetime(DEFAULT_END))

    # Date validation
    if start_date >= end_date:
        st.error("'From' date must be before 'To' date.")
        st.stop()

    date_range_days = (end_date - start_date).days
    if date_range_days < 60:
        st.warning(
            "⚠️ Date range is less than 2 months. This may not have enough data "
            "for charts to work. Try selecting at least 3 months."
        )

    news_source = st.selectbox(
        "News Source",
        ["auto", "newsapi", "newsdata"],
        index=0,
        help="'auto' tries NewsAPI first, falls back to NewsData.io"
    )
    newsapi_key  = st.text_input("NewsAPI Key",  value=NEWS_API_KEY,
                                  type="password", placeholder="Enter NewsAPI key")
    newsdata_key = st.text_input("NewsData Key", value=NEWSDATA_API_KEY,
                                  type="password", placeholder="Enter NewsData.io key")

    st.divider()
    run_lstm          = st.toggle("Enable LSTM (slow)", value=False,
                                   help="LSTM is a deep learning model — takes 2-3 min. Disable for speed.")
    run_backtest_flag = st.toggle("Run Strategy Test", value=True,
                                   help="Simulate buying/selling based on AI predictions")
    refresh           = st.button("🔄 Refresh Data", use_container_width=True)

    st.divider()
    st.markdown(
        "<small>⚠️ **Disclaimer**: This tool is for educational and simulation "
        "purposes only. It is NOT financial advice.</small>",
        unsafe_allow_html=True,
    )

# ─── Top banner ─────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Educational / Simulation Only** — Stock Sentinel is NOT financial advice. "
    "All charts and strategy tests use historical data. Do NOT make real investment "
    "decisions based on this tool.",
    icon="⚠️",
)
st.title(f"📊 {symbol} — Stock Sentinel")

# ─────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching price & news data…")
def load_data(sym, start, end, na_key, nd_key, news_src, _force):
    from data_pipeline import build_dataset
    from sentiment import add_sentiment_to_df, clean_text

    df, texts_by_date = build_dataset(
        sym, str(start), str(end),
        newsapi_key=na_key, newsdata_key=nd_key,
        force_refresh=bool(_force),
    )
    if df.empty:
        return df, {}, []

    cleaned_by_date = {
        date: [clean_text(t) for t in texts]
        for date, texts in texts_by_date.items()
    }
    df = add_sentiment_to_df(df, cleaned_by_date)
    all_cleaned = [t for texts in cleaned_by_date.values() for t in texts]
    return df, cleaned_by_date, all_cleaned


@st.cache_resource(show_spinner="Running price forecast (Prophet)…")
def load_prophet(df_hash, df_json, horizon=30):
    from modeling import add_technical_indicators, add_lag_features, forecast_with_prophet
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    if df.empty:
        return None
    return forecast_with_prophet(df, horizon=horizon)


@st.cache_resource(show_spinner="Training AI (LSTM) model — this takes a few minutes…")
def load_lstm(df_hash, df_json):
    from modeling import add_technical_indicators, add_lag_features, train_lstm, predict_lstm
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    if df.empty:
        return None, {}, {}
    model, scaler, train_m, test_m = train_lstm(df)
    preds = None
    if model is not None:
        try:
            preds = predict_lstm(model, scaler, df)
        except Exception:
            preds = None
    return preds, train_m, test_m


# ─────────────────────────────────────────────
# LOAD DATA — auto-refresh when symbol or dates change
# ─────────────────────────────────────────────
force_token = int(st.session_state.get("refresh_count", 0))

_prev_key = st.session_state.get("_last_query_key", "")
_curr_key = f"{symbol}|{start_date}|{end_date}"
if _prev_key != _curr_key:
    force_token += 1
    st.session_state["refresh_count"] = force_token
    st.session_state["_last_query_key"] = _curr_key

if refresh:
    force_token += 1
    st.session_state["refresh_count"] = force_token
    st.session_state["_last_query_key"] = _curr_key

with st.spinner("Loading stock data…"):
    try:
        df, cleaned_by_date, all_texts = load_data(
            symbol, start_date, end_date,
            newsapi_key, newsdata_key, news_source, force_token,
        )
    except Exception as e:
        st.error(
            f"**Could not load data for `{symbol}`.**\n\n"
            f"Possible reasons:\n"
            f"- Wrong symbol (Pakistani stocks need `.KA` suffix: `NBP.KA`, `MCB.KA`)\n"
            f"- No internet connection\n"
            f"- Yahoo Finance temporarily unavailable\n\n"
            f"Technical detail: `{e}`"
        )
        st.stop()

if df is None or df.empty:
    st.error(
        f"**No data found for `{symbol}`** between {start_date} and {end_date}.\n\n"
        f"Try:\n"
        f"- Check the symbol at [Yahoo Finance](https://finance.yahoo.com)\n"
        f"- Pakistani stocks: use `NBP.KA`, `MCB.KA`, `OGDC.KA` format\n"
        f"- Extend the date range (more historical data = better results)"
    )
    st.stop()

# ─── Technical indicators ────────────────────────────────────────────────────
try:
    from modeling import add_technical_indicators, add_lag_features, naive_baseline, detect_anomalies
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = detect_anomalies(df)
    df.dropna(inplace=True)
except Exception as e:
    st.error(f"Error computing technical indicators: `{e}`")
    st.stop()

if df.empty or len(df) < 5:
    st.error(
        f"**Not enough data for `{symbol}`** to draw charts.\n\n"
        f"**Why?** Price charts need at least **30 trading days** of data "
        f"(about 6 weeks). Technical indicators like MACD require 26 days just to warm up.\n\n"
        f"**Fix:** Select a date range of at least **3 months**, "
        f"or check the symbol is correct (Pakistani stocks: `NBP.KA`)."
    )
    st.stop()

# Safe defaults
metrics    = {}
horizon    = 30
lstm_preds = None

# ─────────────────────────────────────────────
# QUICK SUMMARY CARD (plain English for everyone)
# ─────────────────────────────────────────────
trend = "📈 Going UP" if df["Close"].iloc[-1] > df["Close"].iloc[0] else "📉 Going DOWN"
price_change_pct = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
mean_sentiment = df["sentiment"].mean() if "sentiment" in df.columns else 0.0
sentiment_label = "🟢 Positive news" if mean_sentiment > 0.05 else ("🔴 Negative news" if mean_sentiment < -0.05 else "⚪ Neutral news")

st.markdown("### 📋 Quick Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Price Trend (selected period)", trend, f"{price_change_pct:+.1f}%")
s2.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
s3.metric("News Mood", sentiment_label)
s4.metric("Unusual Days Detected", f"{int(df['anomaly'].sum())} days" if "anomaly" in df.columns else "N/A")

with st.expander("💡 What does this summary mean?"):
    st.markdown("""
| Term | Plain explanation |
|---|---|
| **Price Trend** | Did the stock go up or down over your selected dates? |
| **Current Price** | The last closing price in your date range |
| **News Mood** | Were recent news headlines mostly good or bad about this company? |
| **Unusual Days** | Days when the stock moved in a very unexpected way (AI-detected) |
""")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Price Chart & Forecast", "📰 News Mood", "🎯 Strategy Test"])


# ═══════════════════════════════════════════════
# TAB 1 — PRICE CHART & FORECAST
# ═══════════════════════════════════════════════
with tab1:
    st.subheader(f"{symbol} — Price History")

    with st.expander("💡 How to read the price chart"):
        st.markdown("""
- **Green candles** = the price went UP that day
- **Red candles** = the price went DOWN that day
- **Red ✕ marks** = unusual/unexpected price movements detected by AI
- The **bottom panel** shows how many shares were traded (Volume)
        """)

    try:
        fig_candle = plot_candlestick(df, symbol)
        if "anomaly" in df.columns:
            anomaly_df = df[df["anomaly"]]
            if not anomaly_df.empty:
                import plotly.graph_objects as go
                fig_candle.add_trace(go.Scatter(
                    x=anomaly_df.index, y=anomaly_df["Close"],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="red"),
                    name="Unusual Day",
                    hovertemplate="<b>Unusual Day</b><br>%{x|%b %d, %Y}<br>Price: $%{y:.2f}<extra></extra>",
                ), row=1, col=1)
        st.plotly_chart(fig_candle, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not draw price chart: `{e}`")

    c1, c2, c3, c4 = st.columns(4)
    try:
        c1.metric("Latest Price",    f"${df['Close'].iloc[-1]:.2f}",
                  help="Last closing price in your date range")
        c2.metric("Highest Price",   f"${df['Close'].max():.2f}",
                  help="The highest the stock ever closed in this period")
        c3.metric("Lowest Price",    f"${df['Close'].min():.2f}",
                  help="The lowest the stock ever closed in this period")
        c4.metric("Unusual Days",    f"{int(df['anomaly'].sum())}" if "anomaly" in df.columns else "N/A",
                  help="Days when price moved in a surprising way, detected by AI")
    except Exception as e:
        st.warning(f"Could not compute price stats: `{e}`")

    st.divider()
    st.subheader("🔮 Price Forecast (AI Prediction)")

    with st.expander("💡 How to read the forecast chart"):
        st.markdown("""
- **Solid green line** = actual historical price
- **Orange dashed line** = AI's best guess for future price
- **Golden shaded area** = the range where the AI thinks the price will likely fall (80% confident)
- **Dotted grey line** = simple "if tomorrow = today" baseline (used to check if AI is actually better)

> The wider the golden band, the **more uncertain** the forecast.
> This is just a prediction — real prices can move very differently!
        """)

    horizon = st.slider("How many days to forecast into the future?", 7, 90, 30)

    try:
        df_cols = [c for c in ["Open","High","Low","Close","Volume","sentiment"] if c in df.columns]
        df_json = df[df_cols].to_json()
        df_hash = hash(df_json[:500])
    except Exception:
        df_json = df[["Close"]].to_json()
        df_hash = hash(df_json[:500])

    forecast = None
    with st.spinner("Running AI price forecast…"):
        try:
            forecast = load_prophet(df_hash, df_json, horizon=horizon)
        except Exception as e:
            st.warning(f"Forecast could not run: `{e}`. Make sure 'prophet' is installed.")

    try:
        baseline = naive_baseline(df)
    except Exception:
        baseline = None

    # LSTM (optional)
    lstm_preds = None
    if run_lstm:
        with st.spinner("Training deep learning model (LSTM) — this takes 2-3 minutes…"):
            try:
                lstm_preds, train_m, test_m = load_lstm(df_hash, df_json)
                if lstm_preds is not None:
                    rmse     = test_m.get("rmse", None)
                    r2       = test_m.get("r2", None)
                    rmse_str = f"${rmse:.2f}" if isinstance(rmse, float) else "N/A"
                    r2_str   = f"{r2:.1%}"    if isinstance(r2,   float) else "N/A"
                    st.success(
                        f"✅ Deep learning model trained! "
                        f"Average prediction error: **{rmse_str}** | "
                        f"Accuracy score: **{r2_str}** (1.0 = perfect)"
                    )
            except Exception as e:
                st.warning(f"Deep learning model failed: `{e}`")

    try:
        fig_fc = plot_forecast(df, forecast, symbol=symbol, baseline=baseline)
        if lstm_preds is not None:
            import plotly.graph_objects as go
            fig_fc.add_trace(go.Scatter(
                x=lstm_preds.index, y=lstm_preds.values,
                name="Deep Learning Prediction",
                line=dict(color="#e056fd", width=1.5, dash="dash"),
                hovertemplate="Deep Learning: $%{y:.2f}<extra></extra>",
            ))
        st.plotly_chart(fig_fc, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not draw forecast chart: `{e}`")

    # Technical indicators — hidden by default, with plain labels
    with st.expander("📐 Advanced Indicators (for experienced users)"):
        st.markdown("""
| Indicator | What it measures | Reading |
|---|---|---|
| **RSI (14)** | Is the stock overbought or oversold? | Above 70 = may drop soon, Below 30 = may rise soon |
| **MACD** | Is momentum increasing or decreasing? | Positive = upward momentum, Negative = downward |
| **Bollinger Upper** | Upper price boundary (statistical) | Price near this = potentially overpriced |
| **Bollinger Lower** | Lower price boundary (statistical) | Price near this = potentially underpriced |
| **ATR (14)** | How much does the price swing daily? | Higher = more volatile/risky |
        """)
        try:
            ind_cols = [c for c in ["rsi_14","macd","boll_upper","boll_lower","atr_14"] if c in df.columns]
            if ind_cols:
                display_df = df[ind_cols].tail(20).copy()
                display_df.columns = [
                    c.replace("rsi_14","RSI (0-100)")
                     .replace("macd","Momentum (MACD)")
                     .replace("boll_upper","Upper Boundary")
                     .replace("boll_lower","Lower Boundary")
                     .replace("atr_14","Daily Swing (ATR)")
                    for c in display_df.columns
                ]
                st.dataframe(display_df.round(2), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load indicator table: `{e}`")


# ═══════════════════════════════════════════════
# TAB 2 — NEWS MOOD
# ═══════════════════════════════════════════════
with tab2:
    st.subheader(f"{symbol} — News & Social Media Mood")

    with st.expander("💡 What is 'News Mood'?"):
        st.markdown("""
**Sentiment analysis** reads news headlines and social posts about a company and decides:
- Is the tone **positive** (good news → green bars)?
- Is the tone **negative** (bad news → red bars)?
- Is the tone **neutral** (just facts, no emotion)?

The score ranges from **-1.0** (very negative) to **+1.0** (very positive).

This can sometimes predict whether a stock will go up or down the next day.
        """)

    has_sentiment = "sentiment" in df.columns and df["sentiment"].abs().max() > 0.001

    if not has_sentiment:
        st.info(
            "📭 **No news data available** for this stock in the selected date range.\n\n"
            "To see real news sentiment:\n"
            "1. Get a free API key from [newsapi.org](https://newsapi.org)\n"
            "2. Enter it in the sidebar under 'NewsAPI Key'\n"
            "3. Click 'Refresh Data'\n\n"
            "Without an API key, sentiment defaults to 0 (neutral)."
        )
    else:
        # Sentiment bar chart
        try:
            fig_sent = plot_sentiment_bars(df["sentiment"], symbol=symbol)
            st.plotly_chart(fig_sent, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not draw news mood chart: `{e}`")

        c1, c2, c3 = st.columns(3)
        mean_s = float(df["sentiment"].mean())
        mood   = "Generally Positive 🟢" if mean_s > 0.05 else ("Generally Negative 🔴" if mean_s < -0.05 else "Mixed/Neutral ⚪")
        c1.metric("Overall News Mood", mood,    help="Average tone of all news in this period")
        c2.metric("Best Day Score",    f"{df['sentiment'].max():.2f}", help="Day with the most positive news (scale: -1 to +1)")
        c3.metric("Worst Day Score",   f"{df['sentiment'].min():.2f}", help="Day with the most negative news (scale: -1 to +1)")

        st.divider()
        col_wc, col_kw = st.columns([2, 1])

        with col_wc:
            st.markdown("**Most Talked-About Words in News**")
            st.caption("Bigger word = mentioned more often in headlines")
            if all_texts:
                try:
                    from sentiment import generate_wordcloud
                    wc_img = generate_wordcloud(all_texts)
                    if wc_img is not None:
                        st.image(wc_img, use_container_width=True)
                    else:
                        st.info("Word cloud unavailable — install: pip install wordcloud")
                except Exception as e:
                    st.warning(f"Word cloud failed: `{e}`")
            else:
                st.info("No news text collected. Add an API key to get real news.")

        with col_kw:
            st.markdown("**Top Keywords**")
            if all_texts:
                try:
                    from sentiment import get_top_keywords
                    keywords = get_top_keywords(all_texts, n=15)
                    kw_df = pd.DataFrame(keywords, columns=["Word", "Times Mentioned"])
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Keyword extraction failed: `{e}`")

        st.divider()
        st.markdown("**Does News Mood Affect Price? (Correlation)**")
        with st.expander("💡 How to read this table"):
            st.markdown("""
Numbers close to **+1.0** = when news is good, price tends to go up.
Numbers close to **-1.0** = when news is good, price tends to go down (unusual).
Numbers close to **0** = news mood and price don't seem related.
            """)
        try:
            from sentiment import sentiment_price_correlation
            corr = sentiment_price_correlation(df)
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdYlGn",
                title="Relationship between News Mood and Price (correlation)",
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#444444"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute correlation: `{e}`")


# ═══════════════════════════════════════════════
# TAB 3 — STRATEGY TEST
# ═══════════════════════════════════════════════
with tab3:
    st.subheader(f"{symbol} — AI Strategy Test (Backtest)")

    with st.expander("💡 What is a Strategy Test?"):
        st.markdown("""
We give the AI **$10,000 of pretend money** and let it trade using the historical data.
It decides when to **BUY** (news is positive + price predicted to rise) and
when to **SELL** (news is negative or price predicted to fall).

We then compare: **did the AI do better than just buying and holding?**

> ⚠️ This is a **simulation** — it uses past data. Real markets are unpredictable.
> DO NOT use these results to make real investment decisions.
        """)

    if not run_backtest_flag:
        st.info("Enable **'Run Strategy Test'** in the sidebar to see results.")
    else:
        try:
            from backtesting import generate_signals, run_backtest, calc_all_metrics, generate_insights

            pred_col = "lstm_pred" if lstm_preds is not None else "log_return"
            if lstm_preds is not None:
                try:
                    df["lstm_pred"] = lstm_preds.reindex(df.index)
                except Exception:
                    pred_col = "log_return"

            with st.spinner("Running strategy test on historical data…"):
                signals      = generate_signals(df, pred_col=pred_col)
                portfolio_df = run_backtest(df, signals, capital=INITIAL_CAPITAL, commission=COMMISSION)
                metrics      = calc_all_metrics(portfolio_df, capital=INITIAL_CAPITAL)

            # Equity curve
            try:
                fig_eq = plot_equity_curve(portfolio_df, symbol=symbol)
                st.plotly_chart(fig_eq, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not draw portfolio chart: `{e}`")

            with st.expander("💡 How to read the portfolio chart"):
                st.markdown("""
- **Green line** = your portfolio value if you followed the AI's BUY/SELL signals
- **Grey dotted line** = if you just bought at the start and held (no trading)
- The strategy is **better** when the green line is above the grey line
                """)

            # Plain-English metrics
            st.markdown("### 📊 Results in Simple Terms")
            c1, c2, c3, c4, c5 = st.columns(5)

            ret = metrics.get("total_return_pct", 0)
            ret_label = "Profit" if ret >= 0 else "Loss"
            c1.metric(
                f"Total {ret_label}",
                f"{abs(ret):.1f}%",
                delta=f"{ret:+.1f}%",
                help="Did the AI strategy make or lose money overall?"
            )

            sharpe = metrics.get("sharpe", 0)
            sharpe_label = "Excellent" if sharpe > 1 else ("Good" if sharpe > 0.5 else ("Fair" if sharpe > 0 else "Poor"))
            c2.metric(
                "Profit vs Risk Score",
                f"{sharpe:.2f} ({sharpe_label})",
                help="Higher = better return for the risk taken. Above 1.0 is considered good."
            )

            var = metrics.get("var_95", 0)
            c3.metric(
                "Worst Day Loss Estimate",
                f"{abs(var):.2%}",
                help="On a really bad day, you'd lose about this much. (95% confidence)"
            )

            dd = metrics.get("max_drawdown", 0)
            c4.metric(
                "Biggest Drop from Peak",
                f"{abs(dd):.2%}",
                help="The worst loss from the highest point to the lowest point"
            )

            c5.metric(
                "Number of Trades",
                f"{metrics.get('num_trades', 0)}",
                help="How many times the AI decided to buy or sell"
            )

            with st.expander("💡 What do these numbers mean?"):
                st.markdown(f"""
| Result | Your Score | Plain Meaning |
|---|---|---|
| Total Return | {ret:+.1f}% | You {"made" if ret >= 0 else "lost"} {abs(ret):.1f}% on your pretend $10,000 |
| Profit vs Risk | {sharpe:.2f} ({sharpe_label}) | {"Better than average" if sharpe > 1 else ("About average" if sharpe > 0.5 else "Below average")} compared to a typical good investment |
| Worst Day Loss | {abs(var):.2%} | In 95% of bad days, you wouldn't lose more than this |
| Biggest Drop | {abs(dd):.2%} | The stock once fell this much from its peak in a row |
| Trades | {metrics.get("num_trades",0)} | Each trade has a 0.1% fee, so more trades = more fees |
""")

            # Buy/Sell signal table
            with st.expander("📋 When did the AI say Buy or Sell?"):
                try:
                    sig_df = pd.DataFrame({"Signal": signals, "Price": df["Close"]})
                    sig_df["Action"] = sig_df["Signal"].map({1: "🟢 BUY", -1: "🔴 SELL", 0: "⏸ HOLD"})
                    active = sig_df[sig_df["Signal"] != 0][["Action","Price"]].tail(30)
                    st.dataframe(active, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not show signal table: `{e}`")

            st.divider()

            # Narrative
            st.markdown("**📝 Plain-English Summary**")
            try:
                narrative = generate_insights(metrics, symbol=symbol)
                st.info(narrative)
            except Exception as e:
                st.warning(f"Could not generate summary: `{e}`")

            # Portfolio optimizer
            st.divider()
            st.markdown("**🎯 Which Stocks to Mix? (Portfolio Optimizer)**")
            with st.expander("💡 What is this?"):
                st.markdown("""
Instead of putting all your money in one stock, you can **spread it across several stocks**.
This usually reduces risk. The optimizer figures out the best % to put in each stock
to get a good return with the least risk.

Think of it like: don't put all eggs in one basket.
                """)

            selected_stocks = st.multiselect(
                "Pick 2-5 stocks to combine into a portfolio:",
                options=STOCKS,
                default=["AAPL", "MSFT"],
                max_selections=5,
            )

            if st.button("Find Best Mix", key="opt_btn") and len(selected_stocks) >= 2:
                from backtesting import optimize_portfolio
                from data_pipeline import get_stock_data

                with st.spinner("Fetching data for all selected stocks…"):
                    all_returns = {}
                    failed = []
                    for s in selected_stocks:
                        try:
                            s_df = get_stock_data(s, str(start_date), str(end_date))
                            if not s_df.empty and "Close" in s_df.columns:
                                all_returns[s] = s_df["Close"].pct_change().dropna()
                            else:
                                failed.append(s)
                        except Exception:
                            failed.append(s)

                    if failed:
                        st.warning(f"Could not get data for: {', '.join(failed)}. Continuing without them.")

                    if len(all_returns) >= 2:
                        try:
                            returns_df_multi = pd.DataFrame(all_returns).dropna()
                            if returns_df_multi.empty:
                                st.error("Not enough overlapping data between selected stocks.")
                            else:
                                weights = optimize_portfolio(returns_df_multi, target_return=0.0005)
                                wt_df = pd.DataFrame(list(weights.items()), columns=["Stock", "Recommended %"])
                                wt_df["Recommended %"] = (wt_df["Recommended %"] * 100).round(1)

                                import plotly.graph_objects as go
                                fig_pie = px.pie(
                                    wt_df, names="Stock", values="Recommended %",
                                    title="Suggested Portfolio Split",
                                    color_discrete_sequence=px.colors.qualitative.Safe,
                                )
                                fig_pie.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#444444"),
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                                st.dataframe(wt_df, use_container_width=True, hide_index=True)
                                st.caption(
                                    "This shows the mathematically optimal way to split $10,000 "
                                    "across these stocks to minimize risk while targeting a small daily gain."
                                )
                        except Exception as e:
                            st.error(f"Portfolio optimization failed: `{e}`")
                    else:
                        st.error("Not enough valid stock data to optimize. Try different symbols.")
            elif st.session_state.get("opt_btn") and len(selected_stocks) < 2:
                st.warning("Please select at least 2 stocks.")

        except Exception as e:
            st.error(
                f"**Strategy test failed.**\n\n"
                f"This can happen when there is not enough price data for signals to work.\n\n"
                f"Technical detail: `{e}`"
            )


# ─────────────────────────────────────────────
# CHAT QUERY BOX
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 💬 Ask Stock Sentinel")
st.caption("Ask a question in plain English — no technical knowledge needed!")

user_query = st.text_input(
    "What would you like to know?",
    key="chat_input",
    placeholder="e.g. 'Is AAPL doing well?' or 'Should I be worried about the risk?'",
)

if user_query:
    q = user_query.lower()
    try:
        if any(k in q for k in ["predict", "forecast", "future", "price", "go up", "go down"]):
            direction = "upward" if forecast is not None and not forecast.empty and \
                        forecast["yhat"].iloc[-1] > df["Close"].iloc[-1] else "downward"
            response = (
                f"📈 **Price Forecast for {symbol}**: The AI model suggests a **{direction} trend** "
                f"over the next {horizon} days. See the orange dashed line in the 'Price Chart & Forecast' tab. "
                f"Remember: this is just a prediction, not a guarantee!"
            )
        elif any(k in q for k in ["news", "sentiment", "mood", "feel", "tweet", "social"]):
            mood_word = "positive" if mean_sentiment > 0.05 else ("negative" if mean_sentiment < -0.05 else "neutral")
            response = (
                f"📰 **News Mood for {symbol}**: The overall news tone is **{mood_word}** "
                f"(score: {mean_sentiment:.2f} on a -1 to +1 scale). "
                f"Check the 'News Mood' tab for daily details and word clouds."
            )
        elif any(k in q for k in ["risk", "safe", "dangerous", "loss", "drawdown", "sharpe"]):
            dd_pct = abs(metrics.get("max_drawdown", 0)) * 100
            sharpe_val = metrics.get("sharpe", 0)
            risk_word = "low" if dd_pct < 10 else ("moderate" if dd_pct < 25 else "high")
            response = (
                f"⚠️ **Risk Summary for {symbol}**: The strategy has **{risk_word} risk**. "
                f"Biggest drop from peak: {dd_pct:.1f}%. "
                f"Risk-adjusted score (Sharpe): {sharpe_val:.2f} "
                f"({'good' if sharpe_val > 1 else 'average' if sharpe_val > 0 else 'poor'}). "
                f"See the 'Strategy Test' tab for full details."
            ) if metrics else "Enable **'Run Strategy Test'** in the sidebar to get risk information."
        elif any(k in q for k in ["anomaly", "unusual", "crash", "spike", "strange"]):
            n_anom = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
            response = (
                f"🚨 **Unusual Days for {symbol}**: The AI detected **{n_anom} unusual trading days** "
                f"where the price moved in an unexpected way. "
                f"These are shown as red ✕ marks on the price chart."
            )
        elif any(k in q for k in ["good", "bad", "buy", "sell", "worth", "invest", "recommend"]):
            response = (
                f"⚠️ **Stock Sentinel cannot make investment recommendations.** "
                f"This tool is for education only. We show you data and patterns "
                f"— but the decision is always yours. "
                f"Please consult a qualified financial advisor for real investment decisions."
            )
        else:
            response = (
                f"I can answer questions about: **price forecasts**, **news mood**, "
                f"**risk levels**, and **unusual price movements** for {symbol}. "
                f"Try asking: 'Is the price going up?', 'How risky is this stock?', "
                f"or 'What is the latest news mood?'"
            )
    except Exception:
        response = "Sorry, I couldn't process that question. Try asking differently."

    st.markdown(f"**🤖 Sentinel:** {response}")
