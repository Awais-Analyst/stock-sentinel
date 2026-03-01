"""
app.py — Stock Sentinel Streamlit Dashboard.
Features: Price Charts, Sentiment, Backtest, AI Advisor (XAI), Multi-Language.
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Stock Sentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# LANGUAGE / TRANSLATION SYSTEM
# ─────────────────────────────────────────
LANG_OPTIONS = {
    "English 🇬🇧":  "en",
    "Urdu 🇵🇰":     "ur",
    "Arabic 🇸🇦":   "ar",
    "Chinese 🇨🇳":  "zh-CN",
    "French 🇫🇷":   "fr",
    "Hindi 🇮🇳":    "hi",
}

@st.cache_data(show_spinner=False, ttl=86400)
def translate(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator. Cached for 24h."""
    if target_lang == "en" or not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text  # fallback to English silently

def t(text: str) -> str:
    """Shorthand: translate to current UI language."""
    lang = st.session_state.get("ui_lang", "en")
    return translate(text, lang)


# ─────────────────────────────────────────
# OPTIONAL IMPORTS
# ─────────────────────────────────────────
def _try_import(*modules):
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            st.warning(f"Optional library '{mod}' not installed — some features disabled.")

_try_import("wordcloud", "prophet", "pulp", "shap")

from utils import (
    NEWS_API_KEY, NEWSDATA_API_KEY, NEWS_SOURCE, STOCKS,
    DEFAULT_START, DEFAULT_END, INITIAL_CAPITAL, COMMISSION,
    plot_candlestick, plot_forecast, plot_equity_curve,
    plot_sentiment_bars, generate_report_text,
)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Sentinel")
    st.caption(t("AI-powered stock analysis · For educational use only"))
    st.divider()

    # Language selector
    selected_lang_name = st.selectbox(
        "🌐 Interface Language",
        options=list(LANG_OPTIONS.keys()),
        index=0,
    )
    st.session_state["ui_lang"] = LANG_OPTIONS[selected_lang_name]

    symbol = st.text_input(
        t("Stock Symbol"), value="AAPL",
        placeholder=t("e.g. AAPL, TSLA, NBP.KA"),
        help=t("Pakistani stocks need .KA suffix (e.g. NBP.KA, MCB.KA)")
    ).upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(t("From"), value=pd.to_datetime(DEFAULT_START))
    with col2:
        end_date = st.date_input(t("To"), value=pd.to_datetime(DEFAULT_END))

    if start_date >= end_date:
        st.error(t("'From' date must be before 'To' date."))
        st.stop()

    if (end_date - start_date).days < 60:
        st.warning(t("Select at least 3 months for best results."))

    news_source  = st.selectbox(t("News Source"), ["auto", "newsapi", "newsdata"], index=0)
    newsapi_key  = st.text_input(t("NewsAPI Key"),  value=NEWS_API_KEY,  type="password")
    newsdata_key = st.text_input(t("NewsData Key"), value=NEWSDATA_API_KEY, type="password")

    st.divider()
    run_lstm          = st.toggle(t("Enable Deep Learning (slow)"), value=False)
    run_backtest_flag = st.toggle(t("Run Strategy Test"), value=True)
    run_advisor       = st.toggle(t("Run AI Advisor 🧠"), value=True,
                                   help=t("AI investment advice with probability scores"))
    refresh           = st.button(t("🔄 Refresh Data"), use_container_width=True)

    st.divider()
    st.markdown(f"<small>⚠️ {t('For educational use only. NOT financial advice.')}</small>",
                unsafe_allow_html=True)


# ─── Top banner ────────────────────────────────────────────────────────────
st.warning(
    f"⚠️ {t('Educational / Simulation Only')} — {t('Stock Sentinel is NOT financial advice. All forecasts are simulations using historical data.')}",
    icon="⚠️",
)
st.title(f"📊 {symbol} — Stock Sentinel")


# ─────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────
@st.cache_data(show_spinner=t("Fetching price & news data…"))
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
    cleaned_by_date = {date: [clean_text(tx) for tx in texts]
                       for date, texts in texts_by_date.items()}
    df = add_sentiment_to_df(df, cleaned_by_date)
    all_cleaned = [tx for texts in cleaned_by_date.values() for tx in texts]
    return df, cleaned_by_date, all_cleaned


@st.cache_resource(show_spinner=t("Running price forecast…"))
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


@st.cache_resource(show_spinner=t("Training deep learning model…"))
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
            pass
    return preds, train_m, test_m


@st.cache_resource(show_spinner="Training AI Advisor model...")
def load_advisor(df_hash, df_json):
    """Train RandomForest advisor model. Returns (model, feature_cols, accuracy, error_msg)."""
    try:
        from xai_advisor import train_advisor_model
    except ImportError:
        # xai_advisor.py not found on the server — tell the user exactly what to do
        return None, [], 0, "MODULE_MISSING"
    try:
        df = pd.read_json(io.StringIO(df_json))
        df.index = pd.to_datetime(df.index)
        model, cols, acc = train_advisor_model(df)
        return model, cols, acc, None
    except Exception as e:
        return None, [], 0, str(e)


# ─────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────
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

with st.spinner(t("Loading stock data…")):
    try:
        df, cleaned_by_date, all_texts = load_data(
            symbol, start_date, end_date,
            newsapi_key, newsdata_key, news_source, force_token,
        )
    except Exception as e:
        st.error(t(f"Could not load data for '{symbol}'. Check symbol and connection. Error: {e}"))
        st.stop()

if df is None or df.empty:
    st.error(
        t(f"No data found for '{symbol}' between {start_date} and {end_date}. "
          f"Pakistani stocks: use NBP.KA, MCB.KA format.")
    )
    st.stop()

try:
    from modeling import add_technical_indicators, add_lag_features, naive_baseline, detect_anomalies
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = detect_anomalies(df)
    df.dropna(inplace=True)
except Exception as e:
    st.error(t(f"Error computing indicators: {e}"))
    st.stop()

if df.empty or len(df) < 5:
    st.error(
        t(f"Not enough data for '{symbol}'. Select at least 3 months date range. "
          f"Pakistani stocks: NBP.KA, MCB.KA, OGDC.KA")
    )
    st.stop()

# Safe defaults — these prevent NameError in chat box if tabs haven't run
metrics    = {}
horizon    = 30
lstm_preds = None
forecast   = None
advice     = {}
action     = "HOLD"
confidence = 50
prob_up    = 50
risk       = "Moderate"
health     = 50

# Build JSON for ML models
try:
    df_cols = [c for c in ["Open","High","Low","Close","Volume","sentiment"] if c in df.columns]
    df_json = df[df_cols].to_json()
    df_hash = hash(df_json[:500])
except Exception:
    df_json = df[["Close"]].to_json()
    df_hash = hash(df_json[:500])


# ─────────────────────────────────────────
# QUICK SUMMARY CARD
# ─────────────────────────────────────────
price_change_pct = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
trend_label = ("📈 " + t("Up")) if price_change_pct >= 0 else ("📉 " + t("Down"))
mean_sentiment = float(df["sentiment"].mean()) if "sentiment" in df.columns else 0.0
if   mean_sentiment >  0.05: mood_label = "🟢 " + t("Positive")
elif mean_sentiment < -0.05: mood_label = "🔴 " + t("Negative")
else:                         mood_label = "⚪ " + t("Neutral")

st.markdown(f"### {t('Quick Summary')}")
s1, s2, s3, s4 = st.columns(4)
s1.metric(t("Price Trend"),    trend_label, f"{price_change_pct:+.1f}%")
s2.metric(t("Latest Price"),   f"${df['Close'].iloc[-1]:.2f}")
s3.metric(t("News Mood"),      mood_label)
s4.metric(t("Unusual Days"),   f"{int(df['anomaly'].sum())}" if "anomaly" in df.columns else "N/A")

with st.expander(f"💡 {t('What does this summary mean?')}"):
    st.markdown(t("""
| Term | Plain explanation |
|---|---|
| **Price Trend** | Did the stock go up or down over your selected dates? |
| **Latest Price** | The last closing price in your date range |
| **News Mood** | Were recent news headlines mostly good or bad? |
| **Unusual Days** | AI-detected days when prices moved unexpectedly |
"""))

st.divider()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    f"📈 {t('Price Chart')}",
    f"📰 {t('News Mood')}",
    f"🎯 {t('Strategy Test')}",
    f"🧠 {t('AI Advisor')}",
])


# ═══════════════════════════════════════
# TAB 1 — PRICE CHART & FORECAST
# ═══════════════════════════════════════
with tab1:
    st.subheader(t(f"{symbol} — Price History"))

    with st.expander(f"💡 {t('How to read the price chart')}"):
        st.markdown(t("""
- **Green candles** = the price went UP that day
- **Red candles** = the price went DOWN that day
- **Red ✕ marks** = unusual/unexpected price movements detected by AI
- **Bottom panel** = how many shares were traded (higher = more activity)
        """))

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
                    name=t("Unusual Day"),
                    hovertemplate=f"<b>{t('Unusual Day')}</b><br>%{{x|%b %d, %Y}}<br>{t('Price')}: $%{{y:.2f}}<extra></extra>",
                ), row=1, col=1)
        st.plotly_chart(fig_candle, use_container_width=True)
    except Exception as e:
        st.warning(t(f"Could not draw price chart: {e}"))

    c1, c2, c3, c4 = st.columns(4)
    try:
        c1.metric(t("Latest Price"),    f"${df['Close'].iloc[-1]:.2f}",  help=t("Last closing price"))
        c2.metric(t("Highest Price"),   f"${df['Close'].max():.2f}",     help=t("Highest close in period"))
        c3.metric(t("Lowest Price"),    f"${df['Close'].min():.2f}",     help=t("Lowest close in period"))
        c4.metric(t("Unusual Days"),    f"{int(df['anomaly'].sum())}" if "anomaly" in df.columns else "N/A",
                  help=t("AI-detected days with surprising price moves"))
    except Exception as e:
        st.warning(t(f"Error computing stats: {e}"))

    st.divider()
    st.subheader(t(f"{symbol} — Price Forecast"))

    with st.expander(f"💡 {t('How to read the forecast chart')}"):
        st.markdown(t("""
- **Solid green line** = actual historical price
- **Orange dashed line** = AI's best guess for future price
- **Golden shaded band** = range where AI thinks price will likely fall (80% confident)
- **Grey dotted line** = simple "if tomorrow = today" benchmark
> Wider band = more uncertain prediction. This is only a model output, not a guarantee!
        """))

    # ── Data quality warning for forecast ──────────────────────────────────
    trading_days = len(df)
    if trading_days < 60:
        st.error(
            f"⚠️ **Only {trading_days} trading days of data** — the AI forecast needs at least **60 days** (3 months) to work. "
            f"The orange forecast line and golden confidence band will NOT appear. "
            f"👉 **Fix:** Change the 'From' date to at least 3 months ago."
        )
    elif trading_days < 120:
        st.warning(
            f"⚠️ **Only {trading_days} trading days** selected. The forecast may be unreliable. "
            f"For a full forecast with confidence bands, select **6+ months** of data. "
            f"With short ranges, the AI doesn't have enough history to detect price patterns."
        )

    horizon = st.slider(t("How many days to forecast?"), 7, 90, 30)


    with st.spinner(t("Running AI price forecast…")):
        try:
            forecast = load_prophet(df_hash, df_json, horizon=horizon)
        except Exception as e:
            st.warning(t(f"Forecast failed: {e}"))

    try:
        baseline = naive_baseline(df)
    except Exception:
        baseline = None

    lstm_preds = None
    if run_lstm:
        with st.spinner(t("Training deep learning model (2-3 min)…")):
            try:
                lstm_preds, train_m, test_m = load_lstm(df_hash, df_json)
                if lstm_preds is not None:
                    rmse     = test_m.get("rmse", None)
                    r2       = test_m.get("r2", None)
                    rmse_str = f"${rmse:.2f}" if isinstance(rmse, float) else "N/A"
                    r2_str   = f"{r2:.1%}"    if isinstance(r2,   float) else "N/A"
                    st.success(t(f"Deep learning trained! Avg error: {rmse_str}, Accuracy: {r2_str}"))
            except Exception as e:
                st.warning(t(f"Deep learning model failed: {e}"))

    try:
        fig_fc = plot_forecast(df, forecast, symbol=symbol, baseline=baseline)
        if lstm_preds is not None:
            import plotly.graph_objects as go
            fig_fc.add_trace(go.Scatter(
                x=lstm_preds.index, y=lstm_preds.values,
                name=t("Deep Learning Prediction"),
                line=dict(color="#e056fd", width=1.5, dash="dash"),
            ))
        st.plotly_chart(fig_fc, use_container_width=True)
    except Exception as e:
        st.warning(t(f"Could not draw forecast: {e}"))

    with st.expander(f"📐 {t('Advanced Indicators (for experienced users)')}"):
        st.markdown(t("""
| Indicator | What it measures | How to read |
|---|---|---|
| **RSI (14)** | Is stock overbought or oversold? | >70 may drop, <30 may rise |
| **MACD** | Is momentum increasing? | Positive = uptrend |
| **Bollinger Upper/Lower** | Statistical price boundaries | Near upper = expensive, Near lower = cheap |
| **ATR (14)** | Daily price swing size | Higher = more volatile/risky |
        """))
        try:
            ind_cols = [c for c in ["rsi_14","macd","boll_upper","boll_lower","atr_14"] if c in df.columns]
            if ind_cols:
                st.dataframe(df[ind_cols].tail(20).round(2), use_container_width=True)
        except Exception as e:
            st.warning(t(f"Indicator table error: {e}"))


# ═══════════════════════════════════════
# TAB 2 — NEWS MOOD
# ═══════════════════════════════════════
with tab2:
    st.subheader(t(f"{symbol} — News & Social Media Mood"))

    # ── Explain the date split ──────────────────────────────────────────────
    from datetime import datetime as _dt2, timedelta as _td2
    news_to_date   = _dt2.utcnow().strftime("%b %d, %Y")
    news_from_date = (_dt2.utcnow() - _td2(days=30)).strftime("%b %d, %Y")
    user_range_old = end_date < (_dt2.utcnow() - _td2(days=30)).date()

    with st.expander(f"📅 {t('Why are two different date ranges used?')} ← {t('Read this first')}"):
        st.markdown(f"""
**Price chart & forecast** → uses **your selected dates** ({start_date} to {end_date})
> Historical stock data is freely available for any past period via Yahoo Finance.

**News mood** → always uses the **last 30 days from today** ({news_from_date} → {news_to_date})
> Free news APIs (NewsAPI.org, NewsData.io) only provide recent news — older articles require a paid subscription.

**These do NOT conflict:**
- You can select **2024/01/01 – 2026/12/31** and get: 2 years of price charts **+** last 30 days of news
- The news chart will simply show the 30 most recent days, not your full price range
        """)

    # Warn if user's date range is completely in the past (outside news coverage)
    if user_range_old:
        st.warning(
            f"⚠️ Your selected date range ends on **{end_date}**, which is more than 30 days ago. "
            f"Free news APIs only cover **{news_from_date} → {news_to_date}** (last 30 days). "
            f"News mood will show recent news only, not news from your selected price period."
        )

    st.info(f"📰 **News coverage: {news_from_date} → {news_to_date}** (last 30 days — free API limit)")

    with st.expander(f"💡 {t('What is News Mood?')}"):
        st.markdown(t("""
We read news headlines and decide: is the tone **good** (positive) or **bad** (negative)?
Score: **-1.0** = very negative, **0** = neutral, **+1.0** = very positive.
This sometimes predicts whether a stock will go up or down.
        """))

    has_api_key    = bool(newsapi_key and newsapi_key not in ("", "YOUR_NEWSAPI_KEY"))
    has_nd_key     = bool(newsdata_key and newsdata_key not in ("", "YOUR_NEWSDATA_KEY"))
    has_any_key    = has_api_key or has_nd_key
    has_sentiment  = "sentiment" in df.columns and df["sentiment"].abs().max() > 0.001
    total_articles = sum(len(v) for v in cleaned_by_date.values()) if cleaned_by_date else 0

    if not has_sentiment:
        if not has_any_key:
            st.info(
                "📰 **Add a free API key to see News Mood.**\n\n"
                "1. Go to [newsapi.org](https://newsapi.org) → click **Get API Key** (free)\n"
                "2. Paste the key in the **NewsAPI Key** field in the sidebar\n"
                "3. Click **🔄 Refresh Data**"
            )
        elif total_articles == 0:
            st.warning(
                f"⚠️ API key is set but returned **0 articles** for '{symbol}'.\n\n"
                "**Fix — try these steps:**\n"
                "1. Check your email and **activate** your NewsAPI key\n"
                "2. Free plan covers only **last 30 days** — your key must be active NOW\n"
                "3. Try symbol **AAPL** or **TSLA** to verify the key works\n"
                "4. Click **🔄 Refresh Data**\n\n"
                f"*(News period being searched: {news_from_date} → {news_to_date})*"
            )
        else:
            st.warning(
                f"⚠️ Got {total_articles} news articles but dates didn't match trading days. "
                "Click **🔄 Refresh Data** to retry with the fixed date matching."
            )


    else:
        try:
            fig_sent = plot_sentiment_bars(df["sentiment"], symbol=symbol)
            st.plotly_chart(fig_sent, use_container_width=True)
        except Exception as e:
            st.warning(t(f"Mood chart error: {e}"))

        c1, c2, c3 = st.columns(3)
        mood_word = t("Generally Positive 🟢") if mean_sentiment > 0.05 else (t("Generally Negative 🔴") if mean_sentiment < -0.05 else t("Neutral ⚪"))
        c1.metric(t("Overall News Mood"),  mood_word)
        c2.metric(t("Best Day Score"),     f"{df['sentiment'].max():.2f}")
        c3.metric(t("Worst Day Score"),    f"{df['sentiment'].min():.2f}")

        st.divider()
        col_wc, col_kw = st.columns([2, 1])
        with col_wc:
            st.markdown(f"**{t('Most Talked-About Words')}**")
            if all_texts:
                try:
                    from sentiment import generate_wordcloud
                    wc_img = generate_wordcloud(all_texts)
                    if wc_img:
                        st.image(wc_img, use_container_width=True)
                except Exception as e:
                    st.warning(t(f"Word cloud error: {e}"))
            else:
                st.info(t("No text data. Add NewsAPI key for real headlines."))

        with col_kw:
            st.markdown(f"**{t('Top Keywords')}**")
            if all_texts:
                try:
                    from sentiment import get_top_keywords
                    kw_df = pd.DataFrame(get_top_keywords(all_texts, n=15), columns=[t("Word"), t("Times Mentioned")])
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(t(f"Keywords error: {e}"))

        st.divider()
        st.markdown(f"**{t('Does News Mood Affect Price?')}**")
        with st.expander(f"💡 {t('How to read the correlation table')}"):
            st.markdown(t("Near +1.0 = good news → price up. Near -1.0 = good news → price down. Near 0 = no clear relationship."))
        try:
            from sentiment import sentiment_price_correlation
            corr = sentiment_price_correlation(df)
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 color_continuous_scale="RdYlGn",
                                 title=t("Relationship: News Mood ↔ Price"))
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#444444"))
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(t(f"Correlation error: {e}"))


# ═══════════════════════════════════════
# TAB 3 — STRATEGY TEST
# ═══════════════════════════════════════
with tab3:
    st.subheader(t(f"{symbol} — AI Strategy Test"))

    with st.expander(f"💡 {t('What is a Strategy Test?')}"):
        st.markdown(t("""
We give the AI **$10,000 of pretend money** and let it trade using historical data.
It buys when news is positive + price predicted to rise, sells when negative.
Then we check: **did the AI beat buy-and-hold?**
> ⚠️ This is a simulation, NOT real trading advice.
        """))

    if not run_backtest_flag:
        st.info(t("Enable 'Run Strategy Test' in the sidebar to see results."))
    else:
        try:
            from backtesting import generate_signals, run_backtest, calc_all_metrics, generate_insights

            pred_col = "lstm_pred" if lstm_preds is not None else "log_return"
            if lstm_preds is not None:
                try:
                    df["lstm_pred"] = lstm_preds.reindex(df.index)
                except Exception:
                    pred_col = "log_return"

            with st.spinner(t("Running strategy test…")):
                signals      = generate_signals(df, pred_col=pred_col)
                portfolio_df = run_backtest(df, signals, capital=INITIAL_CAPITAL, commission=COMMISSION)
                metrics      = calc_all_metrics(portfolio_df, capital=INITIAL_CAPITAL)

            try:
                fig_eq = plot_equity_curve(portfolio_df, symbol=symbol)
                st.plotly_chart(fig_eq, use_container_width=True)
            except Exception as e:
                st.warning(t(f"Equity curve error: {e}"))

            with st.expander(f"💡 {t('How to read the portfolio chart')}"):
                st.markdown(t("""
- **Green line** = value if following AI's buy/sell signals
- **Grey dotted line** = value if you just bought and held
- Green above grey = AI strategy performed better
                """))

            st.markdown(f"### {t('Results in Simple Terms')}")
            c1, c2, c3, c4, c5 = st.columns(5)
            ret    = metrics.get("total_return_pct", 0)
            sharpe = metrics.get("sharpe", 0)
            var    = metrics.get("var_95", 0)
            dd     = metrics.get("max_drawdown", 0)

            c1.metric(t("Total Profit/Loss"),     f"{ret:+.1f}%",
                      help=t("Did the AI make or lose money overall?"))
            c2.metric(t("Profit vs Risk Score"),  f"{sharpe:.2f}",
                      help=t("Above 1.0 is good. Measures return vs risk taken."))
            c3.metric(t("Worst Day Loss"),         f"{abs(var):.2%}",
                      help=t("Estimated worst daily loss with 95% confidence."))
            c4.metric(t("Biggest Drop from Peak"), f"{abs(dd):.2%}",
                      help=t("Worst loss from the highest point to the lowest."))
            c5.metric(t("Number of Trades"),       f"{metrics.get('num_trades', 0)}",
                      help=t("Each trade has a 0.1% fee."))

            with st.expander(f"📋 {t('When did the AI Buy/Sell?')}"):
                try:
                    sig_df = pd.DataFrame({"Signal": signals, t("Price"): df["Close"]})
                    sig_df[t("Action")] = sig_df["Signal"].map({1: f"🟢 {t('BUY')}", -1: f"🔴 {t('SELL')}", 0: f"⏸ {t('HOLD')}"})
                    st.dataframe(sig_df[sig_df["Signal"] != 0][[t("Action"), t("Price")]].tail(30), use_container_width=True)
                except Exception as e:
                    st.warning(t(f"Signal table error: {e}"))

            st.divider()
            st.markdown(f"**📝 {t('Strategy Summary')}**")
            try:
                st.info(generate_insights(metrics, symbol=symbol))
            except Exception as e:
                st.warning(t(f"Summary error: {e}"))

            # Portfolio optimizer
            st.divider()
            st.markdown(f"**🎯 {t('Portfolio Optimizer — Best Stock Mix')}**")
            with st.expander(f"💡 {t('What is this?')}"):
                st.markdown(t("Spread your money across multiple stocks to reduce risk. Like not putting all eggs in one basket."))

            st.caption(t("Type any stock symbols, separated by commas. Pakistani stocks need .KA suffix (e.g. NBP.KA, MCB.KA)"))
            stocks_input = st.text_input(
                t("Enter 2–10 stock symbols:"),
                value="AAPL, MSFT, TSLA",
                placeholder="e.g. AAPL, MSFT, NBP.KA, MCB.KA, OGDC.KA",
                key="portfolio_symbols_input",
            )

            # Parse user input into a clean list
            selected_stocks = [s.strip().upper() for s in stocks_input.split(",") if s.strip()]
            selected_stocks = list(dict.fromkeys(selected_stocks))  # deduplicate

            if len(selected_stocks) > 10:
                st.warning(t("Maximum 10 stocks. Only the first 10 will be used."))
                selected_stocks = selected_stocks[:10]

            if len(selected_stocks) < 2:
                st.info(t("Enter at least 2 stock symbols separated by commas."))

            if st.button(t("Find Best Mix"), key="opt_btn") and len(selected_stocks) >= 2:
                from backtesting import optimize_portfolio
                from data_pipeline import get_stock_data
                all_returns, failed = {}, []
                progress = st.progress(0, text=t("Fetching stock data…"))
                for i, s in enumerate(selected_stocks):
                    progress.progress((i + 1) / len(selected_stocks), text=t(f"Loading {s}…"))
                    try:
                        s_df = get_stock_data(s, str(start_date), str(end_date))
                        if not s_df.empty and "Close" in s_df.columns:
                            all_returns[s] = s_df["Close"].pct_change().dropna()
                        else:
                            failed.append(s)
                    except Exception:
                        failed.append(s)
                progress.empty()

                if failed:
                    st.warning(t(f"Could not load data for: {', '.join(failed)}. Check symbols on Yahoo Finance."))
                if len(all_returns) >= 2:
                    try:
                        returns_df_multi = pd.DataFrame(all_returns).dropna()
                        if returns_df_multi.empty:
                            st.error(t("Not enough overlapping data. Try a longer date range."))
                        else:
                            weights = optimize_portfolio(returns_df_multi, target_return=0.0005)
                            wt_df = pd.DataFrame(list(weights.items()), columns=[t("Stock"), t("Recommended %")])
                            wt_df[t("Recommended %")] = (wt_df[t("Recommended %")] * 100).round(1)
                            wt_df = wt_df.sort_values(t("Recommended %"), ascending=False)
                            col_pie, col_tbl = st.columns([3, 2])
                            with col_pie:
                                fig_pie = px.pie(
                                    wt_df, names=t("Stock"), values=t("Recommended %"),
                                    title=t("Suggested Portfolio Split"),
                                    color_discrete_sequence=px.colors.qualitative.Safe,
                                )
                                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#444444"))
                                st.plotly_chart(fig_pie, use_container_width=True)
                            with col_tbl:
                                st.markdown(f"**{t('Allocation')}**")
                                st.dataframe(wt_df, use_container_width=True, hide_index=True)
                                st.caption(t("Put these % of your budget in each stock to minimize risk."))
                    except Exception as e:
                        st.error(t(f"Optimizer failed: {e}"))
                else:
                    st.error(t("None of the symbols returned data. Check symbols and try a longer date range."))

        except Exception as e:
            st.error(t(f"Strategy test failed: {e}"))


# ═══════════════════════════════════════
# TAB 4 — AI ADVISOR (XAI)
# ═══════════════════════════════════════
with tab4:
    st.subheader(t(f"{symbol} — AI Investment Advisor 🧠"))
    st.caption(t("Powered by Explainable AI (SHAP) — every recommendation comes with a full explanation."))

    with st.expander(f"💡 {t('What is Explainable AI?')}"):
        st.markdown(t("""
Normal AI says: *"The price will go up."*
**Explainable AI says**: *"The price will go up BECAUSE:*
*news mood is positive (+40%), RSI shows oversold condition (+30%), MACD is rising (+20%)"*

You can see **why**, not just **what**. This makes the advice trustworthy and educational.
        """))

    if not run_advisor:
        st.info(t("Enable 'Run AI Advisor' in the sidebar to see recommendations."))
    else:
        with st.spinner(t("Training AI advisor model...")):
            try:
                advisor_model, feature_cols, accuracy, advisor_err = load_advisor(df_hash, df_json)
            except Exception as e:
                advisor_model, feature_cols, accuracy, advisor_err = None, [], 0, str(e)

        # Handle missing module clearly
        if advisor_err == "MODULE_MISSING":
            st.error(
                t("**xai_advisor.py file is missing on the server.**") + "\n\n" +
                t("This new file needs to be uploaded to GitHub. Steps:") + "\n"
                "1. Go to your GitHub repo → `stock_sentinel/` folder\n"
                "2. Click **Add file** → **Create new file**\n"
                "3. Name it: `stock_sentinel/xai_advisor.py`\n"
                "4. Paste the contents from your local `d:/Projects/stock/stock_sentinel/xai_advisor.py`\n"
                "5. Commit — Streamlit will redeploy automatically"
            )
        elif advisor_err:
            st.warning(t(f"AI Advisor setup error: {advisor_err}"))

        if not advisor_model or not feature_cols:
            if not advisor_err:
                st.warning(t("Not enough data to train the AI advisor. Select a longer date range (6+ months recommended)."))
        else:
            st.success(t(f"AI Advisor trained on {len(df)} days of data. Model accuracy: {accuracy}%"))

            # Get advice
            try:
                from xai_advisor import get_investment_advice, get_shap_explanation, build_advice_narrative

                advice = get_investment_advice(
                    advisor_model, df, feature_cols, forecast_df=forecast
                )
                action     = advice.get("action", "HOLD")
                confidence = advice.get("confidence", 50)
                prob_up    = advice.get("probability_up", 50)
                risk       = advice.get("risk_level", "Moderate")
                health     = advice.get("health_score", 50)
                reasons    = advice.get("reasons", [])

                # ── Main recommendation badge ─────────────────────────
                action_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                action_emoji  = action_colors.get(action, "⚪")
                action_labels = {
                    "BUY":  t("CONSIDER BUYING"),
                    "SELL": t("CONSIDER SELLING"),
                    "HOLD": t("HOLD / WAIT"),
                }

                st.markdown(f"## {action_emoji} {t('AI Recommendation:')} **{action_labels.get(action, action)}**")

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric(
                    t("Confidence"),
                    f"{confidence:.0f}%",
                    help=t("How confident is the AI in this recommendation?")
                )
                col_b.metric(
                    t("Chance Price Rises (5 days)"),
                    f"{prob_up:.0f}%",
                    delta=f"{prob_up-50:.0f}% vs 50/50",
                    help=t("Probability the price will be higher 5 trading days from now")
                )
                col_c.metric(
                    t("Stock Health Score"),
                    f"{health}/100",
                    help=t("0=very weak, 50=neutral, 100=very healthy. Based on RSI, MACD, sentiment, momentum.")
                )
                col_d.metric(
                    t("Risk Level"),
                    risk,
                    help=t("Based on how much the price swings daily (ATR indicator)")
                )

                # Health score visual bar
                st.markdown(f"**{t('Stock Health Score Visual')}**")
                health_color = "#00c48c" if health >= 60 else ("#f5a623" if health >= 40 else "#e84545")
                st.markdown(
                    f'<div style="background:#eee;border-radius:8px;height:20px;width:100%;">'
                    f'<div style="background:{health_color};width:{health}%;height:20px;border-radius:8px;'
                    f'display:flex;align-items:center;padding-left:8px;color:white;font-size:12px;">'
                    f'{health}/100</div></div>',
                    unsafe_allow_html=True
                )
                st.markdown("")

                st.divider()

                # ── Probability breakdown ─────────────────────────────
                st.markdown(f"### 📊 {t('Probability Breakdown')}")
                prob_df = pd.DataFrame({
                    t("Outcome"):     [t("Price Goes Up 📈"), t("Price Goes Down 📉")],
                    t("Probability"): [prob_up, 100 - prob_up],
                })
                fig_prob = px.bar(
                    prob_df, x=t("Probability"), y=t("Outcome"),
                    orientation="h",
                    color=t("Outcome"),
                    color_discrete_map={
                        t("Price Goes Up 📈"):   "#00c48c",
                        t("Price Goes Down 📉"): "#e84545",
                    },
                    title=t("Next 5-Day Price Direction Probability"),
                    text=t("Probability"),
                )
                fig_prob.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, xaxis_range=[0, 100],
                    font=dict(color="#444444"),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                st.divider()

                # ── SHAP Feature Contributions ────────────────────────
                st.markdown(f"### 🔍 {t('Why is the AI saying this?')} ({t('SHAP Explanation')})")

                with st.expander(f"💡 {t('How to read this chart')}"):
                    st.markdown(t("""
- **Green bars** = this factor is pushing the prediction **toward BUY** (positive)
- **Red bars** = this factor is pushing the prediction **toward SELL** (negative)
- **Longer bar** = stronger influence on the AI's decision
                    """))

                try:
                    shap_vals = get_shap_explanation(advisor_model, df, feature_cols)
                    if shap_vals:
                        shap_df = pd.DataFrame({
                            t("Factor"):      list(shap_vals.keys()),
                            t("Influence"):   list(shap_vals.values()),
                        })
                        shap_df[t("Direction")] = shap_df[t("Influence")].apply(
                            lambda v: f"🟢 {t('Supports BUY')}" if v > 0 else f"🔴 {t('Supports SELL')}"
                        )
                        fig_shap = px.bar(
                            shap_df.head(8),
                            x=t("Influence"), y=t("Factor"),
                            orientation="h",
                            color=t("Direction"),
                            color_discrete_map={
                                f"🟢 {t('Supports BUY')}":  "#00c48c",
                                f"🔴 {t('Supports SELL')}": "#e84545",
                            },
                            title=t("What is driving the AI's recommendation?"),
                        )
                        fig_shap.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False, font=dict(color="#444444"),
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info(t("SHAP explanation not available — install 'shap' package."))
                except Exception as e:
                    st.warning(t(f"SHAP chart error: {e}. Install: pip install shap"))

                st.divider()

                # ── Plain-English reasons ─────────────────────────────
                st.markdown(f"### 💬 {t('Reason-by-Reason Explanation')}")
                for i, reason in enumerate(reasons, 1):
                    st.markdown(f"**{i}.** {t(reason)}")

                st.divider()

                # ── Full narrative ────────────────────────────────────
                st.markdown(f"### 📝 {t('Complete AI Report')}")
                try:
                    narrative = build_advice_narrative(advice, symbol)
                    st.info(t(narrative))
                except Exception as e:
                    st.warning(t(f"Narrative error: {e}"))

                # ── Final disclaimer ──────────────────────────────────
                st.error(
                    f"⚠️ **{t('IMPORTANT DISCLAIMER')}**: {t('This is an AI-generated analysis for educational purposes ONLY. It is NOT financial advice. The AI makes mistakes and past patterns do not guarantee future results. Always consult a qualified financial advisor before making any real investment decisions.')}"
                )

            except Exception as e:
                st.error(t(f"AI Advisor failed to generate advice: {e}"))


# ─────────────────────────────────────────
# CHAT BOX (bottom)
# ─────────────────────────────────────────
st.divider()
st.markdown(f"### 💬 {t('Ask Stock Sentinel')}")

user_query = st.text_input(
    t("Ask anything about this stock in plain language:"),
    key="chat_input",
    placeholder=t("e.g. 'Is AAPL doing well?' or 'Should I be worried?'"),
)

if user_query:
    q = user_query.lower()
    try:
        if any(k in q for k in ["predict","forecast","future","price","up","down"]):
            direction = t("upward") if forecast is not None and not forecast.empty and forecast["yhat"].iloc[-1] > df["Close"].iloc[-1] else t("downward")
            resp = t(f"The AI forecast shows a {direction} trend over the next {horizon} days. See the orange dashed line in the Price Chart tab. Remember: predictions are never 100% accurate!")
        elif any(k in q for k in ["news","mood","sentiment","feel"]):
            mood = t("positive") if mean_sentiment > 0.05 else (t("negative") if mean_sentiment < -0.05 else t("neutral"))
            resp = t(f"The news mood for {symbol} is {mood} (score: {mean_sentiment:.2f}). Check the News Mood tab for day-by-day details.")
        elif any(k in q for k in ["buy","sell","invest","recommend","should i"]):
            if metrics or run_advisor:
                action_now = advice.get("action","HOLD") if "advice" in dir() else "HOLD"
                prob       = advice.get("probability_up", 50) if "advice" in dir() else 50
                resp = t(f"The AI recommends: {action_now} (confidence: {confidence:.0f}%, probability of price rise: {prob:.0f}%). See the AI Advisor tab for full explanation. ⚠️ This is NOT real financial advice!")
            else:
                resp = t("Enable 'Run AI Advisor' in the sidebar to get a buy/sell recommendation.")
        elif any(k in q for k in ["risk","safe","dangerous","loss"]):
            risk_now = advice.get("risk_level","Moderate") if "advice" in dir() else t("unknown")
            resp = t(f"Current risk level for {symbol}: {risk_now}. See the AI Advisor tab for detailed risk breakdown.")
        elif any(k in q for k in ["anomaly","unusual","crash","spike"]):
            n = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
            resp = t(f"The AI detected {n} unusual trading days for {symbol}. They appear as red ✕ marks on the price chart.")
        else:
            resp = t(f"I can answer questions about: price forecasts, news mood, buy/sell recommendations, risk, and unusual price movements for {symbol}. Try: 'Is the price going up?' or 'How risky is this?'")
    except Exception:
        resp = t("Sorry, I couldn't process that. Try rephrasing your question.")

    st.markdown(f"**🤖 Sentinel:** {resp}")
