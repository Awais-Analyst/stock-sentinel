# 📈 Stock Sentinel

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Deploy: Render](https://img.shields.io/badge/Deploy-Render-purple)](https://render.com)

> ⚠️ **Disclaimer**: Stock Sentinel is an **educational and simulation** tool only.
> It is **NOT financial advice**. All forecasts and backtests use historical data.
> Past simulated performance does not guarantee future results.

---

## 🤔 Why This Project?

Stock Sentinel demonstrates a **complete end-to-end quantitative finance workflow** built
entirely with **free, open-source tools**:

```
Public Data → NLP Sentiment → ML Forecasting → Strategy Backtesting → Interactive Dashboard
```

It's designed for students and analysts who want to understand how modern quant tools
work under the hood — without needing expensive data subscriptions or proprietary libraries.

---

## 🗂️ Project Structure

```
stock/
├── stock_sentinel/
│   ├── app.py               # Streamlit dashboard (main entry point)
│   ├── data_pipeline.py     # Data fetching, cleaning, caching
│   ├── sentiment.py         # VADER NLP, word cloud, correlation
│   ├── modeling.py          # Prophet, LSTM, anomaly detection
│   ├── backtesting.py       # Signal generation, simulation, risk metrics
│   ├── utils.py             # Config, Plotly helpers, report generator
│   ├── test_full_pipeline.py# End-to-end smoke test
│   └── data/                # Cached CSVs (auto-created)
├── docs/images/             # Screenshots
├── requirements.txt
├── Procfile                 # Render deploy
├── runtime.txt              # Python 3.11 pin
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/stock-sentinel.git
cd stock-sentinel
pip install -r requirements.txt
```

Download NLTK data (one-time):

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

### 2. Set API Keys

Get your **free** API keys:

| Service | URL | Free Tier |
|---|---|---|
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day |
| NewsData.io | [newsdata.io](https://newsdata.io) | 200 credits/day |

Set them as environment variables (recommended):

```bash
# Windows PowerShell
$env:NEWS_API_KEY   = "your_newsapi_key_here"
$env:NEWSDATA_API_KEY = "your_newsdata_key_here"
```

Or enter them directly in the Streamlit sidebar at runtime.

### 3. Run the Dashboard

```bash
cd stock_sentinel
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Tests

```bash
cd stock_sentinel
python test_full_pipeline.py
```

---

## 🧩 Features

| Feature | Details |
|---|---|
| **Price Data** | yfinance OHLCV — auto-retry, CSV caching |
| **News Sentiment** | NewsAPI + NewsData.io fallback, VADER scoring |
| **Social Sentiment** | Best-effort X scraping (gracefully degrades to news-only) |
| **Technical Indicators** | RSI-14, MACD, Bollinger Bands, ATR |
| **Forecasting** | Prophet (with sentiment regressor) + optional LSTM |
| **Anomaly Detection** | IsolationForest on log returns |
| **Backtesting** | Sentiment-driven signals, equity curve, commissions |
| **Risk Metrics** | Sharpe Ratio, VaR (95%), Max Drawdown |
| **Portfolio Optimization** | PuLP min-variance (2–5 stocks) |
| **Dashboard** | Streamlit 3-tab UI, dark theme, Plotly charts |
| **Chat Interface** | Keyword-based query routing |

---

## 📡 Data Sources & Robustness (2026)

### yfinance
- `auto_adjust=True` for split/dividend-adjusted prices
- Exponential backoff retry (3 attempts) before failing
- Data cached to `data/<SYMBOL>_ohlcv.csv` — re-used on next run

### News
1. **Primary**: NewsAPI.org (~100 req/day, Developer free tier)
2. **Fallback**: NewsData.io (~200 credits/day)
3. If both fail: logs a warning, uses previous cached sentiment

### X / Twitter
Public X scraping is unreliable in 2026 (JS-heavy, bot detection).
The scraper is intentionally minimal (n≤15 posts) with:
- Random user-agent rotation
- 5–10 second sleep before requests
- Silent graceful failure — app continues news-only

**Alternative**: Download a pre-built Kaggle dataset for offline training:
- Search Kaggle: *"stock market tweets sentiment"*
- Drop the CSV into `data/` and it will be picked up automatically

---

## 🚀 Deploy to Render

1. Fork this repo on GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `streamlit run stock_sentinel/app.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variables: `NEWS_API_KEY`, `NEWSDATA_API_KEY`
6. Deploy → Render auto-deploys on every push to `main`

> **Note**: Free Render tier spins down after 15 min inactivity. First load may take ~30 seconds.

---

## ⚙️ Configuration

Edit `stock_sentinel/utils.py` to change defaults:

```python
STOCKS         = ["AAPL", "MSFT", "TSLA", "PSO.KA", "OGDC.KA"]
NEWS_SOURCE    = "auto"    # 'newsapi' | 'newsdata' | 'auto'
DEFAULT_START  = "2023-01-01"
DEFAULT_END    = "2024-12-31"
INITIAL_CAPITAL = 10_000
COMMISSION     = 0.001     # 0.1% per trade
```

---

## ⚠️ Known Limitations

- **X scraping**: Often returns empty results — use news-only mode for reliable sentiment
- **Pakistani stocks (PSO.KA, OGDC.KA)**: Sparse yfinance coverage; test early and cache
- **LSTM**: Slow to train on free hardware; use Prophet for quick runs
- **NewsAPI free tier**: No articles older than 1 month; historical sentiment will be sparse
- **PuLP optimizer**: Simplified variance model; not suitable for real portfolio management

---

## 📄 License

MIT — free for educational and non-commercial use.

---

## 🙏 Acknowledgements

Built with: [yfinance](https://github.com/ranaroussi/yfinance) · [Prophet](https://facebook.github.io/prophet/) · [VADER](https://github.com/cjhutto/vaderSentiment) · [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com) · [scikit-learn](https://scikit-learn.org) · [PuLP](https://coin-or.github.io/pulp/)
