# 📈 Stock Sentinel — AI-Powered Stock Analysis Platform

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Stock_Sentinel-00c48c?style=for-the-badge)](https://stock-sentinel-gj7grodjh3bzf7w3jjpu6s.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Stock Sentinel** is a full-stack AI stock analysis dashboard that combines **real-time news sentiment analysis**, **ML price forecasting**, **strategy backtesting**, and **explainable AI investment advice** — all in one interactive platform.

🔗 **[Live Demo](https://stock-sentinel-gj7grodjh3bzf7w3jjpu6s.streamlit.app/)** · 📂 **[Source Code](https://github.com/Awais-Analyst/stock-sentinel)**

> ⚠️ **Disclaimer**: Stock Sentinel is an **educational and simulation** tool only.
> It is **NOT financial advice**. All forecasts and backtests use historical data.
> Past simulated performance does not guarantee future results.

---

## ✨ Key Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| 📊 **Interactive Price Charts** | Plotly, yfinance | Real-time candlestick charts with volume, anomaly detection, and technical indicators |
| 📰 **News Sentiment Analysis** | VADER NLP, NewsAPI | Automated news fetching with relevance filtering + sentiment scoring |
| 🔮 **AI Price Forecasting** | Prophet, LSTM | Machine learning price predictions with confidence intervals |
| 🎯 **Strategy Backtesting** | Custom Engine | Simulate trading strategies with $10K virtual capital, commissions, and risk metrics |
| 🧠 **Explainable AI Advisor** | RandomForest, SHAP | Investment recommendations with probability scores and feature explanations |
| 🎨 **Portfolio Optimizer** | PuLP | Minimum-variance portfolio allocation across multiple stocks |
| 🌍 **Global Stock Support** | Yahoo Finance | Analyze stocks from USA, UK, Germany, Japan, Saudi Arabia, Pakistan, India, Hong Kong, and more |
| 🌐 **Multi-Language** | deep-translator | Interface available in English, Urdu, Arabic, Chinese, French, Hindi |
| 💬 **AI Chat Assistant** | NLP | Ask questions about stocks in plain language |

---

## 🖥️ Screenshots

### Dark Mode
*Premium glassmorphism UI with gradient sidebar, interactive Plotly charts, and AI-powered insights.*
<img width="1912" height="898" alt="image" src="https://github.com/user-attachments/assets/9a0efdb3-ffae-4ac8-a6b8-c4cb6d9f7cf6" />


### Light Mode
*Fully adaptive design — readable on both dark and light backgrounds.*
<img width="1920" height="877" alt="image" src="https://github.com/user-attachments/assets/8909f46c-30cf-4e34-8478-64fa1d36a2d5" />


---

## 🏗️ Architecture

```
📊 User Interface (Streamlit + Premium CSS)
    ├── 📈 Price Chart & Forecast (Tab 1)
    ├── 📰 News Mood & Sentiment (Tab 2)
    ├── 🎯 Strategy Backtesting (Tab 3)
    └── 🧠 AI Advisor with SHAP (Tab 4)

🔧 Backend Pipeline
    ├── data_pipeline.py    → Data fetching (yfinance + NewsAPI + NewsData.io)
    ├── sentiment.py        → VADER NLP scoring + word clouds
    ├── modeling.py         → Prophet + LSTM + anomaly detection
    ├── backtesting.py      → Signal generation + portfolio simulation
    ├── xai_advisor.py      → RandomForest + SHAP explanations
    └── utils.py            → Config + Plotly chart themes
```

---

## 🗂️ Project Structure

```
stock-sentinel/
├── stock_sentinel/
│   ├── app.py                 # Main Streamlit dashboard
│   ├── data_pipeline.py       # Data fetching, cleaning, caching
│   ├── sentiment.py           # VADER NLP, word cloud, correlation
│   ├── modeling.py            # Prophet, LSTM, anomaly detection
│   ├── backtesting.py         # Signal generation, simulation, risk metrics
│   ├── xai_advisor.py         # Explainable AI advisor (SHAP)
│   ├── utils.py               # Config, Plotly helpers, report generator
│   ├── test_full_pipeline.py  # End-to-end smoke test
│   └── data/                  # Cached CSVs (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Awais-Analyst/stock-sentinel.git
cd stock-sentinel
pip install -r requirements.txt
```

### 2. Set API Keys (Optional — for News Sentiment)

Get your **free** API keys:

| Service | URL | Free Tier |
|---------|-----|-----------|
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day |
| NewsData.io | [newsdata.io](https://newsdata.io) | 200 credits/day |

Set them as environment variables:

```bash
# Windows PowerShell
$env:NEWS_API_KEY     = "your_newsapi_key"
$env:NEWSDATA_API_KEY = "your_newsdata_key"
```

Or enter them directly in the sidebar at runtime.

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

## 🌍 Tested With Global Stocks

Stock Sentinel has been tested with **10 stocks from 7 regions**:

| Stock | Region | Data | Backtest | AI Advisor |
|-------|--------|------|----------|------------|
| AAPL | 🇺🇸 USA | ✅ | ✅ 146 trades | ✅ 54.3% |
| TSLA | 🇺🇸 USA | ✅ | ✅ 155 trades | ✅ 48.6% |
| 2222.SR | 🇸🇦 Saudi Arabia | ✅ | ✅ 132 trades | ✅ 32.4% |
| SHEL.L | 🇬🇧 UK | ✅ | ✅ 147 trades | ✅ 60.0% |
| RELIANCE.NS | 🇮🇳 India | ✅ | ✅ 135 trades | ✅ 47.1% |
| SAP.DE | 🇩🇪 Germany | ✅ | ✅ 142 trades | ✅ 62.5% |
| 7203.T | 🇯🇵 Japan | ✅ | ✅ 150 trades | ✅ 54.5% |
| 0700.HK | 🇭🇰 Hong Kong | ✅ | ✅ 139 trades | ✅ 42.4% |
| NBP.KA | 🇵🇰 Pakistan | ✅ | ✅ 153 trades | ✅ 38.2% |
| TCS.NS | 🇮🇳 India | ✅ | ✅ 138 trades | ✅ 70.6% |

---

## 🧠 How It Works

### 1. Data Collection
- **Price data**: Yahoo Finance with auto-retry and CSV caching
- **News data**: NewsAPI.org (primary) → NewsData.io (fallback)
- **Relevance filtering**: Dynamic company name lookup via yfinance + financial keyword matching

### 2. Sentiment Analysis
- VADER NLP scores each headline (-1.0 to +1.0)
- Weekend/holiday articles are rolled back to nearest trading day
- Word clouds and keyword frequency analysis

### 3. Price Forecasting
- **Prophet**: Time series decomposition with sentiment as extra regressor
- **LSTM** (optional): Deep learning with EarlyStopping
- **Naive baseline**: Yesterday's close for fair model comparison

### 4. Strategy Backtesting
- Signal generation from predicted returns + sentiment agreement
- Long-only simulation with commissions (0.1%)
- Risk metrics: Sharpe Ratio, VaR (95%), Max Drawdown

### 5. AI Advisor (Explainable)
- RandomForest classifier predicting 5-day price direction
- SHAP feature contributions (with Gini importance fallback)
- Health score, confidence level, and plain-English explanations

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → Select `stock_sentinel/app.py`
4. Add secrets: `NEWS_API_KEY`, `NEWSDATA_API_KEY`
5. Deploy!

### Render

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service
3. **Start Command**: `streamlit run stock_sentinel/app.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables and deploy

---

## ⚙️ Configuration

Edit `stock_sentinel/utils.py`:

```python
STOCKS          = ["AAPL", "MSFT", "TSLA", "PSO.KA", "OGDC.KA"]
NEWS_SOURCE     = "auto"       # 'newsapi' | 'newsdata' | 'auto'
INITIAL_CAPITAL = 10_000       # Starting capital for backtests
COMMISSION      = 0.001        # 0.1% per trade
```

Date ranges are automatically set to the last 6 months.

---

## ⚠️ Known Limitations

- **NewsAPI free tier**: Only articles from the last 30 days — historical sentiment will be sparse
- **LSTM**: Slow to train on free hardware; Prophet is recommended for quick runs
- **Pakistani stocks (.KA)**: May have sparse coverage on some tickers
- **PuLP optimizer**: Simplified minimum-variance model; not for real portfolio management

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly, Custom CSS (Glassmorphism) |
| **ML/AI** | scikit-learn, Prophet, TensorFlow/Keras, SHAP |
| **NLP** | VADER Sentiment, NLTK, WordCloud |
| **Data** | yfinance, NewsAPI, NewsData.io, pandas, NumPy |
| **Optimization** | PuLP (Linear Programming) |
| **Deployment** | Streamlit Cloud, Render |

---

## 📄 License

MIT — free for educational and non-commercial use.

---

## 🙏 Acknowledgements

Built with: [yfinance](https://github.com/ranaroussi/yfinance) · [Prophet](https://facebook.github.io/prophet/) · [VADER](https://github.com/cjhutto/vaderSentiment) · [SHAP](https://github.com/shap/shap) · [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com) · [scikit-learn](https://scikit-learn.org) · [PuLP](https://coin-or.github.io/pulp/)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Awais-Analyst">Awais-Analyst</a>
</p>

