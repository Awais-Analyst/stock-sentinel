"""
utils.py — Global config, helper functions, and visualization utilities.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STOCKS = ["AAPL", "MSFT", "TSLA", "PSO.KA", "OGDC.KA"]

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "YOUR_NEWSDATA_KEY")

# 'newsapi' | 'newsdata' | 'auto' (tries newsapi first, falls back to newsdata)
NEWS_SOURCE = os.getenv("NEWS_SOURCE", "auto")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_START = "2023-01-01"
DEFAULT_END   = "2024-12-31"
COMMISSION    = 0.001   # 0.1% per trade
INITIAL_CAPITAL = 10_000


# ─────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, symbol: str = "") -> go.Figure:
    """Interactive Plotly candlestick with volume sub-plot."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} Price", "Volume"),
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4d6d",
        ),
        row=1, col=1,
    )
    if "Volume" in df.columns:
        colors = [
            "#00d4aa" if c >= o else "#ff4d6d"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors),
            row=2, col=1,
        )
    _apply_dark_theme(fig, f"{symbol} — Candlestick Chart")
    return fig


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                  symbol: str = "", baseline: pd.Series = None) -> go.Figure:
    """Actual close + Prophet forecast + optional naive baseline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Actual Close",
        line=dict(color="#00d4aa", width=2),
    ))
    if baseline is not None:
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values,
            name="Naive Baseline",
            line=dict(color="#888", width=1, dash="dot"),
        ))
    if forecast_df is not None and "yhat" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_df["ds"]),
            y=forecast_df["yhat"],
            name="Forecast",
            line=dict(color="#f7b731", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(forecast_df["ds"]).tolist() +
              pd.to_datetime(forecast_df["ds"]).tolist()[::-1],
            y=forecast_df["yhat_upper"].tolist() + forecast_df["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(247,183,49,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        ))
    _apply_dark_theme(fig, f"{symbol} — Price Forecast")
    return fig


def plot_equity_curve(portfolio_df: pd.DataFrame, symbol: str = "") -> go.Figure:
    """Portfolio value over time from backtest."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df["portfolio_value"],
        name="Portfolio Value",
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="#00d4aa", width=2),
    ))
    if "buy_hold_value" in portfolio_df.columns:
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["buy_hold_value"],
            name="Buy & Hold",
            line=dict(color="#888", width=1.5, dash="dot"),
        ))
    _apply_dark_theme(fig, f"{symbol} — Backtest Equity Curve")
    return fig


def plot_sentiment_bars(sentiment_series: pd.Series, symbol: str = "") -> go.Figure:
    """Daily sentiment score bar chart coloured by sign."""
    colors = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in sentiment_series]
    fig = go.Figure(go.Bar(
        x=sentiment_series.index,
        y=sentiment_series.values,
        marker_color=colors,
        name="Daily Sentiment",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    _apply_dark_theme(fig, f"{symbol} — Daily Sentiment (VADER)")
    return fig


def _apply_dark_theme(fig: go.Figure, title: str = "") -> None:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e0e0e0")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )


def generate_report_text(symbol: str, results: dict) -> str:
    """Generate a plain-text narrative summary of backtest results."""
    sharpe   = results.get("sharpe", float("nan"))
    var_95   = results.get("var_95", float("nan"))
    drawdown = results.get("max_drawdown", float("nan"))
    ret      = results.get("total_return_pct", float("nan"))
    trades   = results.get("num_trades", 0)

    lines = [
        f"📊 Stock Sentinel — Strategy Report for {symbol}",
        "=" * 50,
        f"  Total Return    : {ret:+.2f}%",
        f"  Sharpe Ratio    : {sharpe:.3f}",
        f"  Value-at-Risk   : {var_95:.2%} (95% confidence, 1-day)",
        f"  Max Drawdown    : {drawdown:.2%}",
        f"  Number of Trades: {trades}",
        "",
        "⚠️  This simulation is for educational purposes only.",
        "    It is NOT financial advice. Past simulated performance",
        "    does not guarantee future results.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("✅ Config loaded.")
    print(f"   Stocks     : {STOCKS}")
    print(f"   Data dir   : {DATA_DIR}")
    print(f"   News source: {NEWS_SOURCE}")

    dummy_results = {
        "sharpe": 1.32, "var_95": -0.021,
        "max_drawdown": -0.08, "total_return_pct": 14.7, "num_trades": 42,
    }
    print("\n" + generate_report_text("AAPL", dummy_results))
    print("\n✅ utils.py self-test passed.")
