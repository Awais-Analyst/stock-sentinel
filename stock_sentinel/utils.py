"""
utils.py — Global config, helper functions, and visualization utilities.

Charts use transparent backgrounds so they look great on both
dark (Streamlit default) and light (white) device themes.
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
            increasing_line_color="#00c48c",
            decreasing_line_color="#e84545",
        ),
        row=1, col=1,
    )
    if "Volume" in df.columns:
        colors = [
            "#00c48c" if c >= o else "#e84545"
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color=colors, opacity=0.7),
            row=2, col=1,
        )
    _apply_theme(fig, f"{symbol} — Candlestick Chart")
    return fig


def plot_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame,
                  symbol: str = "", baseline: pd.Series = None) -> go.Figure:
    """Actual close + Prophet forecast + naive baseline + confidence band.
    
    Confidence interval shows actual upper/lower values on hover.
    """
    fig = go.Figure()

    # Actual close price
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Actual Close",
        line=dict(color="#00c48c", width=2),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))

    # Naive baseline
    if baseline is not None:
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values,
            name="Naive Baseline",
            line=dict(color="#aaaaaa", width=1.2, dash="dot"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Baseline: $%{y:.2f}<extra></extra>",
        ))

    if forecast_df is not None and "yhat" in forecast_df.columns:
        dates = pd.to_datetime(forecast_df["ds"])
        yhat       = forecast_df["yhat"]
        yhat_upper = forecast_df["yhat_upper"]
        yhat_lower = forecast_df["yhat_lower"]

        # Forecast line
        fig.add_trace(go.Scatter(
            x=dates, y=yhat,
            name="Forecast",
            line=dict(color="#f5a623", width=2.5, dash="dash"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:.2f}<extra></extra>",
        ))

        # Upper confidence bound (invisible line, part of fill reference)
        fig.add_trace(go.Scatter(
            x=dates, y=yhat_upper,
            name="Upper Bound",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hovertemplate="Upper: $%{y:.2f}<extra></extra>",
        ))

        # Lower confidence bound — fills to upper bound
        fig.add_trace(go.Scatter(
            x=dates, y=yhat_lower,
            name="Confidence Band (80%)",
            fill="tonexty",
            fillcolor="rgba(245,166,35,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            hovertemplate="Lower: $%{y:.2f}<extra></extra>",
        ))

    _apply_theme(fig, f"{symbol} — Price Forecast")
    fig.update_layout(hovermode="x")   # show each trace tooltip separately
    return fig


def plot_equity_curve(portfolio_df: pd.DataFrame, symbol: str = "") -> go.Figure:
    """Portfolio value vs buy-and-hold over time from backtest."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df["portfolio_value"],
        name="Strategy",
        fill="tozeroy",
        fillcolor="rgba(0,196,140,0.1)",
        line=dict(color="#00c48c", width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Portfolio: $%{y:,.2f}<extra></extra>",
    ))
    if "buy_hold_value" in portfolio_df.columns:
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["buy_hold_value"],
            name="Buy & Hold",
            line=dict(color="#7f8c8d", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Buy & Hold: $%{y:,.2f}<extra></extra>",
        ))
    _apply_theme(fig, f"{symbol} — Backtest Equity Curve")
    return fig


def plot_sentiment_bars(sentiment_series: pd.Series, symbol: str = "") -> go.Figure:
    """Daily sentiment score bar chart coloured by sign."""
    colors = ["#00c48c" if v >= 0 else "#e84545" for v in sentiment_series]
    fig = go.Figure(go.Bar(
        x=sentiment_series.index,
        y=sentiment_series.values,
        marker_color=colors,
        name="Daily Sentiment",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sentiment: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", opacity=0.6)
    _apply_theme(fig, f"{symbol} — Daily Sentiment (VADER)")
    return fig


def _apply_theme(fig: go.Figure, title: str = "") -> None:
    """
    Adaptive chart theme — transparent background so the chart looks
    correct on BOTH dark (Streamlit default) and white/light device themes.
    Gridlines and text use neutral colours visible in both modes.
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#555555"), x=0),
        # Transparent backgrounds — inherit the page background
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Neutral font readable on both dark and light
        font=dict(color="#444444", family="Inter, Arial, sans-serif", size=13),
        legend=dict(
            bgcolor="rgba(200,200,200,0.15)",
            bordercolor="rgba(150,150,150,0.3)",
            borderwidth=1,
            font=dict(color="#333333"),
        ),
        xaxis=dict(
            gridcolor="rgba(150,150,150,0.25)",
            zerolinecolor="rgba(150,150,150,0.4)",
            tickfont=dict(color="#555555"),
            title_font=dict(color="#555555"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(150,150,150,0.25)",
            zerolinecolor="rgba(150,150,150,0.4)",
            tickfont=dict(color="#555555"),
            title_font=dict(color="#555555"),
            showgrid=True,
        ),
        margin=dict(l=60, r=20, t=55, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#cccccc",
            font=dict(color="#222222", size=13),
        ),
    )


def generate_report_text(symbol: str, results: dict) -> str:
    """Generate a plain-text narrative summary of backtest results."""
    sharpe   = results.get("sharpe", float("nan"))
    var_95   = results.get("var_95", float("nan"))
    drawdown = results.get("max_drawdown", float("nan"))
    ret      = results.get("total_return_pct", float("nan"))
    trades   = results.get("num_trades", 0)

    lines = [
        f"Stock Sentinel — Strategy Report for {symbol}",
        "=" * 50,
        f"  Total Return    : {ret:+.2f}%",
        f"  Sharpe Ratio    : {sharpe:.3f}",
        f"  Value-at-Risk   : {var_95:.2%} (95% confidence, 1-day)",
        f"  Max Drawdown    : {drawdown:.2%}",
        f"  Number of Trades: {trades}",
        "",
        "This simulation is for educational purposes only.",
        "It is NOT financial advice. Past simulated performance",
        "does not guarantee future results.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Config loaded.")
    print(f"   Stocks     : {STOCKS}")
    print(f"   Data dir   : {DATA_DIR}")
    print(f"   News source: {NEWS_SOURCE}")

    dummy_results = {
        "sharpe": 1.32, "var_95": -0.021,
        "max_drawdown": -0.08, "total_return_pct": 14.7, "num_trades": 42,
    }
    print("\n" + generate_report_text("AAPL", dummy_results))
    print("\nutils.py self-test passed.")
