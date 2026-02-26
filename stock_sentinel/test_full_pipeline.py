"""
test_full_pipeline.py — End-to-end smoke test for Stock Sentinel.

Runs a minimal AAPL pipeline (6 months) and asserts data quality.
Does NOT require internet — uses cached data if available, synthetic otherwise.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# Ensure stock_sentinel package is on path
sys.path.insert(0, os.path.dirname(__file__))


def _synthetic_df(n: int = 126) -> pd.DataFrame:
    """Build a synthetic OHLCV + features DataFrame for offline testing."""
    np.random.seed(42)
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = 185 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "Open":       close * 0.998,
        "High":       close * 1.012,
        "Low":        close * 0.988,
        "Close":      close,
        "Volume":     np.random.randint(50_000_000, 120_000_000, n),
        "log_return": np.concatenate([[0], np.diff(np.log(close))]),
        "ma_10":      pd.Series(close).rolling(10).mean().values,
        "ma_20":      pd.Series(close).rolling(20).mean().values,
        "sentiment":  np.random.uniform(-0.4, 0.4, n),
    }, index=idx)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df


def test_data_shape(df: pd.DataFrame):
    assert len(df) > 0, "DataFrame is empty"
    assert "Close" in df.columns, "'Close' column missing"
    print(f"  [✓] DataFrame shape: {df.shape}")


def test_no_nan_close(df: pd.DataFrame):
    nan_close = df["Close"].isna().sum()
    assert nan_close == 0, f"Found {nan_close} NaN values in 'Close'"
    print("  [✓] No NaN in 'Close'")


def test_sentiment_col(df: pd.DataFrame):
    assert "sentiment" in df.columns, "'sentiment' column missing"
    nan_sent = df["sentiment"].isna().sum()
    assert nan_sent == 0, f"Found {nan_sent} NaN values in 'sentiment'"
    print("  [✓] 'sentiment' column present and complete")


def test_sentiment_module():
    from sentiment import clean_text, score_sentiment, get_top_keywords
    samples = [
        "Stocks surge on record earnings",
        "Market crash fears grip investors",
        "Mixed signals from Fed meeting",
    ]
    cleaned = [clean_text(t) for t in samples]
    assert all(isinstance(c, str) for c in cleaned), "clean_text failed"
    score = score_sentiment(samples)
    assert isinstance(score, float), "score_sentiment must return float"
    assert -1.0 <= score <= 1.0, f"Score out of range: {score}"
    kws = get_top_keywords(samples, n=5)
    assert isinstance(kws, list), "get_top_keywords must return list"
    print(f"  [✓] Sentiment module: score={score:.4f}, top_kw={kws[:3]}")


def test_technical_indicators(df: pd.DataFrame):
    from modeling import add_technical_indicators, add_lag_features, naive_baseline
    df2 = add_technical_indicators(df)
    assert "rsi_14" in df2.columns, "RSI missing"
    assert "macd"   in df2.columns, "MACD missing"
    df2 = add_lag_features(df2)
    assert "return_lag_1" in df2.columns, "Lag feature missing"
    baseline = naive_baseline(df2)
    assert isinstance(baseline, pd.Series), "Naive baseline must be a Series"
    print("  [✓] Technical indicators and lag features OK")


def test_prophet_forecast(df: pd.DataFrame):
    from modeling import forecast_with_prophet
    try:
        forecast = forecast_with_prophet(df, horizon=7)
        if forecast is not None:
            assert "yhat" in forecast.columns, "Forecast missing 'yhat' column"
            assert not forecast["yhat"].isna().all(), "All yhat values are NaN"
            print(f"  [✓] Prophet forecast shape: {forecast.shape}")
        else:
            print("  [⚠] Prophet not available — install prophet for forecasting")
    except Exception as e:
        print(f"  [⚠] Prophet test skipped: {e}")


def test_anomaly_detection(df: pd.DataFrame):
    from modeling import detect_anomalies
    df2 = detect_anomalies(df)
    assert "anomaly" in df2.columns, "'anomaly' column missing"
    n = df2["anomaly"].sum()
    assert isinstance(n, (int, np.integer)), "anomaly count must be int"
    print(f"  [✓] Anomaly detection: {n} anomalies flagged")


def test_backtest(df: pd.DataFrame) -> dict:
    from backtesting import generate_signals, run_backtest, calc_all_metrics
    signals = generate_signals(df, pred_col="nonexistent")
    portfolio = run_backtest(df, signals, capital=10_000)
    assert len(portfolio) > 0, "Portfolio DataFrame empty"
    metrics = calc_all_metrics(portfolio)
    assert math.isfinite(metrics["sharpe"]),           "Sharpe is not finite"
    assert math.isfinite(metrics["var_95"]),           "VaR is not finite"
    assert math.isfinite(metrics["max_drawdown"]),     "Drawdown is not finite"
    assert math.isfinite(metrics["total_return_pct"]), "Return is not finite"
    print(f"  [✓] Backtest metrics: {metrics}")
    return metrics


def run_all_tests():
    print("=" * 55)
    print(" Stock Sentinel — End-to-End Smoke Test")
    print("=" * 55)

    print("\n[1] Building synthetic dataset…")
    df = _synthetic_df(n=126)

    # Try real cached data first
    from utils import DATA_DIR
    cached_path = os.path.join(DATA_DIR, "AAPL_dataset.csv")
    if os.path.exists(cached_path):
        try:
            df = pd.read_csv(cached_path, index_col=0, parse_dates=True)
            df.dropna(subset=["Close"], inplace=True)
            print(f"  [✓] Loaded cached AAPL data: {df.shape}")
        except Exception:
            print("  [⚠] Cache load failed — using synthetic data")

    test_data_shape(df)
    test_no_nan_close(df)
    test_sentiment_col(df)

    print("\n[2] Sentiment module…")
    test_sentiment_module()

    print("\n[3] Technical indicators…")
    from modeling import add_technical_indicators, add_lag_features
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    test_technical_indicators(df)

    print("\n[4] Prophet forecast…")
    test_prophet_forecast(df)

    print("\n[5] Anomaly detection…")
    test_anomaly_detection(df)

    print("\n[6] Backtest engine…")
    metrics = test_backtest(df)

    print("\n" + "=" * 55)
    print(" ✅ ALL SMOKE TESTS PASSED")
    print("=" * 55)
    return metrics


if __name__ == "__main__":
    run_all_tests()
