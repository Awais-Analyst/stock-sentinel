"""
modeling.py — Feature engineering, Prophet forecasting, LSTM, and anomaly detection.

Key features (2026 robustness):
  • Naive baseline (yesterday's close) for fair model comparison
  • Prophet: sentiment as extra regressor
  • LSTM: EarlyStopping + validation_split to prevent overfitting
  • IsolationForest for anomaly flagging
  • Model persistence (joblib / Keras .h5)
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add manual RSI, MACD, Bollinger Bands, and ATR to the DataFrame."""
    df = df.copy()

    # RSI (14-period)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-period)
    sma20       = df["Close"].rolling(20).mean()
    std20       = df["Close"].rolling(20).std()
    df["boll_upper"] = sma20 + 2 * std20
    df["boll_lower"] = sma20 - 2 * std20
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / sma20

    # ATR (14-period)
    high_low   = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift()).abs()
    low_close_prev  = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """Add lagged returns and lagged close prices as features."""
    if lags is None:
        lags = [1, 3, 5]
    df = df.copy()
    for lag in lags:
        df[f"return_lag_{lag}"] = df["Close"].pct_change().shift(lag)
        df[f"close_lag_{lag}"]  = df["Close"].shift(lag)
    return df


def naive_baseline(df: pd.DataFrame) -> pd.Series:
    """Yesterday's close as next-day prediction (naive benchmark)."""
    baseline = df["Close"].shift(1)
    baseline.name = "naive_pred"
    return baseline


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ─────────────────────────────────────────────
# PROPHET FORECASTING
# ─────────────────────────────────────────────

def forecast_with_prophet(df: pd.DataFrame, horizon: int = 30,
                           sentiment_col: str = "sentiment") -> pd.DataFrame | None:
    """
    Fit Prophet on Close price with optional sentiment regressor.
    Returns full forecast DataFrame (includes history + `horizon` future days).
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.error("prophet not installed. Run: pip install prophet")
        return None

    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    use_sentiment = sentiment_col in df.columns and df[sentiment_col].notna().any()
    if use_sentiment:
        prophet_df["sentiment"] = df[sentiment_col].values

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    if use_sentiment:
        model.add_regressor("sentiment")

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="B")
    if use_sentiment:
        last_sentiment = float(prophet_df["sentiment"].iloc[-1])
        future["sentiment"] = prophet_df.set_index("ds")["sentiment"].reindex(
            future["ds"]).fillna(last_sentiment).values

    forecast = model.predict(future)
    log.info(f"Prophet forecast: {len(forecast)} rows ({horizon} future days)")

    # Evaluation on in-sample test (last 20%)
    n_test = max(int(len(prophet_df) * 0.2), 5)
    y_true = prophet_df["y"].values[-n_test:]
    y_pred = forecast["yhat"].values[-(n_test + horizon):-horizon]
    if len(y_true) == len(y_pred):
        metrics = _eval_metrics(y_true, y_pred)
        log.info(f"Prophet metrics → RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    return forecast


# ─────────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────────

def _build_sequences(data: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])   # 0th feature = close price
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple) -> object:
    """
    Build compiled Keras LSTM with dropout layers.
    input_shape: (timesteps, features)
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        log.error("TensorFlow not installed. Run: pip install tensorflow")
        return None

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    log.info(f"LSTM model built. Input shape: {input_shape}")
    return model


def train_lstm(df: pd.DataFrame, sentiment_col: str = "sentiment",
               window: int = 20, epochs: int = 50,
               batch_size: int = 32) -> tuple:
    """
    Train LSTM with EarlyStopping + validation_split.
    Returns (model, scaler, train_metrics, test_metrics).
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        log.error("TensorFlow not installed.")
        return None, None, {}, {}

    feature_cols = ["Close"]
    if sentiment_col in df.columns:
        feature_cols.append(sentiment_col)
    optional = ["rsi_14", "macd", "boll_width", "atr_14", "log_return"]
    feature_cols += [c for c in optional if c in df.columns]

    data = df[feature_cols].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = _build_sequences(scaled, window=window)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(input_shape=(window, X.shape[2]))
    if model is None:
        return None, None, {}, {}

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=1),
    ]
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    log.info("LSTM training complete.")

    # Evaluate
    def _inverse_close(arr_scaled):
        """Inverse-transform only the Close column."""
        dummy = np.zeros((len(arr_scaled), scaled.shape[1]))
        dummy[:, 0] = arr_scaled
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_train = _inverse_close(model.predict(X_train, verbose=0).flatten())
    y_pred_test  = _inverse_close(model.predict(X_test,  verbose=0).flatten())
    y_train_inv  = _inverse_close(y_train)
    y_test_inv   = _inverse_close(y_test)

    train_metrics = _eval_metrics(y_train_inv, y_pred_train)
    test_metrics  = _eval_metrics(y_test_inv,  y_pred_test)
    log.info(f"LSTM train → {train_metrics}")
    log.info(f"LSTM test  → {test_metrics}")

    return model, scaler, train_metrics, test_metrics


def predict_lstm(model, scaler, df: pd.DataFrame,
                 window: int = 20, sentiment_col: str = "sentiment") -> pd.Series:
    """Generate LSTM predictions over the full dataset, returns a Series aligned with df."""
    feature_cols = ["Close"]
    if sentiment_col in df.columns:
        feature_cols.append(sentiment_col)
    optional = ["rsi_14", "macd", "boll_width", "atr_14", "log_return"]
    feature_cols += [c for c in optional if c in df.columns]

    data   = df[feature_cols].dropna().values
    scaled = scaler.transform(data)
    X, _   = _build_sequences(scaled, window=window)

    raw_preds = model.predict(X, verbose=0).flatten()
    dummy = np.zeros((len(raw_preds), scaled.shape[1]))
    dummy[:, 0] = raw_preds
    preds = scaler.inverse_transform(dummy)[:, 0]

    idx = df.dropna(subset=feature_cols).index[window:]
    return pd.Series(preds, index=idx, name="lstm_pred")


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame,
                     feature: str = "log_return",
                     contamination: float = 0.03) -> pd.DataFrame:
    """
    Use IsolationForest to flag anomalous trading days.
    Adds 'anomaly' column (True = anomalous).
    """
    df = df.copy()
    if feature not in df.columns:
        log.warning(f"Feature '{feature}' not in DataFrame. Skipping anomaly detection.")
        df["anomaly"] = False
        return df

    feat_data = df[[feature]].dropna()
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(feat_data)
    df.loc[feat_data.index, "anomaly"] = (preds == -1)
    df["anomaly"] = df["anomaly"].fillna(False)
    n_anomalies = df["anomaly"].sum()
    log.info(f"IsolationForest flagged {n_anomalies} anomalous trading days "
             f"({n_anomalies / len(df) * 100:.1f}%)")
    return df


# ─────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model(model, path: str) -> None:
    if hasattr(model, "save"):            # Keras model
        model.save(path if path.endswith(".h5") else path + ".h5")
    else:
        joblib.dump(model, path)
    log.info(f"Model saved → {path}")


def load_model_from_path(path: str):
    if path.endswith(".h5"):
        try:
            from tensorflow import keras
            return keras.models.load_model(path)
        except ImportError:
            log.error("TensorFlow not available for loading .h5 model.")
            return None
    return joblib.load(path)


# ─────────────────────────────────────────────
# SELF-TEST (uses synthetic data — no API needed)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("─── modeling.py self-test (synthetic data) ───")
    np.random.seed(42)
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = 150 + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        "Open":   close * 0.998,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sentiment"]  = np.random.uniform(-0.5, 0.5, n)

    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df.dropna(inplace=True)
    print(f"After feature engineering: {df.shape}")
    print(df[["rsi_14", "macd", "boll_width"]].tail(3))

    baseline = naive_baseline(df)
    print(f"\nNaive baseline (last 3): {baseline.tail(3).values}")

    print("\nRunning Prophet...")
    forecast = forecast_with_prophet(df, horizon=10)
    if forecast is not None:
        print(f"Forecast shape: {forecast.shape}")
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))

    print("\nRunning anomaly detection...")
    df = detect_anomalies(df)
    print(f"Anomalies flagged: {df['anomaly'].sum()} / {len(df)}")

    print("\n✅ modeling.py self-test passed.")
