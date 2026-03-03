"""
xai_advisor.py — Explainable AI + Investment Advisor for Stock Sentinel.

Provides:
  • train_advisor_model(): RandomForest classifier on technical + sentiment features
  • get_shap_explanation():  SHAP values per feature for the latest prediction
  • get_investment_advice(): BUY / HOLD / SELL with probability & plain-English reasoning
  • get_stock_health_score(): 0–100 composite score
  • build_advice_narrative():  Full plain-English paragraph explaining the decision
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FEATURE COLUMNS USED BY THE ADVISOR MODEL
# ─────────────────────────────────────────────
ADVISOR_FEATURES = [
    "rsi_14",
    "macd",
    "macd_hist",
    "boll_width",
    "atr_14",
    "sentiment",
    "log_return",
    "return_lag_1",
    "return_lag_3",
    "return_lag_5",
    "volume_change",   # added if Volume available
]

# Human-readable names for each feature (used in charts and explanations)
FEATURE_LABELS = {
    "rsi_14":       "Momentum (RSI)",
    "macd":         "Trend Strength (MACD)",
    "macd_hist":    "Trend Acceleration",
    "boll_width":   "Price Volatility",
    "atr_14":       "Daily Swing Size",
    "sentiment":    "News & Social Mood",
    "log_return":   "Yesterday's Return",
    "return_lag_1": "1-Day Lagged Return",
    "return_lag_3": "3-Day Lagged Return",
    "return_lag_5": "5-Day Lagged Return",
    "volume_change":"Trading Volume Change",
}


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix; add volume_change if Volume is present."""
    df = df.copy()
    if "Volume" in df.columns:
        df["volume_change"] = df["Volume"].pct_change()
    available = [c for c in ADVISOR_FEATURES if c in df.columns]
    return df[available].dropna()


def _make_labels(df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    """
    Binary label: 1 = price higher in `forward_days` days, 0 = lower or flat.
    Uses forward return — NO look-ahead in production (labels only for training).
    """
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1
    labels = (future_return > 0).astype(int)
    return labels


def train_advisor_model(df: pd.DataFrame):
    """
    Train a RandomForest classifier to predict whether price will rise
    over the next 5 days.

    Returns (model, feature_cols, accuracy_pct) or (None, [], 0) on failure.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        feat_df = _prepare_features(df)
        labels  = _make_labels(df).reindex(feat_df.index).dropna()

        # Align
        common  = feat_df.index.intersection(labels.index)
        X = feat_df.loc[common]
        y = labels.loc[common]

        if len(X) < 50:
            log.warning("Not enough data to train advisor model (need 50+ rows).")
            return None, [], 0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds    = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds) * 100
        log.info(f"Advisor model trained. Accuracy: {accuracy:.1f}%")
        return model, list(X.columns), round(accuracy, 1)

    except Exception as e:
        log.error(f"Advisor model training failed: {e}")
        return None, [], 0


def get_shap_explanation(model, df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Compute SHAP values for the most recent data point.

    Returns {feature_name: shap_value} sorted by absolute importance.
    Positive = pushed toward BUY, Negative = pushed toward SELL/HOLD.
    """
    try:
        import shap

        feat_df  = _prepare_features(df)[feature_cols].dropna()
        if feat_df.empty:
            return {}

        explainer = shap.TreeExplainer(model)
        latest     = feat_df.tail(1)

        shap_values = explainer.shap_values(latest)

        # Handle all possible SHAP output shapes:
        # 1) List of arrays (older shap): [class0_array, class1_array]
        # 2) 3D numpy array (newer shap): shape (n_classes, n_samples, n_features)
        # 3) 2D numpy array: shape (n_samples, n_features)
        if isinstance(shap_values, list):
            # List format: pick class 1 (BUY direction), first sample
            sv = np.array(shap_values[1]).flatten()[:len(feature_cols)]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # 3D array: [n_classes, n_samples, n_features] → class 1, sample 0
                sv = shap_values[1, 0, :]
            elif shap_values.ndim == 2:
                # 2D array: [n_samples, n_features] → sample 0
                sv = shap_values[0, :]
            elif shap_values.ndim == 1:
                sv = shap_values
            else:
                sv = shap_values.flatten()[:len(feature_cols)]
        else:
            sv = np.array(shap_values).flatten()[:len(feature_cols)]

        result = {
            FEATURE_LABELS.get(col, col): float(v)
            for col, v in zip(feature_cols, sv)
        }
        # Sort by absolute importance
        result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
        return result

    except ImportError:
        log.warning("SHAP not installed — pip install shap")
        return {}
    except Exception as e:
        log.error(f"SHAP explanation failed: {e}")
        return {}


def get_investment_advice(model, df: pd.DataFrame, feature_cols: list,
                           forecast_df=None) -> dict:
    """
    Generate investment advice for the latest available data point.

    Returns a dict with:
      action       : 'BUY' | 'HOLD' | 'SELL'
      confidence   : 0–100 (%)
      probability_up : 0–100 (% chance price rises next 5 days)
      reasons      : list of plain-English reason strings
      risk_level   : 'Low' | 'Moderate' | 'High'
      health_score : 0–100 composite stock health
    """
    result = {
        "action": "HOLD", "confidence": 50,
        "probability_up": 50, "reasons": [],
        "risk_level": "Moderate", "health_score": 50,
    }

    try:
        feat_df = _prepare_features(df)[feature_cols].dropna()
        if feat_df.empty:
            return result

        latest   = feat_df.tail(1)
        prob     = model.predict_proba(latest)[0]   # [prob_down, prob_up]
        prob_up  = round(float(prob[1]) * 100, 1)
        prob_dn  = round(float(prob[0]) * 100, 1)

        result["probability_up"] = prob_up

        # Action thresholds
        if prob_up >= 65:
            action, confidence = "BUY",  prob_up
        elif prob_up <= 35:
            action, confidence = "SELL", prob_dn
        else:
            action, confidence = "HOLD", max(prob_up, prob_dn)

        result["action"]     = action
        result["confidence"] = round(confidence, 1)

        # ── Plain-English reasons ────────────────────────────────────────
        reasons = []
        latest_row = df.iloc[-1]

        # RSI
        rsi = latest_row.get("rsi_14", 50)
        if rsi < 30:
            reasons.append(f"RSI is {rsi:.0f} — stock looks oversold, may bounce back up 📈")
        elif rsi > 70:
            reasons.append(f"RSI is {rsi:.0f} — stock looks overbought, may pull back 📉")
        else:
            reasons.append(f"RSI is {rsi:.0f} — neutral momentum, no strong signal")

        # MACD
        macd = latest_row.get("macd", 0)
        macd_h = latest_row.get("macd_hist", 0)
        if macd > 0 and macd_h > 0:
            reasons.append("MACD is positive and rising — upward trend gaining strength 🟢")
        elif macd < 0 and macd_h < 0:
            reasons.append("MACD is negative and falling — downward trend strengthening 🔴")
        else:
            reasons.append("MACD signal is mixed — trend direction unclear ⚪")

        # Sentiment
        sent = latest_row.get("sentiment", 0)
        if sent > 0.1:
            reasons.append(f"News mood is positive ({sent:.2f}) — market optimism is high 🟢")
        elif sent < -0.1:
            reasons.append(f"News mood is negative ({sent:.2f}) — market pessimism is high 🔴")
        else:
            reasons.append("News mood is neutral — no strong news signal ⚪")

        # Bollinger
        bw = latest_row.get("boll_width", 0.05)
        if bw > 0.1:
            reasons.append("Price volatility is HIGH — big moves expected, higher risk ⚠️")
        else:
            reasons.append("Price volatility is LOW — relatively calm market conditions ✅")

        # Prophet forecast direction (if available)
        if forecast_df is not None and "yhat" in forecast_df.columns:
            last_actual = float(df["Close"].iloc[-1])
            last_forecast = float(forecast_df["yhat"].iloc[-1])
            pct_change = (last_forecast - last_actual) / last_actual * 100
            if pct_change > 1:
                reasons.append(
                    f"AI price model forecasts +{pct_change:.1f}% change "
                    f"over the forecast period 📈"
                )
            elif pct_change < -1:
                reasons.append(
                    f"AI price model forecasts {pct_change:.1f}% change "
                    f"over the forecast period 📉"
                )

        result["reasons"] = reasons

        # ── Risk level ───────────────────────────────────────────────────
        atr = latest_row.get("atr_14", 0)
        close = latest_row.get("Close", 1)
        atr_pct = atr / close if close > 0 else 0
        if atr_pct > 0.025:
            risk = "High"
        elif atr_pct > 0.012:
            risk = "Moderate"
        else:
            risk = "Low"
        result["risk_level"] = risk

        # ── Health score (0–100) ─────────────────────────────────────────
        result["health_score"] = get_stock_health_score(df)

    except Exception as e:
        log.error(f"Investment advice generation failed: {e}")

    return result


def get_stock_health_score(df: pd.DataFrame) -> int:
    """
    Composite 0–100 score based on: RSI, MACD, sentiment, recent return.
    50 = neutral, >60 = healthy, <40 = weak.
    """
    try:
        row   = df.iloc[-1]
        score = 50   # start neutral

        rsi = row.get("rsi_14", 50)
        if 40 < rsi < 60:   score += 10   # ideal RSI range
        elif rsi < 30:      score += 5    # oversold — possible bounce
        elif rsi > 70:      score -= 10   # overbought

        macd = row.get("macd", 0)
        score += min(10, max(-10, int(macd / abs(macd) * 10))) if macd != 0 else 0

        sent = row.get("sentiment", 0)
        score += int(sent * 20)  # sentiment -1..+1 → -20..+20

        ret5 = row.get("return_lag_5", 0)
        score += int(ret5 * 100)  # 5-day return contribution

        return max(0, min(100, score))
    except Exception:
        return 50


def build_advice_narrative(advice: dict, symbol: str) -> str:
    """
    Build a full plain-English paragraph from the advice dict.
    Written like a financial commentary — accessible to everyone.
    """
    action     = advice.get("action", "HOLD")
    confidence = advice.get("confidence", 50)
    prob_up    = advice.get("probability_up", 50)
    risk       = advice.get("risk_level", "Moderate")
    health     = advice.get("health_score", 50)
    reasons    = advice.get("reasons", [])

    action_desc = {
        "BUY":  "**consider buying**",
        "SELL": "**consider selling**",
        "HOLD": "**hold and wait**",
    }.get(action, "hold")

    confidence_word = (
        "very confident" if confidence >= 75 else
        "fairly confident" if confidence >= 60 else
        "moderately confident"
    )

    health_word = (
        "very healthy 🟢" if health >= 70 else
        "fairly healthy 🟡" if health >= 50 else
        "weak 🔴"
    )

    narrative = (
        f"Based on the current data for **{symbol}**, our AI analysis suggests you "
        f"{action_desc} at this time. The model is **{confidence_word}** "
        f"({confidence:.0f}% confidence).\n\n"
        f"**Probability of price increase (next 5 days): {prob_up:.0f}%**\n\n"
        f"**Overall Stock Health Score: {health}/100 — {health_word}**\n\n"
        f"**Key reasons for this recommendation:**\n"
    )
    for i, r in enumerate(reasons, 1):
        narrative += f"{i}. {r}\n"

    narrative += (
        f"\n**Risk Level: {risk}**\n\n"
        f"> ⚠️ *This is an AI-generated suggestion for educational purposes only. "
        f"It is NOT financial advice. Always do your own research and consult a "
        f"qualified financial advisor before making any investment decisions.*"
    )
    return narrative
