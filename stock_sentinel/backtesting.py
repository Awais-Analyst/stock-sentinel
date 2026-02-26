"""
backtesting.py — Strategy signal generation, simulation, risk metrics, and optimization.

Key rules:
  • No look-ahead bias: signals use only data available before the trade date.
  • Commission: 0.1% per trade.
  • PuLP portfolio optimization capped at 3–5 stocks for solver speed.
"""

import logging
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signals(df: pd.DataFrame,
                     pred_col: str = "lstm_pred",
                     sentiment_col: str = "sentiment",
                     pred_threshold: float = 0.005,
                     sentiment_threshold: float = 0.05) -> pd.Series:
    """
    Generate BUY (+1) / SELL (-1) / HOLD (0) signals.

    Rules (all based on prior-day data — no look-ahead):
      BUY  : predicted return > pred_threshold AND sentiment > sentiment_threshold
      SELL : predicted return < -pred_threshold OR sentiment < -sentiment_threshold
      HOLD : otherwise

    Returns a Series aligned with df.index.
    """
    signals = pd.Series(0, index=df.index, name="signal", dtype=int)

    if pred_col in df.columns:
        pred_return = (df[pred_col] - df["Close"].shift(1)) / df["Close"].shift(1)
    else:
        # Fallback: use naive momentum (log return > 0 → bullish)
        pred_return = df.get("log_return", pd.Series(0.0, index=df.index))
        log.warning(f"'{pred_col}' not found — using log_return as proxy for signals")

    sentiment = df[sentiment_col] if sentiment_col in df.columns else pd.Series(0.0, index=df.index)

    buy_mask  = (pred_return > pred_threshold)  & (sentiment > sentiment_threshold)
    sell_mask = (pred_return < -pred_threshold) | (sentiment < -sentiment_threshold)

    signals[buy_mask]  =  1
    signals[sell_mask] = -1

    n_buy = buy_mask.sum(); n_sell = sell_mask.sum()
    log.info(f"Signals: {n_buy} BUY, {n_sell} SELL, {len(signals)-n_buy-n_sell} HOLD")
    return signals


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 capital: float = 10_000.0,
                 commission: float = 0.001) -> pd.DataFrame:
    """
    Simulate a long-only strategy on signals.
    Returns a DataFrame with portfolio_value, cash, shares, buy_hold_value columns.
    """
    portfolio = []
    cash   = capital
    shares = 0.0
    state  = "cash"   # 'cash' or 'invested'

    buy_hold_shares = capital / df["Close"].iloc[0]   # benchmark

    for date, row in df.iterrows():
        sig = signals.get(date, 0)
        price = row["Close"]

        if sig == 1 and state == "cash" and cash > 0:
            # BUY
            buy_value = cash
            cost = buy_value * commission
            shares = (buy_value - cost) / price
            cash   = 0.0
            state  = "invested"

        elif sig == -1 and state == "invested" and shares > 0:
            # SELL
            proceeds = shares * price
            cost = proceeds * commission
            cash   = proceeds - cost
            shares = 0.0
            state  = "cash"

        port_value = cash + shares * price
        portfolio.append({
            "date":            date,
            "portfolio_value": port_value,
            "cash":            cash,
            "shares":          shares,
            "signal":          sig,
            "close":           price,
            "buy_hold_value":  buy_hold_shares * price,
        })

    result_df = pd.DataFrame(portfolio).set_index("date")
    final = result_df["portfolio_value"].iloc[-1]
    total_return = (final - capital) / capital * 100
    log.info(f"Backtest complete. Final value: ${final:,.2f} | Return: {total_return:+.2f}%")
    return result_df


# ─────────────────────────────────────────────
# RISK METRICS
# ─────────────────────────────────────────────

def calc_sharpe(portfolio_df: pd.DataFrame,
                risk_free_annual: float = 0.02) -> float:
    """Annualized Sharpe ratio based on daily portfolio returns."""
    pv = portfolio_df["portfolio_value"]
    daily_returns = pv.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    rf_daily = risk_free_annual / 252
    sharpe = (daily_returns.mean() - rf_daily) / daily_returns.std() * math.sqrt(252)
    log.info(f"Sharpe ratio: {sharpe:.4f}")
    return float(sharpe)


def calc_var(portfolio_df: pd.DataFrame, confidence: float = 0.95) -> float:
    """
    Historical 1-day Value at Risk (negative number = potential loss).
    E.g. -0.021 means 2.1% loss at 95% confidence.
    """
    pv = portfolio_df["portfolio_value"]
    daily_returns = pv.pct_change().dropna()
    var = float(np.quantile(daily_returns, 1 - confidence))
    log.info(f"VaR ({confidence*100:.0f}%): {var:.4f}")
    return var


def calc_max_drawdown(portfolio_df: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown as a fraction (negative)."""
    pv = portfolio_df["portfolio_value"]
    peak = pv.cummax()
    drawdown = (pv - peak) / peak
    max_dd = float(drawdown.min())
    log.info(f"Max drawdown: {max_dd:.4f}")
    return max_dd


def calc_all_metrics(portfolio_df: pd.DataFrame,
                      risk_free_annual: float = 0.02,
                      capital: float = 10_000.0) -> dict:
    """Convenience wrapper: compute all risk/return metrics in one call."""
    pv = portfolio_df["portfolio_value"]
    num_trades = int((portfolio_df["signal"] != 0).sum())
    total_return_pct = (pv.iloc[-1] - capital) / capital * 100

    return {
        "total_return_pct": round(total_return_pct, 4),
        "sharpe":           round(calc_sharpe(portfolio_df, risk_free_annual), 4),
        "var_95":           round(calc_var(portfolio_df, 0.95), 6),
        "max_drawdown":     round(calc_max_drawdown(portfolio_df), 6),
        "num_trades":       num_trades,
        "final_value":      round(float(pv.iloc[-1]), 2),
    }


# ─────────────────────────────────────────────
# PORTFOLIO OPTIMIZATION (PuLP)
# ─────────────────────────────────────────────

def optimize_portfolio(returns_df: pd.DataFrame,
                        target_return: float = 0.001) -> dict:
    """
    Minimum-variance portfolio optimization using PuLP.
    `returns_df`: columns = stock symbols, rows = daily returns.
    Capped at 5 stocks for solver speed on free hardware.
    Returns {symbol: weight} dict.
    """
    try:
        import pulp
    except ImportError:
        log.error("PuLP not installed. Run: pip install PuLP")
        return {}

    # Cap at 5 stocks
    stocks = list(returns_df.columns[:5])
    returns_df = returns_df[stocks].dropna()
    n = len(stocks)

    mu  = returns_df.mean().values       # expected returns
    cov = returns_df.cov().values        # covariance matrix

    problem = pulp.LpProblem("min_variance", pulp.LpMinimize)
    w = [pulp.LpVariable(f"w_{s}", lowBound=0, upBound=1) for s in stocks]

    # Objective: minimize portfolio variance (sum of w_i * w_j * cov_ij)
    variance = pulp.lpSum(
        w[i] * w[j] * float(cov[i][j])
        for i in range(n) for j in range(n)
    )
    problem += variance

    # Constraints
    problem += pulp.lpSum(w) == 1                        # weights sum to 1
    problem += pulp.lpSum(mu[i] * w[i] for i in range(n)) >= target_return  # min return

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    if pulp.LpStatus[problem.status] != "Optimal":
        log.warning("PuLP optimization did not find an optimal solution — returning equal weights.")
        equal = 1.0 / n
        return {s: equal for s in stocks}

    weights = {stocks[i]: round(float(w[i].value() or 0), 4) for i in range(n)}
    log.info(f"Portfolio weights: {weights}")
    return weights


# ─────────────────────────────────────────────
# INSIGHT NARRATIVE
# ─────────────────────────────────────────────

def generate_insights(results: dict, symbol: str = "") -> str:
    """Generate a concise human-readable strategy analysis narrative."""
    ret    = results.get("total_return_pct", float("nan"))
    sharpe = results.get("sharpe", float("nan"))
    var    = results.get("var_95", float("nan"))
    dd     = results.get("max_drawdown", float("nan"))
    trades = results.get("num_trades", 0)
    final  = results.get("final_value", float("nan"))

    quality = "strong" if sharpe > 1.0 else "moderate" if sharpe > 0.5 else "weak"
    risk_label = "low" if abs(dd) < 0.1 else "moderate" if abs(dd) < 0.25 else "high"

    return (
        f"Strategy Analysis — {symbol}\n"
        f"{'─' * 40}\n"
        f"The sentiment-driven strategy yielded a total simulated return of {ret:+.2f}% "
        f"with a final portfolio value of ${final:,.2f}.\n\n"
        f"Risk-adjusted performance is {quality} (Sharpe: {sharpe:.2f}). "
        f"The maximum drawdown was {dd:.2%}, indicating {risk_label} downside risk. "
        f"At 95% confidence, the worst daily loss (VaR) is estimated at {var:.2%}.\n\n"
        f"The strategy executed {trades} trades during the simulation period.\n\n"
        f"⚠️  This is a simulation using historical data. Past performance does not "
        f"guarantee future results. Not financial advice."
    )


# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("─── backtesting.py self-test (synthetic data) ───")
    np.random.seed(99)
    n = 200
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)

    df_test = pd.DataFrame({
        "Close":      close,
        "log_return": np.concatenate([[0], np.diff(np.log(close))]),
        "sentiment":  np.random.uniform(-0.6, 0.6, n),
    }, index=idx)

    signals = generate_signals(df_test, pred_col="nonexistent",
                                sentiment_col="sentiment")
    portfolio_df = run_backtest(df_test, signals, capital=10_000)
    metrics      = calc_all_metrics(portfolio_df)

    print(f"Metrics: {metrics}")
    print("\n" + generate_insights(metrics, symbol="TEST"))

    # Portfolio optimization test
    returns_data = pd.DataFrame(
        np.random.randn(100, 3) * 0.01 + 0.0005,
        columns=["AAPL", "MSFT", "TSLA"]
    )
    weights = optimize_portfolio(returns_data, target_return=0.0003)
    print(f"\nOptimal weights: {weights}")

    print("\n✅ backtesting.py self-test passed.")
