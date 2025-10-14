# user_data/TFT_PPO_modules/performance_metrics.py
import numpy as np

def performance_metrics(
    returns: np.ndarray,
    freq: str = "daily",
    risk_free_rate: float = 0.0
) -> dict:
    """
    Compute key trading performance metrics for RL evaluation or backtesting.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (simple returns, not log returns).
    freq : {"daily", "hourly", "minute"}
        Data frequency for annualization scaling.
    risk_free_rate : float, optional
        Annualized risk-free rate for Sharpe/Sortino adjustment.

    Returns
    -------
    dict
        {
            "sharpe": float,
            "sortino": float,
            "calmar": float,
            "mdd": float,
            "win_rate": float,
            "skew": float,
            "kurtosis": float,
            "avg_return": float,
            "volatility": float,
        }
    """

    # --- Safety & conversion ---
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return {k: np.nan for k in ["sharpe", "sortino", "calmar", "mdd", "win_rate", "skew", "kurtosis", "avg_return", "volatility"]}

    # --- Annualization scaling ---
    ann_factor = {"daily": 252, "hourly": 24 * 252, "minute": 24 * 60 * 252}.get(freq, 252)

    # --- Mean & volatility ---
    mean_ret = np.mean(returns)
    vol = np.std(returns, ddof=1) + 1e-12

    # --- Sharpe ratio ---
    excess_ret = mean_ret - (risk_free_rate / ann_factor)
    sharpe = (excess_ret / vol) * np.sqrt(ann_factor)

    # --- Sortino ratio (downside risk only) ---
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) + 1e-12
    sortino = (excess_ret / downside_std) * np.sqrt(ann_factor)

    # --- Cumulative equity curve ---
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1
    mdd = np.min(dd)
    calmar = (mean_ret * ann_factor) / (abs(mdd) + 1e-12)

    # --- Win rate ---
    win_rate = np.mean(returns > 0)

    # --- Distribution shape ---
    skew = np.mean(((returns - mean_ret) / vol) ** 3)
    kurtosis = np.mean(((returns - mean_ret) / vol) ** 4) - 3

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "mdd": float(mdd),
        "win_rate": float(win_rate),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "avg_return": float(mean_ret),
        "volatility": float(vol),
    }
