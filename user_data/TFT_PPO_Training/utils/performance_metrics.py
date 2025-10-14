# user_data/TFT_PPO_Training/utils/performance_metrics.py
import numpy as np


def performance_metrics(
    returns: np.ndarray,
    freq: str = "daily",
    risk_free_rate: float = 0.0
) -> dict:
    """
    Compute advanced performance metrics for trading / RL evaluation.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (simple returns, e.g. [0.01, -0.002, ...]).
    freq : {"daily", "hourly", "minute"}
        Data frequency used for annualization.
    risk_free_rate : float, optional
        Annualized risk-free rate to adjust Sharpe and Sortino ratios.

    Returns
    -------
    dict
        Dictionary containing Sharpe, Sortino, Calmar, MDD, WinRate, Skew, Kurtosis, etc.
    """
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "mdd": np.nan,
            "win_rate": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "avg_return": np.nan,
            "volatility": np.nan,
        }

    # --- Annualization factor ---
    ann_factor = {"daily": 252, "hourly": 24 * 252, "minute": 60 * 24 * 252}.get(freq, 252)

    # --- Basic stats ---
    mean_ret = np.mean(returns)
    vol = np.std(returns, ddof=1) + 1e-12
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) + 1e-12

    # --- Sharpe & Sortino ---
    excess_ret = mean_ret - (risk_free_rate / ann_factor)
    sharpe = (excess_ret / vol) * np.sqrt(ann_factor)
    sortino = (excess_ret / downside_std) * np.sqrt(ann_factor)

    # --- Cumulative curve for drawdown ---
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1
    mdd = np.min(dd)
    calmar = (mean_ret * ann_factor) / (abs(mdd) + 1e-12)

    # --- Win rate ---
    win_rate = np.mean(returns > 0)

    # --- Distribution shape ---
    z = (returns - mean_ret) / vol
    skew = np.mean(z**3)
    kurtosis = np.mean(z**4) - 3

    return {
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "calmar": round(float(calmar), 4),
        "mdd": round(float(mdd), 4),
        "win_rate": round(float(win_rate), 4),
        "skew": round(float(skew), 4),
        "kurtosis": round(float(kurtosis), 4),
        "avg_return": round(float(mean_ret), 6),
        "volatility": round(float(vol), 6),
    }
