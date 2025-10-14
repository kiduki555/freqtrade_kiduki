# user_data/TFT_PPO_Modules/performance_metrics.py
import numpy as np

def performance_metrics(returns: np.ndarray):
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    mdd = np.min(returns.cumsum() / np.maximum.accumulate(returns.cumsum()) - 1)
    sortino = np.mean(returns) / (np.std(returns[returns < 0]) + 1e-9)
    win_rate = np.mean(returns > 0)
    return dict(sharpe=sharpe, mdd=mdd, sortino=sortino, win_rate=win_rate)
