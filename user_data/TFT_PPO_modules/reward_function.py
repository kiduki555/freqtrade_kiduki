import numpy as np

def custom_reward(
    pnl: float,
    vol: float,
    dd: float,
    kurtosis: float,
    skewness: float,
    trade_freq: float = 0,
    target_trades: float = 5,
    normalize: bool = True
) -> float:
    """
    Reward function for trading RL agents (PPO/TFT hybrid).
    
    Combines profitability and risk-awareness:
      - Rewards high PnL and positive skewness
      - Penalizes volatility, drawdowns, fat tails, and overtrading
    """

    # === Weight coefficients ===
    λ_vol, λ_dd, λ_kurt, λ_skew, λ_freq = 0.4, 0.25, 0.1, 0.05, 0.1

    # === Base reward ===
    reward = (
        pnl
        - λ_vol * vol
        - λ_dd * abs(dd)
        - λ_kurt * max(kurtosis, 0)   # only penalize fat tails
        + λ_skew * skewness
        - λ_freq * max(trade_freq - target_trades, 0)
    )

    # === Normalization for PPO stability ===
    if normalize:
        reward = np.tanh(reward)  # keep within [-1, 1] range to stabilize gradient updates

    return float(reward)
