import gymnasium as gym
import numpy as np
import torch
from TFT_PPO_modules.reward_function import custom_reward


class TradingEnv(gym.Env):
    """
    TradingEnv (Gymnasium-compatible)
    ---------------------------------
    Reinforcement Learning environment for algorithmic trading.
    Designed for TFT + PPO hybrid setups.

    Key Improvements:
      - Gymnasium API compliant
      - Mark-to-market PnL computation
      - Position persistence (Hold/Buy/Sell)
      - Transaction cost & slippage modeling
      - Risk-aware reward shaping via custom_reward()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        tft_model,
        features,
        window: int = 72,
        reward_func=custom_reward,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        device: str = "cpu",
    ):
        super().__init__()

        # === Core dataset & model ===
        self.df = df.reset_index(drop=True)
        self.tft = tft_model
        self.features = features
        self.window = window
        self.reward_func = reward_func
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.device = torch.device(device)

        # === Environment state ===
        self.current_step = window
        self.position = 0  # -1: short, 0: flat, +1: long
        self.trades_last_n = []  # store trade frequency

        # === Spaces (Gymnasium style) ===
        self.action_space = gym.spaces.Discrete(3)  # 0=Hold, 1=Buy(Long), 2=Sell(Short)
        
        # === obs_dim을 데이터에서 동적으로 계산 ===
        self.obs_dim = self.window * len(self.features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    # -----------------------------
    # State extraction
    # -----------------------------
    def _get_state(self):
        """Compute state representation - flatten window features."""
        # 최근 window를 꺼내 1D로 평탄화
        window_df = self.df[self.features].iloc[self.current_step - self.window : self.current_step]
        x = window_df.values.astype(np.float32).reshape(-1)   # (window*features,)
        return x

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, action):
        terminated = False
        truncated = False

        if self.current_step >= len(self.df) - 2:
            truncated = True
            return self._get_state(), 0.0, terminated, truncated, {}

        curr_price = self.df["close"].iloc[self.current_step]
        next_price = self.df["close"].iloc[self.current_step + 1]

        # === Position/PnL Dynamics ===
        prev_position = self.position
        if action == 1:
            self.position = 1   # Long
        elif action == 2:
            self.position = -1  # Short
        else:
            self.position = self.position  # Hold

        # Price change (return)
        price_ret = (next_price - curr_price) / curr_price

        # PnL depends on previous position
        pnl = price_ret * prev_position if prev_position != 0 else 0

        # Transaction cost if position changed
        if self.position != prev_position:
            pnl -= self.transaction_cost + self.slippage * abs(price_ret)

        # === Risk and statistical metrics ===
        vol = float(self.df["realized_vol_24"].iloc[self.current_step])
        dd = float(abs(self.df["rolling_drawdown_48"].iloc[self.current_step]))
        kurt = float(self.df["kurtosis_24"].iloc[self.current_step])
        skew = float(self.df["skewness_24"].iloc[self.current_step])

        # === Trade frequency memory ===
        trade_flag = 1 if action != 0 else 0
        self.trades_last_n.append(trade_flag)
        if len(self.trades_last_n) > 20:
            self.trades_last_n.pop(0)
        trade_freq = np.mean(self.trades_last_n)

        # === Reward ===
        reward = self.reward_func(pnl, vol, dd, kurt, skew, trade_freq)

        # === Step advance ===
        self.current_step += 1
        if self.current_step >= len(self.df) - 2:
            terminated = True

        next_state = self._get_state()
        info = {
            "pnl": pnl,
            "vol": vol,
            "drawdown": dd,
            "position": self.position,
            "trade_freq": trade_freq,
        }

        return next_state, float(reward), terminated, truncated, info

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.position = 0
        self.trades_last_n.clear()
        obs = self._get_state()
        info = {"position": self.position}
        return obs, info

    # -----------------------------
    # Render
    # -----------------------------
    def render(self):
        """Optional visualization hook for backtesting or live tracking."""
        step_data = self.df.iloc[self.current_step]
        print(
            f"Step {self.current_step}: Price={step_data['close']:.2f}, Position={self.position}, "
            f"Drawdown={step_data['rolling_drawdown_48']:.4f}"
        )
