import pandas as pd
import numpy as np
import ta
from typing import List

class FeaturePipeline:
    """
    FeaturePipeline
    ----------------
    Robust, production-grade feature engineering pipeline for OHLCV (Open, High, Low, Close, Volume) data.
    Designed for:
      - Temporal models (e.g. TFT, LSTM, Transformer)
      - Reinforcement learning agents (e.g. PPO)
    
    Core principles:
      • All indicators computed with rolling consistency (no lookahead bias)
      • Safe numeric operations (inf/nan handling)
      • Extendable, model-agnostic design
    """

    def __init__(self):
        # Expected feature names downstream (for schema alignment)
        self.features: List[str] = [
            'close','high','low','volume',
            'ema_12','ema_50','supertrend','rsi_14',
            'roc_12','price_momentum_20','atr_14',
            'bollinger_width','realized_vol_24','rolling_drawdown_48',
            'obv','vwap_distance','volume_ma_20',
            'candle_body_ratio','kurtosis_24','skewness_24'
        ]

    def _check_columns(self, df: pd.DataFrame):
        """Validate minimum required OHLCV columns before feature generation."""
        required = {'open','high','low','close','volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a consistent, non-leaky feature set from raw OHLCV data.
        Each feature is designed to capture a different aspect of market microstructure:
          - Trend, Momentum, Volatility, Volume dynamics, and Statistical shape.
        """
        self._check_columns(df)
        df = df.copy()
        df = df.sort_index()  # Ensure chronological order (critical for rolling ops)

        # === Trend indicators ===
        df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()

        # === Momentum & Oscillators ===
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['roc_12'] = df['close'].pct_change(12)                   # Short-term rate of change
        df['price_momentum_20'] = df['close'].pct_change(20)        # Medium-term momentum

        # === Volatility metrics ===
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr_14'] = atr.average_true_range()

        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bollinger_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']  # Normalized bandwidth

        # === SuperTrend (Adaptive volatility-based trend tracker) ===
        try:
            st = ta.trend.SuperTrend(high=df['high'], low=df['low'], close=df['close'],
                                     window=10, multiplier=3.0)
            df['supertrend'] = st.super_trend()
        except Exception:
            # For environments with older `ta` versions
            df['supertrend'] = np.nan

        # === Statistical volatility & distribution shape ===
        logret = np.log(df['close']).diff()
        df['realized_vol_24'] = logret.rolling(24).std()             # Realized volatility (rolling std of log returns)
        df['kurtosis_24'] = logret.rolling(24).kurt()                # Fat-tail tendency
        df['skewness_24'] = logret.rolling(24).skew()                # Asymmetry of return distribution

        # === Drawdown (relative peak decline) ===
        df['rolling_drawdown_48'] = (df['close'] / df['close'].rolling(48).max()) - 1

        # === Volume-based indicators ===
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume']
        ).on_balance_volume()

        # VWAP distance using typical price (H+L+C)/3; rolling to limit early bias
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        window_vwap = 20
        tpv = (typical_price * df['volume']).rolling(window_vwap).sum()
        vv = df['volume'].rolling(window_vwap).sum().replace(0, np.nan)
        vwap = tpv / vv
        df['vwap_distance'] = (df['close'] - vwap) / df['close']     # Relative deviation from local VWAP

        # === Volume structure & candle shape ===
        df['volume_ma_20'] = df['volume'].rolling(20).mean()         # Volume smoothing (helps remove micro noise)
        df['candle_body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-9)

        # === Data hygiene ===
        # Replace non-finite values and drop incomplete rows to ensure model stability
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna().reset_index(drop=True)

        return df
