from __future__ import annotations
from typing import Optional
import os, sys, numpy as np, pandas as pd, torch
from freqtrade.strategy import IStrategy, IntParameter
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_modules'))
from feature_pipeline import FeaturePipeline
from ppo_agent import PPOAgent
from simple_tft_encoder import load_simple_tft_encoder, create_fallback_features

class PPO_TFT_Strategy_v2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    informative_timeframe = "4h"
    startup_candle_count = 300
    process_only_new_candles = True
    can_short = False

    # 엔트리/엑싯 정책
    use_exit_signal = False  # 엑싯은 custom_stoploss + 타임스탑으로만
    ignore_roi_if_entry_signal = False

    # ROI는 백업 — 멀지도, 가깝지도 않게 (트레일링이 주력)
    minimal_roi = {"0": 0.010, "60": 0.008, "180": 0.006, "360": 0.004}
    stoploss = -0.10  # 실제로는 custom_stoploss가 관리 (이 값은 보험)

    # 트레일링(전역): 트레일링이 먼저 작동하도록 낮춘 오프셋
    trailing_stop = True
    trailing_stop_positive = 0.0025      # 0.25%
    trailing_stop_positive_offset = 0.0035  # +0.35%에 활성 → 즉시 양수 SL 확보
    trailing_only_offset_is_reached = True

    # 하이퍼옵트 / 게이트 기본값
    prob_th = IntParameter(45, 60, default=52, space="buy")  # PPO buy-prob(%) 게이트
    atr_pct_th = IntParameter(18, 30, default=22, space="buy")
    trend_ema_fast = IntParameter(80, 120, default=100, space="buy")
    trend_ema_slow = IntParameter(180, 260, default=200, space="buy")

    # 커스텀 스탑
    use_custom_stoploss = True
    use_custom_stop = True  # Freqtrade 2025 기준 custom_stoploss만 써도 됨 (타임스탑은 내부 처리)

    # 모델 설정
    _WIN = 72
    _EMB_DIM = 64

    def __init__(self, config: dict):
        super().__init__(config)
        self.fp = FeaturePipeline()
        self.tft_path = config.get("tft_path", "user_data/models/tft_encoder.pt")
        self.ppo_path = config.get("ppo_path", "user_data/models/ppo_policy.zip")
        self.tft_encoder = None
        self.ppo_agent = None
        self._models_ready = False
        self._use_fallback = False
        self._emb_mu = None; self._emb_sigma = None; self._ema_beta = 0.05
        self._proj: Optional[np.ndarray] = None; self._proj_in_dim: Optional[int] = None

        # 스무딩: 진입만 최소 간격, 청산은 규칙이 즉시
        self._last_action = 0
        self._since_last_change = 10**9
        self._min_entry_interval = 2

    # --------- 인포머티브(4h) ---------
    def informative_pairs(self):
        pairs = [(p, self.informative_timeframe) for p in self.dp.current_whitelist()]
        return pairs

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = self.fp.add_features(df.copy())
        self._ensure_models_loaded()

        # 4h 추세 필터
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        ema_slow = self.trend_ema_slow.value if hasattr(self.trend_ema_slow, 'value') else 200
        informative['ema4h'] = informative['close'].ewm(span=ema_slow, adjust=False).mean()
        informative['up4h'] = (informative['close'] > informative['ema4h']).astype(int)
        df = df.merge(informative[['date','up4h']].rename(columns={'date':'date_4h'}),
                      left_on='date', right_on='date_4h', how='left')
        df['up4h'] = df['up4h'].ffill().fillna(0).astype(int)

        # 1h 변동성
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close'] * 100).fillna(0)
        atr_th = (self.atr_pct_th.value if hasattr(self.atr_pct_th,'value') else 22)/100.0
        df['vol_ok'] = (df['atr_pct']/100.0 > atr_th).astype(int)

        # PPO 액션(벡터화 대신 간단 캐시, 필요시 배치화로 확장)
        if 'policy_action' not in df.columns:
            df['policy_action'] = np.nan
        feats = self.fp.features
        start = max(self._WIN, int(df['policy_action'].first_valid_index() or 0))
        for i in range(start, len(df)):
            w = df[feats].iloc[i-self._WIN:i]
            df.iat[i, df.columns.get_loc("policy_action")] = self._infer_action(w)

        df['prev_action'] = df['policy_action'].shift(1).fillna(0).astype(int)
        df['enter_long']  = (df['policy_action'] == 1) & (df['prev_action'] != 1)
        return df

    def populate_buy_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = df.copy()
        if self._use_fallback:
            # 예비: RSI 단순 규칙
            rsi = self._rsi_series(df['close'])
            df['buy'] = ((rsi<40).astype(int))
            return df

        # 게이트: PPO 엔트리 + 4h 상승 + 변동성 OK
        ppo = df['enter_long'].astype(int)
        prob_th = (self.prob_th.value if hasattr(self.prob_th,'value') else 52)/100.0
        # PPOAgent가 확률 제공 못한다면, 시그널 스무딩으로 대체 (여기선 엔트리 전환만)
        trend_ok = df['up4h'] == 1
        vol_ok   = df['vol_ok'] == 1

        # 간단: PPO 전환 & 4h up & vol_ok
        df['buy'] = (ppo & trend_ok & vol_ok).astype(int)
        return df

    def populate_sell_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 엑싯은 custom_stoploss / trailing / ROI가 처리 → 여기선 no-op
        df = df.copy()
        df['sell'] = 0
        return df

    # --------- 수수료 인지형 BE + ATR 트레일 + 타임스탑 ---------
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """
        목표:
        - BE=0% 금지 (수수료 고려)
        - +0.35% 근처에서 즉시 '양수' SL 확보 (trailing_offset과 조응)
        - 2단 트레일: 초반 미세(0.25%), 이후 강(0.3~0.4%)
        - 타임스탑: 24~36캔들 넘어가면 청산
        """
        # 1) 타임스탑
        max_bars = 30  # 30h
        age = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if age >= max_bars:
            return 0.0  # 즉시 청산

        # 2) 기본 SL (초기 보호) - 얕게
        sl = -0.006  # -0.6%

        # 3) 트레일링 활성 이전엔 BE를 0%로 올리지 말고, -0.2% 정도로만 당겨서 헛손실 방지
        if current_profit > 0.0025:   # +0.25%
            sl = max(sl, -0.002)

        # 4) 트레일링 활성 영역(+0.35%~)
        if current_profit > 0.0035:
            # 초반 미세 트레일 (약 +0.10% SL 근사)
            sl = max(sl, current_profit - 0.0025)

        if current_profit > 0.0080:   # +0.8% 이후 강한 트레일
            sl = max(sl, current_profit - 0.0035)  # 이익 더 끌기

        return sl

    # --------- 내부 유틸 ---------
    @staticmethod
    def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
        d = close.diff()
        up = d.clip(lower=0); dn = (-d).clip(lower=0)
        ru = up.ewm(alpha=1/period, adjust=False).mean()
        rd = dn.ewm(alpha=1/period, adjust=False).mean()
        rs = ru/(rd+1e-12)
        return 100 - (100/(1+rs))

    def _ensure_models_loaded(self):
        if self._models_ready: return
        ok_tft = ok_ppo = False
        if os.path.exists(self.tft_path):
            try:
                self.tft_encoder = load_simple_tft_encoder(self.tft_path, device="cpu")
                ok_tft = self.tft_encoder is not None
            except Exception as e:
                print(f"[WARN] TFT load failed: {e}")
        if os.path.exists(self.ppo_path):
            try:
                self.ppo_agent = PPOAgent(model_path=self.ppo_path, obs_dim=self._EMB_DIM)
                ok_ppo = True
            except Exception as e:
                print(f"[WARN] PPO load failed: {e}")
        self._use_fallback = not ok_ppo
        self._models_ready = True

    def _build_projection(self, in_dim: int, out_dim: int=64, seed: int=42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        W = rng.normal(0, 1.0/np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
        return W

    def _project_and_standardize(self, emb: np.ndarray) -> np.ndarray:
        if emb.shape[0] != self._EMB_DIM:
            if (self._proj is None) or (self._proj_in_dim != emb.shape[0]):
                self._proj = self._build_projection(emb.shape[0], self._EMB_DIM, seed=42)
                self._proj_in_dim = emb.shape[0]
            z = emb @ self._proj
        else:
            z = emb.astype(np.float32)
        v = z.astype(np.float32)
        if self._emb_mu is None:
            self._emb_mu = v.copy(); self._emb_sigma = np.ones_like(v, dtype=np.float32)
        else:
            self._emb_mu = (1-self._ema_beta)*self._emb_mu + self._ema_beta*v
            diff = np.abs(v - self._emb_mu)
            self._emb_sigma = (1-self._ema_beta)*self._emb_sigma + self._ema_beta*diff
        z = (v - self._emb_mu) / (self._emb_sigma + 1e-6)
        z = np.clip(z, -5, 5); z = z / (np.linalg.norm(z)+1e-8)
        return z.astype(np.float32)

    def _infer_action(self, window_df: pd.DataFrame) -> int:
        try:
            emb = None
            if self.tft_encoder is not None:
                xt = torch.from_numpy(window_df[self.fp.features].values.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    out = self.tft_encoder(xt)
                    if isinstance(out, dict): first = next(iter(out.values())); emb = (first if torch.is_tensor(first) else torch.tensor(first)).cpu().numpy().reshape(-1)
                    elif torch.is_tensor(out): emb = out.cpu().numpy().reshape(-1)
                    else: emb = np.array(out, dtype=np.float32).reshape(-1)
            if emb is None:
                emb = create_fallback_features(window_df)
            z = self._project_and_standardize(np.asarray(emb, dtype=np.float32).reshape(-1))
            if self.ppo_agent is None: return 0
            raw = int(self.ppo_agent.predict(z, deterministic=True))
            # 진입만 스무딩
            self._since_last_change += 1
            if raw != self._last_action:
                if self._last_action != 1 and raw == 1:
                    if self._since_last_change < self._min_entry_interval: raw = self._last_action
                    else: self._last_action = raw; self._since_last_change = 0
                else:
                    self._last_action = raw; self._since_last_change = 0
            return raw if raw in (0,1,2) else 0
        except Exception as e:
            print(f"[DEBUG] infer error: {e}")
            return 0

    # --------- 변동성 기반 스테이크 조절(선택) ---------
    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_stake: float, **kwargs) -> float:
        """
        ATR%가 높을수록 스테이크 축소 → 손실 변동성 균등화
        """
        try:
            df = self.dp.get_analyzed_dataframe(pair)
            atrp = float(df.iloc[-1]['atr_pct'])  # % 
            # 기준 1.0%면 1.0로 환산
            scale = 1.0 / max(0.7, min(1.5, atrp))  # 0.7~1.5 범위에 역비례
            return proposed_stake * scale
        except Exception:
            return proposed_stake
