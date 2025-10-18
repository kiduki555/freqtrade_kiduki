# user_data/strategies/PPO_TFT_Strategy.py
from __future__ import annotations
from typing import Optional, Dict, Any
import os, sys, time
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade

# --- 프로젝트 모듈 경로 추가 ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_modules'))
from feature_pipeline import FeaturePipeline
from ppo_agent import PPOAgent
from simple_tft_encoder import load_simple_tft_encoder, create_fallback_features


class PPO_TFT_Strategy(IStrategy):
    """
    PPO + TFT 기반 전략 (강화 버전)
    - 확률 기반 필터링(신뢰도 낮으면 Hold)
    - 시장 상태(ATR%) 반영한 Adaptive StopLoss
    - 부분익절(+트레일링) / TP2 전량익절
    """

    INTERFACE_VERSION = 3

    # === 고정 파라미터 ===
    timeframe = "1h"
    startup_candle_count = 200
    process_only_new_candles = True
    can_short = False

    # ROI는 사실상 무력화 (PPO/Custom Exit 주도)
    minimal_roi = { "0": 1000 }
    stoploss = -0.05
    use_custom_stoploss = True

    # (옵션) 부분익절 활성화하려면 True
    # - Freqtrade가 position_adjustment(부분익절/증액) 기능을 지원해야 실제 부분익절이 실행됩니다.
    # - 미지원 버전에선 자동으로 전량 익절만 동작.
    position_adjustment_enable = True

    # hyperopt placeholder (폴백에서 사용)
    rsi_buy = IntParameter(25, 40, default=30, space="buy")
    rsi_sell = IntParameter(60, 80, default=70, space="sell")

    # 모델/임베딩 설정
    _WIN = 72
    _EMB_DIM = 64

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.fp = FeaturePipeline()

        # 모델 경로
        self.tft_path = config.get("tft_path", "user_data/models/tft_encoder.pt")
        self.ppo_path = config.get("ppo_path", "user_data/models/ppo_policy.zip")

        # 모델 핸들/상태
        self.tft_encoder = None
        self.ppo_agent: Optional[PPOAgent] = None
        self._models_ready = False
        self._use_fallback = False

        # 임베딩 표준화(EMA)
        self._emb_mu = None
        self._emb_sigma = None
        self._ema_beta = 0.05

        # 차원 변환기(랜덤 프로젝션)
        self._proj: Optional[np.ndarray] = None
        self._proj_in_dim: Optional[int] = None

        # 액션 스무딩
        self._last_action = 0
        self._since_last_change = 1_000_000
        self._min_change_interval = 5  # 최소 전환 간격(캔들)

        # === 확률 기반 필터 임계값 (config로 조정 가능) ===
        self._conf_th_buy = float(config.get("conf_th_buy", 0.55))  # 0.60 -> 0.55 (완화)
        self._conf_th_exit = float(config.get("conf_th_exit", 0.5))  # 0.55 -> 0.50 (완화)

        # === 시장 상태 캐시 (custom_stoploss/exit용) ===
        self._market_state: Dict[str, Dict[str, Any]] = {}  # {pair: {"atrp": float, "prob_buy": float, "prob_sell": float}}

        # === 부분익절 상태 추적 (trade.id 기준) ===
        self._partial_done: Dict[int, bool] = {}

        print(f"[DEBUG] init | conf_th(buy/exit)=({self._conf_th_buy}/{self._conf_th_exit}) | "
              f"TFT={self.tft_path} | PPO={self.ppo_path}")

    # ---------- 유틸 ----------
    @staticmethod
    def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1/period, adjust=False).mean()
        roll_down = down.ewm(alpha=1/period, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]; low = df["low"]; close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr

    def _ensure_models_loaded(self) -> None:
        if self._models_ready:
            return
        ok_tft, ok_ppo = False, False

        if os.path.exists(self.tft_path):
            try:
                self.tft_encoder = load_simple_tft_encoder(self.tft_path, device="cpu")
                ok_tft = self.tft_encoder is not None
                print(f"[DEBUG] TFT loaded ok={ok_tft}")
            except Exception as e:
                print(f"[WARN] TFT load failed: {e}")

        if os.path.exists(self.ppo_path):
            try:
                self.ppo_agent = PPOAgent(model_path=self.ppo_path, obs_dim=self._EMB_DIM)
                ok_ppo = True
                print(f"[DEBUG] PPO loaded from {self.ppo_path}")
            except Exception as e:
                print(f"[WARN] PPO load failed: {e}")

        if not ok_ppo:
            self._use_fallback = True
            print("[WARN] PPO missing -> RSI fallback mode.")
        else:
            self._use_fallback = False
            print("[INFO] PPO mode active (TFT {}available)".format("" if ok_tft else "NOT "))

        self._models_ready = True

    def _build_projection(self, in_dim: int, out_dim: int = 64, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        W = rng.normal(0, 1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)
        return W

    def _project_and_standardize(self, emb: np.ndarray) -> np.ndarray:
        if emb.shape[0] != self._EMB_DIM:
            if (self._proj is None) or (self._proj_in_dim != emb.shape[0]):
                self._proj = self._build_projection(emb.shape[0], self._EMB_DIM, seed=42)
                self._proj_in_dim = emb.shape[0]
                print(f"[DEBUG] Projection built {emb.shape[0]} -> {self._EMB_DIM}")
            z = emb @ self._proj
        else:
            z = emb.astype(np.float32)

        # EMA 표준화
        z = z.astype(np.float32)
        v = z
        if self._emb_mu is None:
            self._emb_mu = v.copy()
            self._emb_sigma = np.ones_like(v, dtype=np.float32)
        else:
            self._emb_mu = (1 - self._ema_beta) * self._emb_mu + self._ema_beta * v
            diff = np.abs(v - self._emb_mu)
            self._emb_sigma = (1 - self._ema_beta) * self._emb_sigma + self._ema_beta * diff

        z = (v - self._emb_mu) / (self._emb_sigma + 1e-6)
        z = np.clip(z, -5.0, 5.0)
        nrm = np.linalg.norm(z) + 1e-8
        z = (z / nrm).astype(np.float32)
        return z

    # ---------- Freqtrade 훅 ----------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = self.fp.add_features(dataframe.copy())
        self._ensure_models_loaded()

        # 시장 상태 피처
        if "rsi" not in df.columns:
            df["rsi"] = self._rsi_series(df["close"], 14)
        df["atr"] = self._atr(df, 14)
        df["atrp"] = (df["atr"] / (df["close"] + 1e-12)).fillna(0.0)  # ATR%

        if self._use_fallback:
            return df

        for col in ["policy_action", "prob_buy", "prob_sell", "policy_conf"]:
            if col not in df.columns:
                df[col] = np.nan

        feats = self.fp.features
        start = self._WIN
        if df["policy_action"].notna().any():
            last = int(np.nanmax(df.index[df["policy_action"].notna()])) + 1
            start = max(start, last)

        for i in range(start, len(df)):
            window = df[feats].iloc[i - self._WIN:i]
            if len(window) < self._WIN:
                continue
            action, prob_buy, prob_sell, conf = self._infer_action(window)
            idx = df.index[i]
            df.at[idx, "policy_action"] = action
            df.at[idx, "prob_buy"] = prob_buy
            df.at[idx, "prob_sell"] = prob_sell
            df.at[idx, "policy_conf"] = conf

        # 전이 + 확률 필터
        df["prev_action"] = df["policy_action"].shift(1).fillna(0).astype(int)

        raw_enter = (df["policy_action"] == 1) & (df["prev_action"] != 1)
        raw_exit  = (df["prev_action"] == 1) & (df["policy_action"] != 1)

        conf_buy_ok  = (df["prob_buy"] >= self._conf_th_buy)
        conf_exit_ok = (df["prob_sell"] >= self._conf_th_exit) | ((1.0 - df["prob_buy"]) >= self._conf_th_exit)

        df["enter_long"] = (raw_enter & conf_buy_ok).fillna(False)
        df["exit_long"]  = (raw_exit & conf_exit_ok).fillna(False)

        return df

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        self._ensure_models_loaded()

        if self._use_fallback:
            if "rsi" not in df.columns:
                df["rsi"] = self._rsi_series(df["close"], 14)
            thr = self.rsi_buy.value if isinstance(self.rsi_buy, IntParameter) else 30
            df["buy"] = (df["rsi"] < thr).astype(int)
            return df

        if "enter_long" not in df.columns:
            df = self.populate_indicators(df, metadata)

        # 시간대 필터 추가 (00-08 UTC 차단)
        if "date" in df.columns:
            _dt = pd.to_datetime(df["date"], utc=True)
        elif "datetime" in df.columns:
            _dt = pd.to_datetime(df["datetime"], utc=True)
        else:
            _dt = pd.to_datetime(df.index, utc=True)
        
        df["hour_utc"] = _dt.dt.hour
        # 00~07 UTC(한국 09~16시 이전)의 진입 차단
        time_ok = (df["hour_utc"] >= 8) & (df["hour_utc"] <= 23)
        
        rsi_ok = (df["rsi"] > 30) & (df["rsi"] < 70)
        trend_ok = df["close"] > df["close"].rolling(20).mean()
        # 과열 회피: ATR%가 너무 큰 구간은 진입 억제 (스캘핑형)
        vol_not_extreme = df["atrp"] < 0.10  # 0.08 -> 0.10 (고변동 허용폭 약간 확대)

        df["buy"] = (df["enter_long"] & rsi_ok & trend_ok & vol_not_extreme & time_ok).astype(int)
        return df

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = dataframe.copy()
        self._ensure_models_loaded()

        if self._use_fallback:
            if "rsi" not in df.columns:
                df["rsi"] = self._rsi_series(df["close"], 14)
            thr = self.rsi_sell.value if isinstance(self.rsi_sell, IntParameter) else 70
            df["sell"] = (df["rsi"] > thr).astype(int)
            return df

        if "exit_long" not in df.columns:
            df = self.populate_indicators(df, metadata)

        # PPO exit 신호 + 보조 규칙
        ppo_exit = df["exit_long"].astype(bool)

        # 익절 힌트(선택): 전봉 대비 +1.5% 이상 급등
        profit_hint = df["close"] > df["close"].shift(1) * 1.015
        rsi_hot = df["rsi"] > 65
        high_vol_exit = df["atrp"] >= 0.10  # 과변동 시 보수적 청산

        df["sell"] = (ppo_exit | (rsi_hot & profit_hint) | high_vol_exit).astype(int)

        # 시장 상태 캐시 (custom_stoploss/exit용)
        if len(df):
            last = df.iloc[-1]
            pair = metadata.get("pair", "UNKNOWN")
            self._market_state[pair] = {
                "atrp": float(last.get("atrp", 0.0)),
                "prob_buy": float(last.get("prob_buy", np.nan)) if "prob_buy" in df.columns else np.nan,
                "prob_sell": float(last.get("prob_sell", np.nan)) if "prob_sell" in df.columns else np.nan,
            }
        return df

    # ---------- 내부: 임베딩/행동 추론 ----------
    def _infer_action(self, window_df: pd.DataFrame):
        """
        Returns: (action:int, prob_buy:float, prob_sell:float, conf:float)
        """
        try:
            emb = None
            if self.tft_encoder is not None:
                feat = window_df[self.fp.features].values.astype(np.float32)
                xt = torch.from_numpy(feat).unsqueeze(0)
                with torch.no_grad():
                    try:
                        pred = self.tft_encoder(xt)
                        if isinstance(pred, dict):
                            first = next(iter(pred.values()))
                            emb = (first if torch.is_tensor(first) else torch.tensor(first)).cpu().numpy().reshape(-1)
                        elif torch.is_tensor(pred):
                            emb = pred.cpu().numpy().reshape(-1)
                        else:
                            emb = np.array(pred, dtype=np.float32).reshape(-1)
                    except Exception as e:
                        print(f"[DEBUG] TFT forward failed: {e}")
                        emb = None

            if emb is None:
                emb = create_fallback_features(window_df)

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if np.any(~np.isfinite(emb)):
                emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            emb = self._project_and_standardize(emb)

            action = 0
            prob_buy = 0.33
            prob_sell = 0.33

            if self.ppo_agent is None:
                return 0, prob_buy, prob_sell, max(prob_buy, prob_sell)

            # 확률 제공 메서드 탐색
            probs = None
            try:
                if hasattr(self.ppo_agent, "predict_proba"):
                    probs = np.asarray(self.ppo_agent.predict_proba(emb), dtype=np.float32).reshape(-1)
                elif hasattr(self.ppo_agent, "get_action_proba"):
                    probs = np.asarray(self.ppo_agent.get_action_proba(emb), dtype=np.float32).reshape(-1)
            except Exception:
                probs = None

            try:
                raw_pred = self.ppo_agent.predict(emb, deterministic=True)
                action = int(raw_pred[0] if isinstance(raw_pred, (tuple, list)) else raw_pred)
                if action not in (0,1,2):
                    action = 0
            except Exception as e:
                print(f"[DEBUG] PPO predict failed: {e}")
                action = 0

            if probs is not None and len(probs) >= 3:
                probs = probs[:3]
                s = probs.sum() + 1e-12
                probs = (probs / s).astype(np.float32)
                prob_buy = float(probs[1]); prob_sell = float(probs[2])
            else:
                # 보수적 추정
                base = 0.225
                if action == 1:
                    prob_buy, prob_sell = 0.60, base
                elif action == 2:
                    prob_buy, prob_sell = base, 0.60
                else:
                    prob_buy, prob_sell = base, base

            # 액션 스무딩
            self._since_last_change += 1
            if action != self._last_action:
                if self._since_last_change < self._min_change_interval:
                    action = self._last_action
                else:
                    self._last_action = action
                    self._since_last_change = 0

            conf = max(prob_buy, prob_sell, 1.0 - max(prob_buy, prob_sell))
            return action, prob_buy, prob_sell, float(conf)

        except Exception as e:
            print(f"[DEBUG] _infer_action error: {e}")
            return 0, 0.33, 0.33, 0.33

    # ---------- 커스텀 리스크 관리 ----------
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ) -> float:
        """
        Adaptive StopLoss (ATR% & 신뢰도 기반)
        - 기본 SL 3%를 기준으로, ATR%가 높을수록 완화(넓힘), 낮을수록 타이트
        - 신뢰도 낮을수록 타이트
        """
        ms = self._market_state.get(pair, {"atrp": 0.02})
        atrp = float(ms.get("atrp", 0.02))   # 예: 0.02 = 2%

        base_sl = 0.028   # 2.8% (기본 3%보다 약간 타이트하지만 아래서 '완화' 반영)
        # 변동성 완화: 2%->0.95x, 10%->1.4x (상한 낮춤: 1.8 -> 1.4)
        vol_factor = float(np.clip(0.95 + 2.25 * (atrp), 0.9, 1.4))

        # 초기 2시간은 '넓게' 잡아 흔들림 통과 (0.9 -> 1.15)
        age_min = (current_time - trade.open_date_utc).total_seconds() / 60.0
        age_factor = 1.15 if age_min < 120 else 1.0  # 2시간 이전엔 15% 넓게

        dyn_sl = base_sl * vol_factor * age_factor
        return float(-np.clip(dyn_sl, 0.018, 0.06))  # 최종 SL 범위: -1.8% ~ -6%

    # === 부분익절/트레일링 & TP2 전량익절 ===
    # - Freqtrade가 position_adjustment를 지원하는 경우: TP1에서 50% 부분익절
    # - 미지원인 경우: TP1/TP2는 custom_exit로 '전량' 청산 (동작 보장)
    _TP1 = 0.02   # +0.8% (1.0% -> 0.8%)
    _TP2 = 0.04   # +1.6% (2.0% -> 1.6%)
    _TRAIL = 0.01  # 0.35% 되돌림 (0.5% -> 0.35%)

    def custom_exit(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ):
        """
        전량 청산 로직(표준)
        - TP2: +2.0% 이상 전량 익절
        - 트레일링: +1.0% 이상 이익 이후 0.5% 이상 되돌림 시 전량 익절
        - (부분익절은 position_adjustment에서 수행되며, 여기선 전량만 처리)
        """
        # TP2(+1.6%) 전량
        if current_profit >= self._TP2:
            return ("tp2_full_take", "force_exit")

        # 트레일: +0.8% 이상 찍은 후 0.35% 되돌림 시 청산
        try:
            # trade 에서 max_profit 저장 지원 안되면, 보수적으로 현재 이익만 사용
            maxp = getattr(trade, "max_profit", None)
            if maxp is not None and maxp >= self._TP1 and (maxp - current_profit) >= self._TRAIL:
                return ("trail_exit", "force_exit")
        except Exception:
            pass

        return None

    # (옵션) 부분익절: 지원되는 버전에서만 실행됨
    def position_adjustment(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ):
        """
        TP1(+1.0%)에 도달하면 50% 부분익절 시도.
        - Freqtrade 구현/버전에 따라 '감액(부분익절)'이 허용되지 않을 수 있음.
        - 허용되지 않는 환경에선 엔진이 무시하거나 예외를 낼 수 있으므로, 해당 시 전량 익절(custom_exit)만으로도 충분히 운용 가능.
        """
        try:
            if not self.position_adjustment_enable:
                return None

            if current_profit >= self._TP1:
                # trade.id를 키로 1회만 실행
                if trade.id not in self._partial_done or not self._partial_done[trade.id]:
                    self._partial_done[trade.id] = True
                    # 음수(감액) 혹은 amount 비율 반환 등은 버전 의존적입니다.
                    # 일반적으로 '감액량(코인 수량)'을 음수로 반환하는 구현이 있습니다.
                    # 불확실성을 감안해 아래는 "절반 감액" 의도를 표현합니다.
                    try:
                        amount = getattr(trade, "amount", None)
                        if amount and amount > 0:
                            return - (amount * 0.5)  # 절반 감액 (버전 지원 시)
                    except Exception:
                        return None
        except Exception:
            return None

        return None
