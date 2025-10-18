from __future__ import annotations
from typing import Optional
import os, sys, numpy as np, pandas as pd, torch
from datetime import timedelta
from freqtrade.strategy import IStrategy, IntParameter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_modules'))
from feature_pipeline import FeaturePipeline
from ppo_agent import PPOAgent
from simple_tft_encoder import load_simple_tft_encoder, create_fallback_features


class PPO_TFT_Strategy_v3(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    informative_timeframe = "4h"
    startup_candle_count = 300
    process_only_new_candles = True
    can_short = False

    # === 전역 트레일링은 끕니다 (모든 엑싯은 custom_stoploss로만) ===
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = False

    # ROI는 백업(멀지도 가깝지도 않게)
    minimal_roi = {"0": 0.010, "60": 0.008, "180": 0.006, "360": 0.004}
    stoploss = -0.10  # 보험값(실제론 custom_stoploss로 관리)

    # 게이트 파라미터
    atr_pct_th = IntParameter(18, 30, default=22, space="buy")
    trend_ema_slow = IntParameter(180, 260, default=200, space="buy")

    # 모델 설정
    _WIN = 72
    _EMB_DIM = 64

    # 고정 리스크(계좌 대비)
    _risk_frac = 0.003  # 0.3%

    def __init__(self, config: dict):
        super().__init__(config)
        self.fp = FeaturePipeline()
        self.tft_path = config.get("tft_path", "user_data/models/tft_encoder.pt")
        self.ppo_path = config.get("ppo_path", "user_data/models/ppo_policy.zip")
        self.tft_encoder = None
        self.ppo_agent = None
        self._models_ready = False
        self._use_fallback = False

        # 임베딩 표준화
        self._emb_mu = None; self._emb_sigma = None; self._ema_beta = 0.05
        self._proj: Optional[np.ndarray] = None; self._proj_in_dim: Optional[int] = None

        # PPO 전환 스무딩(진입만)
        self._last_action = 0
        self._since_last_change = 10**9
        self._min_entry_interval = 2

        # 각 트레이드별 계산된 초기 SL(퍼센트)을 저장 → custom_stake_amount에서 사용
        self._last_stop_dist_pct = {}

    # --------- 인포머티브(4h) ---------
    def informative_pairs(self):
        return [(p, self.informative_timeframe) for p in self.dp.current_whitelist()]

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
        return rng.normal(0, 1.0/np.sqrt(in_dim), size=(in_dim, out_dim)).astype(np.float32)

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
        z = np.clip(z, -5, 5)
        z = z / (np.linalg.norm(z)+1e-8)
        return z.astype(np.float32)

    def _infer_action(self, window_df: pd.DataFrame) -> int:
        try:
            emb = None
            if self.tft_encoder is not None:
                xt = torch.from_numpy(window_df[self.fp.features].values.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    out = self.tft_encoder(xt)
                    if isinstance(out, dict):
                        first = next(iter(out.values()))
                        emb = (first if torch.is_tensor(first) else torch.tensor(first)).cpu().numpy().reshape(-1)
                    elif torch.is_tensor(out):
                        emb = out.cpu().numpy().reshape(-1)
                    else:
                        emb = np.array(out, dtype=np.float32).reshape(-1)
            if emb is None:
                emb = create_fallback_features(window_df)
            z = self._project_and_standardize(np.asarray(emb, dtype=np.float32).reshape(-1))
            if self.ppo_agent is None: return 0
            raw = int(self.ppo_agent.predict(z, deterministic=True))
            # 진입만 스무딩
            self._since_last_change += 1
            if raw != self._last_action:
                if self._last_action != 1 and raw == 1:
                    if self._since_last_change < self._min_entry_interval:
                        raw = self._last_action
                    else:
                        self._last_action = raw; self._since_last_change = 0
                else:
                    self._last_action = raw; self._since_last_change = 0
            return raw if raw in (0,1,2) else 0
        except Exception as e:
            print(f"[DEBUG] infer error: {e}")
            return 0

    # --------- 인디케이터/게이트 ---------
    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df = self.fp.add_features(df.copy())
        self._ensure_models_loaded()

        # 4h 추세
        info = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        ema_slow = self.trend_ema_slow.value if hasattr(self.trend_ema_slow,'value') else 200
        info['ema4h'] = info['close'].ewm(span=ema_slow, adjust=False).mean()
        info['up4h'] = (info['close'] > info['ema4h']).astype(int)
        # asof-merge로 정렬(가장 최근 4h 상태를 1h에 부여)
        info = info[['date','up4h']].rename(columns={'date':'date4h'})
        df = pd.merge_asof(df.sort_values('date'), info.sort_values('date4h'),
                           left_on='date', right_on='date4h', direction='backward')
        df['up4h'] = df['up4h'].fillna(0).astype(int)

        # 1h 변동성 / Donchian
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']).fillna(0.0)    # 비율(예: 0.01=1%)
        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low']  = df['low'].rolling(20).min()
        atr_th = (self.atr_pct_th.value if hasattr(self.atr_pct_th,'value') else 22)/100.0
        df['vol_ok'] = (df['atr_pct'] > atr_th).astype(int)

        # PPO 전환
        if 'policy_action' not in df.columns:
            df['policy_action'] = np.nan
        feats = self.fp.features
        start = max(self._WIN, int(df['policy_action'].first_valid_index() or 0))
        for i in range(start, len(df)):
            w = df[feats].iloc[i-self._WIN:i]
            df.iat[i, df.columns.get_loc("policy_action")] = self._infer_action(w)
        df['prev_action'] = df['policy_action'].shift(1).fillna(0).astype(int)
        df['enter_long_raw']  = (df['policy_action'] == 1) & (df['prev_action'] != 1)

        # 확인 진입: 다음 봉에서 신호봉 고가 재돌파
        df['signal_high'] = np.where(df['enter_long_raw'], df['high'], np.nan)
        df['signal_high'] = df['signal_high'].ffill()  # 최근 신호봉 고가 유지
        df['confirm_break'] = df['close'] > (df['signal_high'] * 1.0005)  # +0.05% 재돌파

        # 최종 엔트리 게이트
        df['buy_gate'] = (df['up4h'].eq(1) &
                          df['vol_ok'].eq(1) &
                          (df['close'] > df['donchian_high'].shift(1)) &
                          df['confirm_break'].eq(True))

        # 엔트리 시그널
        df['enter_long'] = (df['enter_long_raw'] & df['buy_gate']).astype(int)

        # 초기 SL 퍼센트(스테이크 계산용): max(0.006, 0.4*ATR)
        init_sl = np.maximum(0.006, 0.4 * df['atr_pct'].values)  # 퍼센트 단위
        df['init_sl_pct'] = init_sl

        return df

    def populate_buy_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        out = df.copy()
        out['buy'] = out['enter_long'].astype(int)
        # 체결 시점에 사용할 초기 SL%를 메모 (trade open 직후 custom_stake_amount에서 접근)
        if len(out) > 0:
            self._last_stop_dist_pct[metadata['pair']] = float(out['init_sl_pct'].iloc[-1])
        return out

    def populate_sell_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        out = df.copy()
        out['sell'] = 0
        return out

    # --------- 고정 리스크 사이징 ---------
    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_stake: float, **kwargs) -> float:
        """
        계좌의 _risk_frac(예: 0.3%)만큼을 해당 트레이드의 stop-distance로 나눈 금액.
        init_sl_pct는 populate 단계에서 계산/저장.
        """
        try:
            wallet = self.wallets.get_total_stake_amount()  # 계좌 총액
        except Exception:
            wallet = proposed_stake  # 백테일 때 대략치
        risk = wallet * self._risk_frac
        # 최근 계산한 초기 SL% 사용 (없으면 0.8% 가정)
        sl_pct = max(0.008, float(self._last_stop_dist_pct.get(pair, 0.008)))
        # 스테이크 = 리스크 / (SL 퍼센트)
        stake = risk / sl_pct
        return float(min(stake, wallet))  # 안전장치

    # --------- 커스텀 스탑(수수료 인지형 BE + 2단 트레일 + 타임스탑) ---------
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # 타임스탑
        age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if age_h >= 30:  # 30h 지나면 종료
            return 0.0

        # 초기 SL (진입 시점 init_sl_pct 기억, 없으면 0.8%)
        try:
            df = self.dp.get_analyzed_dataframe(pair)
            init_sl_pct = float(df.iloc[-1]['init_sl_pct'])
        except Exception:
            init_sl_pct = 0.008

        sl = -init_sl_pct  # 음수(%)

        # BE는 0% 금지 → +0.5% 이상일 때만 +0.25%로 승격
        if current_profit > 0.005:
            sl = max(sl, 0.0025)

        # 2단 트레일
        if current_profit > 0.008:   # +0.8%
            sl = max(sl, current_profit - 0.0035)  # 대략 +0.45% 잠금
        if current_profit > 0.012:   # +1.2%
            sl = max(sl, current_profit - 0.0045)  # 잠금 폭 확대

        return float(sl)
