# user_data/strategies/PPO_TFT_Strategy.py
from __future__ import annotations

from freqtrade.strategy import IStrategy, IntParameter
from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.ppo_agent import PPOAgent
from TFT_PPO_modules.tft_encoder import TFTEncoder

import pandas as pd
import numpy as np
import os
import warnings


class PPO_TFT_Strategy(IStrategy):
    """
    Freqtrade Strategy
    -------------------
    - TFT(Temporal Fusion Transformer)로 시계열 창(window)의 상태를 임베딩
    - PPO 정책으로 액션 추론(0=Hold, 1=Buy, 2=Sell)
    - 인디케이터 단계에서 policy_action을 한 번만 계산하여 buy/sell에서 재사용
    """

    INTERFACE_VERSION = 3

    # 기본 파라미터 (필요시 config로 대체)
    minimal_roi = {"0": 10}
    stoploss = -0.10
    timeframe = "1h"
    startup_candle_count = 200
    process_only_new_candles = True

    # 선택적 임계치 (수동 오버라이드 가능)
    rsi_buy = IntParameter(25, 40, default=30, space="buy")
    rsi_sell = IntParameter(60, 80, default=70, space="sell")

    # 전략 상수
    _WIN = 72           # TFT 인코딩에 사용할 창 길이
    _EMB_DIM = 64       # TFTEncoder가 반환하는 상태 벡터 차원 (PPO 입력과 일치)

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Feature pipeline
        self.fp = FeaturePipeline()

        # 모델 경로 (필요시 config에서 주입)
        self.tft_path = config.get("tft_path", "user_data/models/best/tft_best.pt")
        self.ppo_path = config.get("ppo_path", "user_data/models/best/ppo_best.zip")

        # 경로 검증
        if not os.path.exists(self.tft_path):
            raise FileNotFoundError(f"TFT model not found at: {self.tft_path}")
        if not os.path.exists(self.ppo_path):
            raise FileNotFoundError(f"PPO model not found at: {self.ppo_path}")

        # 모델 로드
        # 주의: TFTEncoder는 체크포인트 형식(load_from_checkpoint)을 기대할 수 있음.
        # 학습 스크립트에서 state_dict(.pt)로 저장했다면, TFTEncoder 내부 구현이 호환되는지 확인 필요.
        try:
            self.tft_encoder = TFTEncoder(model_path=self.tft_path)
        except Exception as e:
            warnings.warn(f"TFTEncoder load failed ({e}). Strategy will fallback to zero embeddings.")
            self.tft_encoder = None

        self.ppo_agent = PPOAgent(model_path=self.ppo_path, obs_dim=self._EMB_DIM)

    # ===========================
    # 인디케이터 생성
    # ===========================
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        - 피처 생성
        - policy_action 캐시(윈도우 확보된 행부터만 계산)
        - 이미 존재하면 새 구간만 갱신하여 불필요한 재계산 방지
        """
        df = self.fp.add_features(dataframe.copy())

        # 캐시 컬럼 초기화/존재 여부 확인
        if "policy_action" not in df.columns:
            df["policy_action"] = np.nan

        # 계산 시작 인덱스 결정: 기존 계산된 마지막 인덱스 이후부터
        last_idx = (
            int(np.nanmax(df.index[df["policy_action"].notna()])) + 1
            if df["policy_action"].notna().any()
            else self._WIN
        )
        start = max(last_idx, self._WIN)

        if start < len(df):
            # 순차적으로 새 구간에 대해만 임베딩/액션 계산
            # 주의: Freqtrade는 인디케이터 단계가 각 스텝마다 호출되므로 과도한 연산을 피해야 함
            for i in range(start, len(df)):
                window = df[self.fp.features].iloc[i - self._WIN : i]
                if len(window) < self._WIN:
                    continue
                action = self._infer_action(window)
                df.iat[i, df.columns.get_loc("policy_action")] = action

        return df

    # ===========================
    # 내부: 임베딩/액션 추론
    # ===========================
    def _infer_action(self, window_df: pd.DataFrame) -> int:
        """
        주어진 윈도우에 대해 TFT 임베딩 -> PPO 행동 추론
        실패 시 0(Hold) 반환
        """
        if self.tft_encoder is None:
            return 0
        try:
            emb = self.tft_encoder.encode(window_df)
            # 형태 보정 및 예외 방어
            if emb is None or np.ndim(emb) == 0:
                return 0
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.shape[0] != self._EMB_DIM:
                # 임베딩 차원이 예기치 않다면 안전하게 Hold
                return 0
            return int(self.ppo_agent.predict(emb, deterministic=True))
        except Exception:
            return 0

    # ===========================
    # 매수 시그널
    # ===========================
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        policy_action == 1 인 구간에 buy=1
        """
        df = dataframe.copy()

        # policy_action이 없으면 인디케이터를 먼저 보장
        if "policy_action" not in df.columns:
            df = self.populate_indicators(df, metadata)

        df["buy"] = 0
        mask = (df.index >= self._WIN) & (df["policy_action"] == 1)
        df.loc[mask, "buy"] = 1
        return df

    # ===========================
    # 매도 시그널
    # ===========================
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        policy_action == 2 인 구간에 sell=1
        """
        df = dataframe.copy()

        if "policy_action" not in df.columns:
            df = self.populate_indicators(df, metadata)

        df["sell"] = 0
        mask = (df.index >= self._WIN) & (df["policy_action"] == 2)
        df.loc[mask, "sell"] = 1
        return df
