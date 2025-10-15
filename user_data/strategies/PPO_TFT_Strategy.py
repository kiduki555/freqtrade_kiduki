# user_data/strategies/PPO_TFT_Strategy.py
from __future__ import annotations
from typing import Optional

from freqtrade.strategy import IStrategy, IntParameter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_modules'))

from feature_pipeline import FeaturePipeline
from ppo_agent import PPOAgent
from simple_tft_encoder import load_simple_tft_encoder, create_fallback_features

import pandas as pd
import numpy as np
import os
import warnings
import torch


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
        self.tft_path = config.get("tft_path", "user_data/models/tft_encoder.pt")
        self.ppo_path = config.get("ppo_path", "user_data/models/best/best_sharpe_0.288.zip")

        # 경로 검증
        if not os.path.exists(self.tft_path):
            raise FileNotFoundError(f"TFT model not found at: {self.tft_path}")
        if not os.path.exists(self.ppo_path):
            raise FileNotFoundError(f"PPO model not found at: {self.ppo_path}")

        # 모델 로드
        try:
            self.tft_encoder = load_simple_tft_encoder(self.tft_path, device="cpu")
            if self.tft_encoder is None:
                raise RuntimeError("SimpleTFTEncoder 로드 실패")
            print(f"[DEBUG] TFT Encoder 로드 성공: {self.tft_path}")
        except Exception as e:
            warnings.warn(f"TFTEncoder load failed ({e}). Strategy will use fallback features.")
            self.tft_encoder = None

        # PPO 에이전트 로드 (CPU 강제)
        try:
            self.ppo_agent = PPOAgent(model_path=self.ppo_path, obs_dim=self._EMB_DIM)
            print(f"[DEBUG] PPO Agent 로드 성공: {self.ppo_path}")
        except Exception as e:
            warnings.warn(f"PPO Agent load failed ({e}). Strategy will use random actions.")
            self.ppo_agent = None

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

        # 변화 기반 신호 생성 (매도 남발 방지)
        df["prev_action"] = df["policy_action"].shift(1).fillna(0).astype(int)
        
        # 엔트리는 '변화'일 때만 (전이 기반)
        df["enter_long"] = (df["policy_action"] == 1) & (df["prev_action"] != 1)
        df["enter_short"] = (df["policy_action"] == 2) & (df["prev_action"] != 2)
        
        # 롱 보유 중 → 롱 종료는 action이 1에서 벗어날 때만
        df["exit_long"] = (df["prev_action"] == 1) & (df["policy_action"] != 1)
        
        # 숏 보유 중 → 숏 종료는 action이 2에서 벗어날 때만  
        df["exit_short"] = (df["prev_action"] == 2) & (df["policy_action"] != 2)
        
        # 액션 분포 디버그 로그
        valid_actions = df["policy_action"].dropna()
        if len(valid_actions) > 0:
            action_counts = valid_actions.value_counts().sort_index()
            print(f"[DEBUG] Action distribution: {dict(action_counts)}")
            print(f"[DEBUG] Action percentages: Flat={action_counts.get(0,0)/len(valid_actions)*100:.1f}%, Long={action_counts.get(1,0)/len(valid_actions)*100:.1f}%, Short={action_counts.get(2,0)/len(valid_actions)*100:.1f}%")
            
            # 변화 기반 신호 수 확인
            enter_long_count = df["enter_long"].sum()
            exit_long_count = df["exit_long"].sum()
            print(f"[DEBUG] Signal counts - Enter Long: {enter_long_count}, Exit Long: {exit_long_count}")
            
            # 액션 전환 패턴 분석
            action_changes = (df["policy_action"] != df["prev_action"]).sum()
            print(f"[DEBUG] Action changes: {action_changes} out of {len(valid_actions)} total actions")
            
            # 샘플 액션 시퀀스 확인
            sample_actions = df["policy_action"].dropna().tail(10).tolist()
            print(f"[DEBUG] Last 10 actions: {sample_actions}")
            
            # 연속 액션 패턴 확인
            consecutive_sell = (df["policy_action"] == 2).sum()
            consecutive_long = (df["policy_action"] == 1).sum()
            consecutive_hold = (df["policy_action"] == 0).sum()
            print(f"[DEBUG] Consecutive patterns - Sell: {consecutive_sell}, Long: {consecutive_long}, Hold: {consecutive_hold}")
        else:
            print(f"[DEBUG] No valid actions found! DataFrame length: {len(df)}")

        return df

    # ===========================
    # 내부: 임베딩/액션 추론
    # ===========================
    def _infer_action(self, window_df: pd.DataFrame) -> int:
        """
        주어진 윈도우에 대해 TFT 임베딩 -> PPO 행동 추론
        TFT 실패 시 fallback 피처 사용, PPO 실패 시 Hold
        """
        try:
            # TFT 인코더 사용
            if self.tft_encoder is not None:
                # DataFrame을 tensor로 변환
                feature_values = window_df[self.fp.features].values  # (T, F)
                feature_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
                
                with torch.no_grad():
                    emb = self.tft_encoder(feature_tensor).cpu().numpy().flatten()  # (64,)
            else:
                # Fallback 피처 사용
                emb = create_fallback_features(window_df)
                print(f"[DEBUG] Fallback features 사용, shape: {emb.shape}")
            
            # 형태 보정
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.shape[0] != self._EMB_DIM:
                print(f"[DEBUG] 임베딩 차원 불일치: {emb.shape[0]} != {self._EMB_DIM}")
                return 0
            
            # PPO 에이전트 사용
            if self.ppo_agent is not None:
                action = int(self.ppo_agent.predict(emb, deterministic=True))
                # 확률(가능하면) 조회
                try:
                    proba = self.ppo_agent.action_proba(emb)
                    conf = float(proba[action]) if proba.size else np.nan
                    print(f"[DEBUG] PPO Action: {action}, conf={conf:.3f} (0=Hold,1=Buy,2=Sell)")
                    # (옵션) 확신도 임계값
                    # if not np.isnan(conf) and conf < 0.55:
                    #     return 0  # 확신 낮으면 Hold
                except Exception:
                    print(f"[DEBUG] action_proba not available")
                return action
            else:
                print(f"[DEBUG] PPO Agent 없음, Hold 반환")
                return 0
                
        except Exception as e:
            print(f"[DEBUG] Action inference 실패: {e}")
            return 0

    # ===========================
    # 매수 시그널
    # ===========================
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """전이 기반 엔트리만 최종 buy로 사용"""
        df = dataframe.copy()
        if "enter_long" not in df.columns:
            df = self.populate_indicators(df, metadata)
        df["buy"] = df["enter_long"].astype(int)
        # 디버그: 최종 buy=enter_long 카운트 일치 확인
        print(f"[DEBUG] Buy signals (enter_long): {int(df['buy'].sum())}")
        return df

    # ===========================
    # 매도 시그널
    # ===========================
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """전이 기반 엑싯만 최종 sell로 사용"""
        df = dataframe.copy()
        if "exit_long" not in df.columns:
            df = self.populate_indicators(df, metadata)
        df["sell"] = df["exit_long"].astype(int)
        print(f"[DEBUG] Sell signals (exit_long): {int(df['sell'].sum())}")
        return df
