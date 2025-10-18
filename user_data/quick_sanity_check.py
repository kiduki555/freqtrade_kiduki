#!/usr/bin/env python3
"""
양수 보상 존재 여부 자가진단 스크립트
사용자 요청에 따라 환경의 보상 분포를 확인합니다.
"""
import sys
import os
import numpy as np
import inspect

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from TFT_PPO_modules.trading_env import TradingEnv
from TFT_PPO_modules.feature_pipeline import FeaturePipeline
import pandas as pd
import torch

def quick_sanity_check():
    """환경 보상 분포 확인"""
    print("=" * 60)
    print("양수 보상 존재 여부 자가진단")
    print("=" * 60)
    
    # TradingEnv 파일 경로 확인
    print("[DEBUG] TradingEnv.step defined at:", inspect.getsourcefile(TradingEnv))
    
    # 데이터 로드
    try:
        # 실제 데이터 구조에 맞게 수정
        df = pd.read_feather("data/binance/BTC_USDT-1h.feather")
        print(f"SUCCESS: Data loaded: {len(df)} samples")
        print(f"   - Columns: {df.columns.tolist()}")
        print(f"   - Date range: {df['date'].min()} ~ {df['date'].max()}")
    except Exception as e:
        print(f"ERROR: Data load failed: {e}")
        print("   Available files:")
        try:
            import os
            data_dir = "data/binance"
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                for f in files:
                    if f.endswith('.feather'):
                        print(f"   - {f}")
        except:
            pass
        return
    
    # 피처 엔지니어링
    fp = FeaturePipeline()
    df = fp.add_features(df)
    
    # 더미 TFT 모델 (평가용)
    class DummyTFT:
        def __init__(self):
            self.hidden_size = 64
            self.device = "cpu"
        
        def __call__(self, x):
            return {"encoder_repr": torch.randn(1, 64)}
    
    tft = DummyTFT()
    
    # 환경 생성 (sanity 모드 활성화)
    env = TradingEnv(
        df.tail(2000),  # 최근 2000개 샘플만 사용
        tft_model=tft,
        features=fp.features,
        reward_mode="pnl_delta",
        fee_bps=3,
        slippage_bps=1,
        sanity_mode=True  # 생성자에서 직접 설정
    )
    
    print(f"SUCCESS: Environment created (sanity_mode=True)")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    
    # 랜덤 정책으로 보상 분포 확인
    print("\nRandom policy reward distribution check...")
    
    obs, info = env.reset(seed=42)
    rewards = []
    actions = []
    trade_count = 0
    
    for t in range(1000):
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        actions.append(action)
        
        # 거래 횟수 집계
        if isinstance(info, dict):
            trade_count += int(info.get("trade", 0))
        
        if terminated or truncated:
            print(f"   Episode ended: step {t}")
            break
    
    # 통계 출력
    rewards = np.array(rewards)
    actions = np.array(actions)
    
    print(f"\nReward Statistics:")
    print(f"   - Count: {len(rewards)}")
    print(f"   - Min: {rewards.min():.6f}")
    print(f"   - Max: {rewards.max():.6f}")
    print(f"   - Mean: {rewards.mean():.6f}")
    print(f"   - Std: {rewards.std():.6f}")
    
    print(f"\nAction Distribution:")
    unique, counts = np.unique(actions, return_counts=True)
    for action, count in zip(unique, counts):
        action_name = {0: "Hold", 1: "Buy", 2: "Sell"}[action]
        print(f"   - {action_name}: {count} times ({count/len(actions)*100:.1f}%)")
    
    print(f"\nTrade Statistics:")
    print(f"   - Total trades: {trade_count}")
    print(f"   - Trade ratio: {trade_count/len(actions)*100:.1f}%")
    
    # 양수 보상 확인
    positive_rewards = np.sum(rewards > 0)
    print(f"\nPositive Reward Check:")
    print(f"   - Positive rewards: {positive_rewards}")
    print(f"   - Positive ratio: {positive_rewards/len(rewards)*100:.1f}%")
    
    if positive_rewards > 0:
        print(f"   SUCCESS: Positive rewards exist! Max: {rewards.max():.6f}")
        print(f"   -> Environment reward logic is working")
    else:
        print(f"   FAIL: No positive rewards!")
        print(f"   -> Environment reward logic needs fixing")
    
    # 진단 결과
    print(f"\nDiagnosis Result:")
    if rewards.max() > 0:
        print(f"   PASS: Positive rewards exist")
        print(f"   -> Can solve overtrading by lowering ent_coef")
    else:
        print(f"   FAIL: No positive rewards at all")
        print(f"   -> Must fix environment reward logic first")
    
    print("=" * 60)

if __name__ == "__main__":
    quick_sanity_check()
