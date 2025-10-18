#!/usr/bin/env python3
"""
평가 루프 검증 스크립트
"""
import sys
import os
sys.path.append('..')
sys.path.append('../TFT_PPO_modules')
sys.path.append('../TFT_PPO_Training/scripts')

from TFT_PPO_modules.trading_env import TradingEnv
from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_Training.scripts.optuna_tuning import _score_rewards
import pandas as pd
import numpy as np

def test_evaluation():
    print("=" * 60)
    print("평가 루프 검증 테스트")
    print("=" * 60)
    
    # 간단한 테스트 데이터 (ATR 계산을 위해 충분한 길이)
    n = 100
    prices = np.linspace(100, 110, n)
    df = pd.DataFrame({
        'close': prices,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'volume': [1.0] * n
    })

    fp = FeaturePipeline()
    df = fp.add_features(df)

    class DummyTFT:
        def __call__(self, x): 
            return {'encoder_repr': np.random.randn(1, 64)}

    # 정상 모드로 환경 생성
    env = TradingEnv(df, tft_model=DummyTFT(), features=fp.features, sanity_mode=False)
    obs, info = env.reset()

    print(f"Environment created (sanity_mode=False)")
    print(f"Action space: {env.action_space}")
    print(f"ActionFilter: {env.action_filter is not None}")

    # 랜덤 액션으로 몇 스텝 실행
    rewards = []
    trade_count = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if isinstance(info, dict):
            trade_count += int(info.get('trade', 0))
        if terminated or truncated:
            break

    print(f"\nTest Results:")
    print(f"  - Steps: {len(rewards)}")
    print(f"  - Trades: {trade_count}")
    print(f"  - Reward range: {min(rewards):.6f} ~ {max(rewards):.6f}")
    print(f"  - Positive rewards: {sum(1 for r in rewards if r > 0)}")

    # 점수 계산
    score = _score_rewards(rewards, freq='1h', trades_override=trade_count)
    print(f"\nEvaluation Score: {score:.3f}")
    
    if score != -1.0:
        print("SUCCESS: Score is not -1.0 (no overtrading penalty)")
    else:
        print("FAIL: Score is -1.0 (overtrading penalty applied)")
    
    print("=" * 60)

if __name__ == "__main__":
    test_evaluation()
