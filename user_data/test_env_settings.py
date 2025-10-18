#!/usr/bin/env python3
"""
환경 설정 확인 스크립트
"""
import sys
sys.path.append('..')
sys.path.append('../TFT_PPO_modules')
sys.path.append('../TFT_PPO_Training/scripts')

from TFT_PPO_modules.trading_env import TradingEnv
from TFT_PPO_modules.feature_pipeline import FeaturePipeline
import pandas as pd
import numpy as np

def test_env_settings():
    print("=" * 60)
    print("환경 설정 확인 테스트")
    print("=" * 60)
    
    # 간단한 테스트 데이터
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
        def eval(self): return self

    # 환경 생성 (Optuna와 동일한 설정)
    env = TradingEnv(
        df, 
        tft_model=DummyTFT().eval(), 
        features=fp.features,
        reward_mode='pnl_delta',
        fee_bps=3,
        slippage_bps=1
    )
    env.use_tft_encoding = True

    print('Environment created with:')
    print(f'  reward_mode: {env.reward_mode}')
    print(f'  fee_rate: {env.fee_rate}')
    print(f'  sanity_mode: {getattr(env, "sanity_mode", "Not set")}')
    print(f'  ActionFilter: {env.action_filter is not None}')

    # 몇 스텝 실행하여 거래 플래그 확인
    obs, info = env.reset()
    trade_flags = []
    actions = []
    rewards = []
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        trade_flag = info.get('trade', 0) if isinstance(info, dict) else 0
        trade_flags.append(trade_flag)
        actions.append(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break

    print(f'\nTest Results:')
    print(f'  Steps: {len(trade_flags)}')
    print(f'  Trade flags: {trade_flags[:10]}... (first 10)')
    print(f'  Total trades: {sum(trade_flags)}')
    print(f'  Actions: {actions[:10]}... (first 10)')
    print(f'  Reward range: {min(rewards):.6f} ~ {max(rewards):.6f}')
    print(f'  Positive rewards: {sum(1 for r in rewards if r > 0)}')
    
    if sum(trade_flags) > 0:
        print("SUCCESS: Trades are being detected!")
    else:
        print("FAIL: No trades detected - ActionFilter may be too strong")
    
    print("=" * 60)

if __name__ == "__main__":
    test_env_settings()
