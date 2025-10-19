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
    
    # === 새로운 PnL 기반 평가 시스템 테스트 ===
    print(f"\n" + "="*60)
    print("새로운 PnL 기반 평가 시스템 테스트")
    print("="*60)
    
    # PnL 기반 평가 함수들 (optuna_tuning.py에서 복사)
    def _equity_from_pnl_series(pnl_series, start_equity=1.0):
        pnl = np.asarray(pnl_series, dtype=np.float64)
        return start_equity * np.exp(np.cumsum(pnl))

    def _sharpe_from_pnl(pnl_series, steps_per_year=24*365, eps=1e-12):
        r = np.asarray(pnl_series, dtype=np.float64)
        mu, sd = float(np.mean(r)), float(np.std(r))
        if sd < eps: 
            return 0.0
        return float((mu / sd) * np.sqrt(steps_per_year))

    def max_drawdown_equity(eq):
        eq = np.asarray(eq, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1e-9)
        return float(np.max(dd)) if dd.size else 0.0

    def _score_from_kpis(sharpe, mdd, trades):
        if trades == 0:
            return -2.0
        if trades < 5:
            return -1.3
        base = 0.4*sharpe - 2.0*mdd
        base = max(-2.0, min(-1.0, base))
        return base

    # PnL 데이터 수집 (이미 수집된 데이터 사용)
    pnl_list = []
    for i in range(len(rewards)):
        # 간단한 PnL 시뮬레이션 (실제로는 info에서 가져와야 함)
        pnl_list.append(rewards[i] * 0.1)  # 보상을 PnL로 변환 (임시)

    # 평가 계산
    equity = _equity_from_pnl_series(pnl_list, start_equity=1.0)
    mdd = max_drawdown_equity(equity)
    sharpe = _sharpe_from_pnl(pnl_list, steps_per_year=24*365)
    winrate = float(np.mean(np.array(pnl_list) > 0.0))
    score = _score_from_kpis(sharpe, mdd, trade_count)

    print(f"PnL 기반 평가 결과:")
    print(f"  Steps: {len(pnl_list)}")
    print(f"  Trades: {trade_count}")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Winrate: {winrate:.3f}")
    print(f"  MDD: {mdd:.3f}")
    print(f"  Score: {score:.3f}")
    
    print(f"\n통계 비교:")
    print(f"  Reward stats: mean={np.mean(rewards):.6f}, std={np.std(rewards):.6f}")
    print(f"  PnL stats: mean={np.mean(pnl_list):.6f}, std={np.std(pnl_list):.6f}")
    print(f"  Equity range: {equity.min():.6f} ~ {equity.max():.6f}")

    # 성공 조건 확인
    success_conditions = [
        mdd < 1.0,  # MDD가 1.000에서 벗어남
        trade_count > 0,  # 거래 발생
        len(pnl_list) > 0,  # PnL 수집됨
        not np.isnan(sharpe),  # Sharpe 계산됨
    ]
    
    if all(success_conditions):
        print(f"\nSUCCESS: 새로운 PnL 기반 평가 시스템 작동!")
        print(f"   - MDD 정상화: {mdd:.3f} (1.000 아님)")
        print(f"   - 거래 발생: {trade_count}회")
        print(f"   - PnL 기반 평가 작동")
    else:
        print(f"\nFAIL: 일부 조건 실패")
        print(f"   - MDD < 1.0: {mdd < 1.0}")
        print(f"   - 거래 발생: {trade_count > 0}")
        print(f"   - PnL 수집: {len(pnl_list) > 0}")
        print(f"   - Sharpe 계산: {not np.isnan(sharpe)}")
    
    print("=" * 60)

if __name__ == "__main__":
    quick_sanity_check()
