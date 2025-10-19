#!/usr/bin/env python3
"""
새로운 PnL 기반 평가 시스템 테스트
"""
import pandas as pd
import numpy as np
import torch
import os
import sys

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MDPI_TFT_PPO_Training', 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TFT_PPO_Training', 'scripts'))

from feature_pipeline import FeaturePipeline
from trading_env import TradingEnv
from multi_task_tft import MultiTaskTFT, create_multi_horizon_targets
from mdpi_normalization import MDPIStandardizer

def test_pnl_evaluation():
    """새로운 PnL 기반 평가 시스템 테스트"""
    print("=" * 60)
    print("PnL 기반 평가 시스템 테스트")
    print("=" * 60)
    
    # 데이터 로드
    try:
        df = pd.read_feather("data/binance/BTC_USDT-1h.feather")
        print(f"SUCCESS: Data loaded: {len(df)} samples")
    except Exception as e:
        print(f"ERROR: Data load failed: {e}")
        return
    
    # 피처 엔지니어링
    fp = FeaturePipeline()
    df = fp.add_features(df)
    
    # 멀티-호라이즌 타깃 생성
    horizons = [24, 48, 96]
    df = (
        df.groupby("asset", group_keys=False, observed=True)
          .apply(lambda g: create_multi_horizon_targets(g, horizons=horizons), include_groups=False)
          .reset_index(drop=True)
    )
    df["log_return"] = df["return_24h"].fillna(0.0)

    # MDPI 정규화 (더미 스케일러)
    class DummyScaler:
        def transform(self, df, cols):
            return df
    scaler = DummyScaler()
    
    # TFT 모델 로드 (더미)
    class DummyTFT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 64
        def eval(self): return self
        def forward(self, x):
            return {"encoder_repr": torch.randn(1, self.hidden_size)}
    tft_model = DummyTFT()

    # 환경 생성 (평가용 - 래퍼 없음)
    env = TradingEnv(
        df.tail(1000),
        tft_model=tft_model,
        features=fp.features,
        reward_mode="pnl_delta",
        fee_bps=3,
        slippage_bps=1,
        sanity_mode=False
    )
    env.use_tft_encoding = True

    print(f"Environment created:")
    print(f"  - reward_mode: {env.reward_mode}")
    print(f"  - fee_rate: {env.fee_rate}")
    print(f"  - ActionFilter: {env.action_filter is not None}")

    # 랜덤 정책으로 PnL 수집
    obs, info = env.reset(seed=0)
    pnl_list = []
    rewards = []
    trade_count = 0
    
    for t in range(100):
        a = env.action_space.sample()  # 랜덤 액션
        obs, r, term, trunc, info = env.step(a)
        
        pnl_list.append(info.get("pnl_step", 0.0))
        rewards.append(r)
        trade_count += info.get("trade", 0)
        
        if term or trunc:
            break

    if not pnl_list:
        print("❌ PnL이 수집되지 않았습니다.")
        return

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

    # 평가 계산
    equity = _equity_from_pnl_series(pnl_list, start_equity=1.0)
    mdd = max_drawdown_equity(equity)
    sharpe = _sharpe_from_pnl(pnl_list, steps_per_year=24*365)
    winrate = float(np.mean(np.array(pnl_list) > 0.0))
    score = _score_from_kpis(sharpe, mdd, trade_count)

    print(f"\n--- PnL 기반 평가 결과 ---")
    print(f"  Steps: {len(pnl_list)}")
    print(f"  Trades: {trade_count}")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Winrate: {winrate:.3f}")
    print(f"  MDD: {mdd:.3f}")
    print(f"  Score: {score:.3f}")
    
    print(f"\n--- 통계 비교 ---")
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
        print(f"\n✅ SUCCESS: 모든 조건 통과!")
        print(f"   - MDD 정상화: {mdd:.3f} (1.000 아님)")
        print(f"   - 거래 발생: {trade_count}회")
        print(f"   - PnL 기반 평가 작동")
    else:
        print(f"\n❌ FAIL: 일부 조건 실패")
        print(f"   - MDD < 1.0: {mdd < 1.0}")
        print(f"   - 거래 발생: {trade_count > 0}")
        print(f"   - PnL 수집: {len(pnl_list) > 0}")
        print(f"   - Sharpe 계산: {not np.isnan(sharpe)}")

    print("=" * 60)

if __name__ == "__main__":
    test_pnl_evaluation()
