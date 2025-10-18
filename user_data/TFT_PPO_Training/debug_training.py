#!/usr/bin/env python3
"""
학습 문제 디버깅 스크립트
- 데이터 품질 확인
- 환경 초기화 테스트
- 간단한 랜덤 정책으로 평가 테스트
"""

import os
import sys
import numpy as np
import pandas as pd

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_data_quality():
    """데이터 품질 확인"""
    print("=== 데이터 품질 확인 ===")
    
    data_path = "user_data/data/binance/BTC_USDT-1h.feather"
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        return False
    
    try:
        df = pd.read_feather(data_path)
        print(f"✅ 데이터 로딩 성공: {df.shape}")
        
        # 기본 정보
        print(f"  - 컬럼: {df.columns.tolist()}")
        print(f"  - 날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  - 총 길이: {len(df)} 스텝")
        
        # 품질 확인
        missing_count = df.isnull().sum().sum()
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"  - 누락값: {missing_count}")
        print(f"  - 무한대값: {inf_count}")
        
        if missing_count > 0 or inf_count > 0:
            print("⚠️  데이터 품질 문제 발견")
            return False
        
        # 최소 길이 확인
        if len(df) < 1000:
            print(f"⚠️  데이터가 너무 짧습니다: {len(df)} < 1000")
            return False
        
        print("✅ 데이터 품질 양호")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return False

def test_environment():
    """환경 초기화 테스트"""
    print("\n=== 환경 초기화 테스트 ===")
    
    try:
        from TFT_PPO_modules.feature_pipeline import FeaturePipeline
        from TFT_PPO_modules.trading_env import TradingEnv
        from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
        from gymnasium.wrappers import TimeLimit
        
        # 데이터 로딩
        df = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 피처 파이프라인
        fp = FeaturePipeline()
        df = fp.add_features(df)
        
        # 최근 180일 데이터 사용 (finetune_days)
        cut_date = df["date"].max() - pd.Timedelta(days=180)
        df_subset = df[df["date"] >= cut_date].reset_index(drop=True)
        
        print(f"  - 피처 수: {len(fp.features)}")
        print(f"  - 서브셋 길이: {len(df_subset)}")
        
        # 더미 TFT 모델 (평가용)
        class DummyTFT:
            def eval(self):
                return self
            def __call__(self, x):
                # 랜덤 예측 반환
                batch_size = list(x.values())[0].shape[0] if x else 1
                return {
                    "prediction": np.random.randn(batch_size, 1),
                    "attention": np.random.randn(batch_size, 1, 1)
                }
        
        dummy_tft = DummyTFT()
        
        # 환경 생성
        env = TradingEnv(
            df_subset, 
            tft_model=dummy_tft, 
            features=fp.features,
            reward_mode="pnl",
            fee_bps=10,
            slippage_bps=5
        )
        
        env = PriceTapWrapper(env)
        env = TimeLimit(env, max_episode_steps=1000)
        
        print("✅ 환경 초기화 성공")
        
        # 간단한 테스트
        obs = env.reset()
        print(f"  - 초기 관측치 형태: {type(obs)}")
        
        # 몇 스텝 실행
        for i in range(5):
            action = env.action_space.sample()  # 랜덤 액션
            obs, reward, terminated, truncated, info = env.step(action)
            
            if isinstance(info, dict) and "close" in info:
                print(f"  - 스텝 {i}: 액션={action}, 보상={reward:.4f}, 가격={info['close']:.2f}")
            else:
                print(f"  - 스텝 {i}: 액션={action}, 보상={reward:.4f}, 가격정보 없음")
            
            if terminated or truncated:
                print("  - 에피소드 종료")
                break
        
        env.close()
        print("✅ 환경 테스트 성공")
        return True
        
    except Exception as e:
        print(f"❌ 환경 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_policy():
    """랜덤 정책으로 평가 테스트"""
    print("\n=== 랜덤 정책 평가 테스트 ===")
    
    try:
        from TFT_PPO_modules.feature_pipeline import FeaturePipeline
        from TFT_PPO_modules.trading_env import TradingEnv
        from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
        from gymnasium.wrappers import TimeLimit
        
        # 데이터 준비
        df = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        fp = FeaturePipeline()
        df = fp.add_features(df)
        
        cut_date = df["date"].max() - pd.Timedelta(days=180)
        df_subset = df[df["date"] >= cut_date].reset_index(drop=True)
        
        # 더미 TFT
        class DummyTFT:
            def eval(self):
                return self
            def __call__(self, x):
                batch_size = list(x.values())[0].shape[0] if x else 1
                return {
                    "prediction": np.random.randn(batch_size, 1),
                    "attention": np.random.randn(batch_size, 1, 1)
                }
        
        dummy_tft = DummyTFT()
        
        # 환경 생성
        env = TradingEnv(df_subset, dummy_tft, fp.features, reward_mode="pnl", fee_bps=10, slippage_bps=5)
        env = PriceTapWrapper(env)
        env = TimeLimit(env, max_episode_steps=2000)
        
        # 랜덤 정책 실행
        obs = env.reset()
        actions = []
        rewards = []
        closes = []
        
        done = False
        step = 0
        while not done and step < 2000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            
            if isinstance(info, dict) and "close" in info:
                closes.append(info["close"])
            
            done = terminated or truncated
            step += 1
        
        env.close()
        
        # 결과 분석
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        action_counts = {0: np.sum(actions == 0), 1: np.sum(actions == 1), 2: np.sum(actions == 2)}
        
        print(f"  - 총 스텝: {step}")
        print(f"  - 액션 분포: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
        print(f"  - 평균 보상: {np.mean(rewards):.4f}")
        print(f"  - 총 보상: {np.sum(rewards):.4f}")
        print(f"  - 가격 데이터 수: {len(closes)}")
        
        # Buy 액션이 있는지 확인
        if action_counts[1] > 0:
            print("✅ Buy 액션 발견 - 트레이드 가능")
            return True
        else:
            print("⚠️  Buy 액션이 없음 - 트레이드 불가능")
            return False
        
    except Exception as e:
        print(f"❌ 랜덤 정책 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 디버깅 함수"""
    print("🔍 TFT-PPO 학습 문제 디버깅 시작\n")
    
    # 1. 데이터 품질 확인
    data_ok = test_data_quality()
    
    if not data_ok:
        print("\n❌ 데이터 문제로 인해 학습이 불가능합니다.")
        return
    
    # 2. 환경 초기화 테스트
    env_ok = test_environment()
    
    if not env_ok:
        print("\n❌ 환경 초기화 문제로 인해 학습이 불가능합니다.")
        return
    
    # 3. 랜덤 정책 평가 테스트
    policy_ok = test_random_policy()
    
    if not policy_ok:
        print("\n⚠️  랜덤 정책에서도 Buy 액션이 없습니다. 액션 공간 설정을 확인하세요.")
        return
    
    print("\n✅ 모든 테스트 통과 - 학습 환경이 정상입니다.")
    print("💡 문제는 PPO 모델 학습 과정에 있을 가능성이 높습니다.")

if __name__ == "__main__":
    main()
