#!/usr/bin/env python3
"""
간단한 학습 테스트 스크립트
- 랜덤 정책으로 환경 테스트
- PPO 모델의 기본 학습 테스트
"""

import os
import sys
import numpy as np

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_random_policy():
    """랜덤 정책으로 환경 테스트"""
    print("=== 랜덤 정책 테스트 ===")
    
    try:
        from TFT_PPO_modules.feature_pipeline import FeaturePipeline
        from TFT_PPO_modules.trading_env import TradingEnv
        from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
        from gymnasium.wrappers import TimeLimit
        import pandas as pd
        
        # 데이터 로딩
        df = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # 피처 파이프라인
        fp = FeaturePipeline()
        df = fp.add_features(df)
        
        # 최근 180일 데이터 사용
        cut_date = df["date"].max() - pd.Timedelta(days=180)
        df_subset = df[df["date"] >= cut_date].reset_index(drop=True)
        
        print(f"데이터 길이: {len(df_subset)}")
        
        # 더미 TFT 모델
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
        env = TimeLimit(env, max_episode_steps=1000)
        
        # 랜덤 정책 실행
        obs, info = env.reset()
        actions = []
        rewards = []
        
        for step in range(100):  # 100 스텝만 테스트
            action = env.action_space.sample()  # 랜덤 액션
            obs, reward, terminated, truncated, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # 결과 분석
        actions = np.array(actions)
        action_counts = {0: np.sum(actions == 0), 1: np.sum(actions == 1), 2: np.sum(actions == 2)}
        
        print(f"총 스텝: {len(actions)}")
        print(f"액션 분포: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
        print(f"평균 보상: {np.mean(rewards):.4f}")
        
        if action_counts[1] > 0:
            print("✅ Buy 액션 발견 - 환경이 정상 작동")
            return True
        else:
            print("❌ Buy 액션이 없음 - 환경 문제")
            return False
            
    except Exception as e:
        print(f"❌ 랜덤 정책 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_basic():
    """PPO 기본 학습 테스트"""
    print("\n=== PPO 기본 학습 테스트 ===")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from TFT_PPO_modules.feature_pipeline import FeaturePipeline
        from TFT_PPO_modules.trading_env import TradingEnv
        from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
        from gymnasium.wrappers import TimeLimit
        import pandas as pd
        
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
        
        # 환경 생성 함수
        def make_env():
            env = TradingEnv(df_subset, dummy_tft, fp.features, reward_mode="pnl", fee_bps=10, slippage_bps=5)
            env = PriceTapWrapper(env)
            env = TimeLimit(env, max_episode_steps=1000)
            return env
        
        # PPO 모델 생성 (높은 엔트로피로 탐색 강화)
        env = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=0.001,  # 높은 학습률
            ent_coef=0.1,         # 높은 엔트로피
            verbose=1,
            device="cpu"
        )
        
        print("PPO 모델 생성 완료")
        
        # 짧은 학습
        print("학습 시작...")
        model.learn(total_timesteps=10000, progress_bar=True)
        
        # 학습된 모델 테스트
        print("학습된 모델 테스트...")
        obs = env.reset()
        actions = []
        
        for step in range(50):  # 50 스텝 테스트
            action, _ = model.predict(obs, deterministic=False)  # 비결정적 예측
            obs, reward, done, info = env.step(action)
            
            actions.append(action[0])  # 벡터화된 환경이므로 [0] 인덱싱
            
            if done[0]:
                break
        
        env.close()
        
        # 결과 분석
        actions = np.array(actions)
        action_counts = {0: np.sum(actions == 0), 1: np.sum(actions == 1), 2: np.sum(actions == 2)}
        
        print(f"학습된 모델 액션 분포: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
        
        if action_counts[1] > 0:
            print("✅ 학습된 모델이 Buy 액션을 선택함")
            return True
        else:
            print("❌ 학습된 모델이 Buy 액션을 선택하지 않음")
            return False
            
    except Exception as e:
        print(f"❌ PPO 학습 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🔍 학습 문제 진단 테스트 시작\n")
    
    # 1. 랜덤 정책 테스트
    random_ok = test_random_policy()
    
    if not random_ok:
        print("\n❌ 환경 자체에 문제가 있습니다.")
        return
    
    # 2. PPO 기본 학습 테스트
    ppo_ok = test_ppo_basic()
    
    if not ppo_ok:
        print("\n❌ PPO 모델 학습에 문제가 있습니다.")
        print("💡 해결 방안:")
        print("  1. 학습률을 더 높이기")
        print("  2. 엔트로피 계수를 더 높이기")
        print("  3. 액션 필터를 비활성화하기")
        return
    
    print("\n✅ 모든 테스트 통과 - 학습 환경이 정상입니다.")

if __name__ == "__main__":
    main()
