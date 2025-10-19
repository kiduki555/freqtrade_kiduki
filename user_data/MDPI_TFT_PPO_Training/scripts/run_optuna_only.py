# user_data/MDPI_TFT_PPO_Training/scripts/run_optuna_only.py
"""
TFT 학습 스킵하고 PPO만 실행하는 스크립트
- 기존에 학습된 mdpi_tft.pt + mdpi_scaler.pkl 로드
- Optuna 하이퍼파라미터 튜닝 + PPO 파인튜닝만 수행
"""
from __future__ import annotations

import json
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import warnings
from datetime import datetime

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
warnings.filterwarnings("ignore", message=".*Only one class is present in y_true.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# --- Anti-zero wrapper ---
import gymnasium as gym
import numpy as np

class ActionThresholdHoldPenalty(gym.Wrapper):
    def __init__(self, env, min_action_threshold=0.05, hold_penalty_bps=0.2):
        super().__init__(env)
        self.min_action_threshold = float(min_action_threshold)
        self.hold_penalty = float(hold_penalty_bps) / 1e4  # bps→ratio

    def step(self, action):
        # action 라운딩(너무 작으면 0으로)
        a = np.array(action, dtype=np.float32)
        if np.isscalar(a):
            a = np.array([a], dtype=np.float32)
        a = np.where(np.abs(a) < self.min_action_threshold, 0.0, a)
        obs, reward, terminated, truncated, info = self.env.step(a)

        # 포지션 거의 0일 때 아주 소액 페널티(기회비용)
        # env에 current_position/position 같은 속성이 있으면 그걸 사용
        pos = 0.0
        for key in ("current_position", "position", "pos"):
            if hasattr(self.env, key):
                pos = float(getattr(self.env, key))
                break
        if abs(pos) < 1e-6:
            reward = float(reward) - self.hold_penalty

        return obs, reward, terminated, truncated, info

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.trading_env import TradingEnv
from TFT_PPO_modules.checkpoint import ModelCheckpoint
from TFT_PPO_modules.performance_metrics import performance_metrics
from TFT_PPO_Training.scripts.utils import setup_device, set_seed, ensure_dir
from TFT_PPO_Training.scripts.optuna_tuning import tune_ppo
from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.utils import get_linear_fn
from torch import nn

from TFT_PPO_modules.multi_task_tft import MultiTaskTFT, create_multi_horizon_targets

# MDPI 정규화 모듈 import
from mdpi_normalization import MDPIStandardizer


def parse_args():
    """CLI 인자 파싱"""
    ap = argparse.ArgumentParser(description="TFT 학습 스킵하고 PPO만 실행")
    ap.add_argument("--config", default="user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft_ppo.yml",
                   help="설정 파일 경로")
    ap.add_argument("--tft-ckpt", default="user_data/models/best/mdpi_tft.pt",
                   help="사전학습된 TFT 체크포인트 경로")
    ap.add_argument("--scaler", default="user_data/models/best/mdpi_scaler.pkl",
                   help="MDPI 스케일러 경로")
    ap.add_argument("--optuna-config", default="user_data/TFT_PPO_Training/configs/optuna_config.yml",
                   help="Optuna 설정 파일 경로")
    ap.add_argument("--ppo-steps", type=int, default=None,
                   help="PPO 파인튜닝 total timesteps override")
    ap.add_argument("--data-path", default=None,
                   help="데이터 파일 경로 override")
    ap.add_argument("--finetune-days", type=int, default=None,
                   help="PPO 파인튜닝용 최근 데이터 일수 override")
    ap.add_argument("--seed", type=int, default=42,
                   help="랜덤 시드")
    return ap.parse_args()


def load_mdpi_config(config_path):
    """MDPI 설정 파일 로드 및 기본값 설정"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 기본값 설정
    defaults = {
        "seed": 42,
        "data": {
            "ohlcv_path": "user_data/data/binance/BTC_USDT-1h.feather",
            "asset": "BTC/USDT",
            "finetune_days": 180
        },
        "split": {
            "train_val_cut": "2024-06-01"
        },
        "tft": {
            "enc_len": 64,
            "hidden_size": 160,
            "attention_heads": 4,
            "dropout": 0.2,
            "horizons": [24, 48, 96],
            "train_split": 0.8,
            "batch_size": 128,
            "max_epochs": 50,
            "early_stopping": {
                "patience": 8,
                "min_delta": 1e-4
            },
            "learning_rate": 1e-3,
            "grad_clip": 1.0,
            "loss_weights": {
                "returns": 1.0,
                "direction": 0.5,
                "volatility": 0.25
            }
        },
        "ppo": {
            "timesteps": 100000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5
        }
    }
    
    # 중첩 딕셔너리 업데이트
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    update_nested_dict(defaults, config)
    return defaults


def run_ppo_only(
    config_path="user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft_ppo.yml",
    tft_ckpt_path="user_data/models/best/mdpi_tft.pt",
    scaler_path="user_data/models/best/mdpi_scaler.pkl",
    optuna_cfg_path="user_data/TFT_PPO_Training/configs/optuna_config.yml",
    ppo_steps_override=None,
    data_path_override=None,
    finetune_days_override=None,
    seed=42
):
    """TFT 학습 스킵하고 PPO만 실행"""
    
    print("=" * 60)
    print("PPO-ONLY 모드: TFT 학습 스킵하고 PPO 파인튜닝만 실행")
    print("=" * 60)
    
    # ============================
    # 1) Load Config & Setup
    # ============================
    config = load_mdpi_config(config_path)
    
    # CLI 인자로 override
    if data_path_override:
        config["data"]["ohlcv_path"] = data_path_override
    if finetune_days_override:
        config["data"]["finetune_days"] = finetune_days_override
    config["seed"] = seed
    
    device = setup_device(verbose=True)
    set_seed(config.get("seed", 42), deterministic=True)
    
    ensure_dir("user_data/models")
    ensure_dir("user_data/models/best")
    print(f"[PATH] models dir = {os.path.abspath('user_data/models')}")
    
    # ============================
    # 2) Data Preparation
    # ============================
    print("Loading OHLCV data...")
    data_path = config["data"]["ohlcv_path"]
    if data_path.endswith('.feather'):
        df = pd.read_feather(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 자산 컬럼 처리
    if "asset" not in df.columns:
        df["asset"] = config["data"].get("asset", "ASSET")
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} samples, columns: {df.columns.tolist()}")
    print(f"Asset column: {df['asset'].unique()}")
    
    # 피처 엔지니어링 (train_mdpi_pipeline.py와 동일한 방식)
    fp = FeaturePipeline()
    print(f"Before feature engineering: columns={df.columns.tolist()}")
    print(f"Asset column exists: {'asset' in df.columns}")
    
    # asset 컬럼을 임시로 저장
    asset_col = df["asset"].copy()
    
    df = (
        df.groupby("asset", group_keys=False, observed=True)
          .apply(lambda g: fp.add_features(g), include_groups=False)
          .reset_index(drop=True)
    )
    
    # asset 컬럼을 다시 추가
    df["asset"] = asset_col
    
    print(f"After feature engineering: columns={df.columns.tolist()}")
    print(f"Asset column exists: {'asset' in df.columns}")
    
    # 멀티-호라이즌 타깃 생성 (train_mdpi_pipeline.py와 동일한 방식)
    horizons = config["tft"]["horizons"]
    print(f"Before target creation: asset column exists: {'asset' in df.columns}")
    
    # asset 컬럼을 임시로 저장
    asset_col = df["asset"].copy()
    
    df = (
        df.groupby("asset", group_keys=False, observed=True)
          .apply(lambda g: create_multi_horizon_targets(g, horizons=horizons), include_groups=False)
          .reset_index(drop=True)
    )
    
    # asset 컬럼을 다시 추가
    df["asset"] = asset_col
    
    print(f"After target creation: columns={df.columns.tolist()}")
    print(f"Asset column exists: {'asset' in df.columns}")
    
    # 기존 log_return도 유지 (호환성)
    df["log_return"] = df["return_24h"].fillna(0.0)
    
    # 데이터 품질 검증
    print("\nData quality check:")
    for col in [f"return_{h}h" for h in horizons] + ["direction", "volatility", "log_return"]:
        if col in df.columns:
            na_count = df[col].isna().sum()
            inf_count = np.isinf(df[col]).sum()
            print(f"  {col}: {na_count} NA, {inf_count} infinite values")
            
            if na_count > 0 or inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                print(f"    -> Fixed: replaced with 0.0")
    
    # Train/Val split by date (PPO용 데이터만 추출)
    cutoff = pd.to_datetime(config["split"]["train_val_cut"])
    if df["date"].dt.tz is not None:
        cutoff = cutoff.tz_localize('UTC')
    
    df_ppo = df[df["date"] >= cutoff].copy()
    
    # 평가용 고정 구간 (trial 무관)
    eval_df = df_ppo.tail(4000).reset_index(drop=True)
    
    # 학습/튜닝용 (trial별 변동 가능)
    train_df = df_ppo.iloc[:-4000].reset_index(drop=True)
    
    # PPO용 최근 데이터 제한 (finetune_days)
    if config["data"]["finetune_days"] > 0:
        cut_date = train_df["date"].max() - pd.Timedelta(days=config["data"]["finetune_days"])
        train_df = train_df[train_df["date"] >= cut_date].reset_index(drop=True)
    
    print(f"PPO training data: {len(train_df)} samples")
    print(f"PPO evaluation data: {len(eval_df)} samples")
    
    # ============================
    # 3) MDPI Normalization (기존 스케일러 로드)
    # ============================
    print("Loading MDPI normalization scaler...")
    feature_cols = [c for c in df_ppo.columns if c not in [
        "date", "asset", "open", "high", "low", "close", 
        "return_24h", "return_48h", "return_96h", 
        "direction", "volatility", "log_return"
    ]]
    
    print(f"Normalizing {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    # 기존 스케일러 로드 후 transform만
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # PPO 범위(df_ppo)만 transform
    df_ppo[feature_cols] = scaler.transform(df_ppo, feature_cols)[feature_cols]
    
    # NaN 처리 (피처만)
    df_ppo[feature_cols] = df_ppo[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print("Normalization completed using pre-trained scaler.")
    
    # ============================
    # 4) TFT 모델 로드 (학습 스킵)
    # ============================
    print("[TFT] Skipping training. Loading pretrained encoder...")
    
    # 인덱싱은 PPO에도 필요
    df_ppo["time_idx"] = (df_ppo.groupby("asset")["date"].rank(method="first").astype(int) - 1)
    df_ppo["group_id"] = df_ppo["asset"]
    
    # 기존 모델의 설정을 로드하여 정확한 구조 재현
    try:
        with open("user_data/models/best/mdpi_config.json", "r") as f:
            original_config = json.load(f)
        print("[TFT] Loaded original model configuration")
    except Exception as e:
        print(f"[TFT] Warning: Could not load original config: {e}")
        original_config = config
    
    # 최소한의 더미 데이터셋 생성 (TFT 초기화용) - 정확히 21개 변수로 맞춤
    from pytorch_forecasting.data import TimeSeriesDataSet

    enc_len = int(original_config["tft"]["enc_len"])

    # 1) 더미 DF 준비 (값은 대충이어도 OK, "개수/이름"이 중요)
    dummy_df = df_ppo.head(100).copy()
    dummy_df = dummy_df.sort_values(["asset", "date"]).reset_index(drop=True)

    # 2) 훈련과 동일한 피처 목록 강제: 정확히 17개 피처만 사용
    # log_return은 제외해야 함 (원래 학습 시에는 17개였음)
    feature_cols = [c for c in dummy_df.columns if c not in [
        "date", "asset", "open", "high", "low", "close", 
        "return_24h", "return_48h", "return_96h", 
        "direction", "volatility", "log_return", "time_idx", "group_id"
    ]]
    
    print(f"[TFT] feature_cols(len)={len(feature_cols)}")
    print(f"[TFT] fp.features={fp.features}")
    print(f"[TFT] filtered feature_cols={feature_cols}")
    assert len(feature_cols) == 17, f"feature_cols must be 17, got {len(feature_cols)}: {feature_cols}"
    print(f"[TFT] Creating dummy dataset with {len(feature_cols)} features: {feature_cols[:5]}...")
    print(f"[TFT] unknown_reals={len(feature_cols)+1}, known_reals=1+2(auto) -> total encoder vars should be 21")

    # 원래 학습 시와 정확히 동일한 구성 (디코더 문제 해결)
    dummy_dataset = TimeSeriesDataSet(
        dummy_df,
        time_idx="time_idx",
        target="return_24h",
        group_ids=["group_id"],
        min_encoder_length=enc_len,
        max_encoder_length=enc_len,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        # 학습시와 동일하게: unknown = feature_cols + return_24h
        time_varying_unknown_reals=feature_cols + ["return_24h"],   # ← 17 + 1 = 18
        # 학습시와 동일하게: known = time_idx만 (디코더 호환성)
        time_varying_known_reals=["time_idx"],  # ← 1개만 (원래 학습 시와 동일)
        add_relative_time_idx=True,   # 인코더에만 자동 추가
        add_target_scales=False,
        add_encoder_length=True,      # 인코더에만 자동 추가
    )

    # 디버깅 출력: 최종 인코더 변수 개수 확인 (모델 로딩 전)
    try:
        reals = dummy_dataset.reals
        print(f"[TFT] encoder variables = {len(reals)} | names(head)={reals[:6]}")
        print(f"[TFT] all reals = {reals}")
        
        # unknown_reals와 known_reals 개수 확인
        unknown_count = len(dummy_dataset.time_varying_unknown_reals)
        known_count = len(dummy_dataset.time_varying_known_reals)
        print(f"[TFT] unknown_reals count = {unknown_count}")
        print(f"[TFT] known_reals count = {known_count}")
        print(f"[TFT] total reals = {unknown_count + known_count}")
    except Exception as e:
        print(f"[TFT] Debug error: {e}")

    # 4) 모델 초기화 및 로드 (이제 shape 일치)
    # PPO는 CPU에서 실행되므로 TFT도 CPU로 설정하여 디바이스 불일치 방지
    tft = MultiTaskTFT(
        dataset=dummy_dataset,
        hidden_size=original_config["tft"]["hidden_size"],
        attention_head_size=original_config["tft"]["attention_heads"],
        dropout=original_config["tft"]["dropout"],
        horizons=original_config["tft"]["horizons"],
        loss_weights=original_config["tft"]["loss_weights"],
        device="cpu"  # CPU로 강제 설정
    )

    # 호환되지 않는 레이어를 제외하고 로딩
    print(f"[TFT] Loading checkpoint with selective loading...")
    state = torch.load(tft_ckpt_path, map_location="cpu")  # CPU로 로드
    
    if isinstance(state, dict) and "state_dict" in state:
        checkpoint_state = state["state_dict"]
    else:
        checkpoint_state = state
    
    # 호환되는 레이어만 필터링
    model_state = tft.state_dict()
    compatible_state = {}
    
    for key, value in checkpoint_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                print(f"[TFT] Skipping incompatible layer: {key} (checkpoint: {value.shape}, model: {model_state[key].shape})")
        else:
            print(f"[TFT] Skipping missing layer: {key}")
    
    # 호환되는 레이어만 로딩
    missing_keys, unexpected_keys = tft.load_state_dict(compatible_state, strict=False)
    
    print(f"[TFT] Loaded {len(compatible_state)} compatible layers")
    print(f"[TFT] Missing keys: {len(missing_keys)}")
    print(f"[TFT] Unexpected keys: {len(unexpected_keys)}")
    
    tft = tft.eval()
    print(f"[TFT] Loaded: {tft_ckpt_path}")
    
    # ============================
    # 5) PPO Training
    # ============================
    print("PPO fine-tuning...")
    
    def make_env(offset=0, trial_seed=None):
        """환경 생성 함수"""
        if trial_seed is not None:
            np.random.seed(trial_seed)
            max_random_offset = min(500, len(df_ppo) // 10)
            random_offset = np.random.randint(0, max_random_offset)
            offset += random_offset
        
        if offset > 0:
            df_subset = eval_df.iloc[offset:].reset_index(drop=True)
        else:
            df_subset = eval_df
        
        env = TradingEnv(
            df_subset, 
            tft_model=tft.eval(), 
            features=fp.features,
            reward_mode="pnl_delta",  # 포지션 델타형 보상
            fee_bps=3,               # 수수료 완화 (0.03%)
            slippage_bps=1           # 슬리피지 완화 (0.01%)
        )
        
        # TFT 인코딩 활성화
        env.use_tft_encoding = True
        
        env = PriceTapWrapper(env)
        
        # ★ ActionThresholdHoldPenalty 래퍼 제거 (이산 액션에 부적합)
        # env = ActionThresholdHoldPenalty(env, min_action_threshold=0.05, hold_penalty_bps=0.2)
        
        # 동적 max_episode_steps 설정
        try:
            total_len = env.get_wrapper_attr("length")
        except Exception:
            total_len = getattr(getattr(env, "unwrapped", env), "length", None)
        
        min_steps = 1000
        max_steps = 5000
        if total_len is not None and isinstance(offset, int):
            remaining = max(0, int(total_len) - int(offset))
            steps = max(min_steps, min(max_steps, remaining - 2))
        else:
            steps = 2000
        if steps < 50:
            steps = 50
        
        if not isinstance(env, TimeLimit):
            env = TimeLimit(env, max_episode_steps=int(steps))
        
        return env
    
    # Optuna 튜닝
    print("Starting Optuna hyperparameter tuning...")
    best_params = tune_ppo(
        make_env, 
        config, 
        optuna_cfg_path=optuna_cfg_path,
        tft_model=tft,
        df_ppo=df_ppo,
        feature_pipeline=fp
    )
    
    # 홀드 붓박이 방지를 위한 파라미터 조정
    best_params["ent_coef"] = max(best_params.get("ent_coef", 0.10), 0.05)  # 엔트로피 계수 증가 (최소 0.05)
    best_params["clip_range"] = 0.2  # 클립 범위 조정
    
    print(f"[PPO] Adjusted params for final training:")
    print(f"[PPO] ent_coef={best_params['ent_coef']:.4f} | clip_range={best_params['clip_range']:.2f}")
    
    # 엔트로피/클립 스케줄 + target_kl (정책 붕괴 방지)
    ent_schedule = get_linear_fn(start=0.10, end=0.03, end_fraction=1.0)  # 엔트로피 증가: 0.10 → 0.03
    clip_schedule = get_linear_fn(start=best_params.get("clip_range", 0.25),
                                  end=max(0.15, 0.8 * best_params.get("clip_range", 0.25)),
                                  end_fraction=1.0)
    
    # 최종 학습
    venv_train = DummyVecEnv([make_env])
    venv_train = VecNormalize(venv_train, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0)
    # 저장용 경로
    vecnorm_path = "user_data/models/best/vecnorm.pkl"
    
    # 정책 분산 하한(log_std_init) 주기 (연속액션 σ 바닥 방지)
    policy_kwargs = dict(
        log_std_init=-0.5,   # 너무 작지 않게(≈σ~0.6 수준)
        ortho_init=False,
        activation_fn=nn.Tanh,
    )
    
    # PPO 초기화 시에 스케줄 적용 + target_kl 설정
    bp = {k:v for k,v in best_params.items() if k not in ["ent_coef", "clip_range"]}
    ppo = PPO("MlpPolicy", venv_train,
              policy_kwargs=policy_kwargs,
              ent_coef=ent_schedule,
              clip_range=clip_schedule,
              target_kl=0.03,
              **bp,
              seed=config.get("seed", 42), device="cpu", verbose=1)
    
    checkpoint = ModelCheckpoint(save_dir="user_data/models/best")
    
    # 학습 진행
    set_seed(config.get("seed", 42), deterministic=False)
    
    total_steps = int(config["ppo"]["timesteps"] if ppo_steps_override is None else ppo_steps_override)
    trained = 0
    log_every = max(10_000, total_steps // 20)
    
    print(f"Starting PPO training for {total_steps} timesteps...")
    
    while trained < total_steps:
        step = min(log_every, total_steps - trained)
        ppo.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False)
        trained += step
        try:
            ent = float(getattr(ppo.logger.name_to_value, "train/entropy_loss", np.nan))
        except Exception:
            ent = np.nan
        print(f"[Train] PPO trained_steps={trained}/{total_steps} | entropy={ent:.4f}")
    
    # 모델 저장
    ppo.save("user_data/models/best/mdpi_ppo_policy.zip")
    print("PPO policy saved to user_data/models/best/mdpi_ppo_policy.zip")
    
    # VecNormalize 저장
    venv_train.save(vecnorm_path)
    print(f"VecNormalize saved to {vecnorm_path}")
    
    # ============================
    # 6) Evaluation (Stochastic + Deterministic)
    # ============================
    print("Evaluating final policy (stochastic + deterministic)...")

    from collections import Counter

    # VecNormalize 로드하여 평가
    venv_eval = DummyVecEnv([make_env])
    venv_eval = VecNormalize.load(vecnorm_path, venv_eval)
    venv_eval.training = False
    venv_eval.norm_reward = False  # 리포트는 원단위로

    def rollout(vec_env, model, deterministic, max_steps=10000):
        obs = vec_env.reset()
        rr = []
        aa = []
        done = [False]
        while not done[0] and len(rr) < max_steps:
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _ = vec_env.step(a)
            rr.append(float(r[0]))
            aa.append(float(a[0]))
        return np.asarray(rr, dtype=np.float64), np.asarray(aa, dtype=np.float32)

    # 액션 분포 점검(정책이 0으로 굳었는지 즉시 확인)
    r_det, a_det = rollout(venv_eval, ppo, True)
    r_sto, a_sto = rollout(venv_eval, ppo, False)
    print(f"[EVAL] det_action_std={a_det.std():.6f}, sto_action_std={a_sto.std():.6f}, "
          f"det_steps={len(r_det)}, sto_steps={len(r_sto)}")

    # 1) 확률 샘플링 평가(정책이 균등에 가까울 때 argmax 고정 방지)
    print(f"[EVAL/STO] steps={len(r_sto)} | action_counts={Counter(a_sto)} "
          f"| r_mean={np.mean(r_sto):.6g} r_std={np.std(r_sto):.6g}")

    metrics_sto = performance_metrics(r_sto, freq="1h", is_log_return=True)
    print(f"[EVAL/STO] Sharpe={metrics_sto['sharpe']:.4f} | WinRate={metrics_sto['win_rate']:.2%} | MDD={metrics_sto['mdd']:.4f}")

    # 2) 결정적 평가(기존 방식)도 같이 확인
    print(f"[EVAL/DET] steps={len(r_det)} | action_counts={Counter(a_det)} "
          f"| r_mean={np.mean(r_det):.6g} r_std={np.std(r_det):.6g}")

    metrics_det = performance_metrics(r_det, freq="1h", is_log_return=True)
    print(f"[EVAL/DET] Sharpe={metrics_det['sharpe']:.4f} | WinRate={metrics_det['win_rate']:.2%} | MDD={metrics_det['mdd']:.4f}")

    # 체크포인트는 확률 평가 기준으로
    stop = checkpoint.update(ppo, metrics={"sharpe": metrics_sto["sharpe"]})
    
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"[STO] Sharpe={metrics_sto['sharpe']:.4f} | "
        f"Sortino={metrics_sto.get('sortino', np.nan):.4f} | "
        f"Calmar={metrics_sto.get('calmar', np.nan):.4f} | "
        f"MDD={metrics_sto['mdd']:.4f} | "
        f"WinRate={metrics_sto['win_rate']:.2%}"
    )
    print(
        f"[DET] Sharpe={metrics_det['sharpe']:.4f} | "
        f"Sortino={metrics_det.get('sortino', np.nan):.4f} | "
        f"Calmar={metrics_det.get('calmar', np.nan):.4f} | "
        f"MDD={metrics_det['mdd']:.4f} | "
        f"WinRate={metrics_det['win_rate']:.2%}"
    )
    print("=" * 60)
    print("PPO-only training complete!")
    print("Best model stored under user_data/models/best/")
    
    return {
        "tft_model": tft,
        "ppo_model": ppo,
        "scaler": scaler,
        "config": config,
        "metrics_sto": metrics_sto,
        "metrics_det": metrics_det
    }


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    print("Configuration:")
    print(f"  Config: {args.config}")
    print(f"  TFT Checkpoint: {args.tft_ckpt}")
    print(f"  Scaler: {args.scaler}")
    print(f"  Optuna Config: {args.optuna_config}")
    print(f"  PPO Steps: {args.ppo_steps}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Finetune Days: {args.finetune_days}")
    print(f"  Seed: {args.seed}")
    print()
    
    # 파일 존재 확인
    if not os.path.exists(args.tft_ckpt):
        print(f"ERROR: TFT checkpoint not found: {args.tft_ckpt}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler):
        print(f"ERROR: Scaler not found: {args.scaler}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        result = run_ppo_only(
            config_path=args.config,
            tft_ckpt_path=args.tft_ckpt,
            scaler_path=args.scaler,
            optuna_cfg_path=args.optuna_config,
            ppo_steps_override=args.ppo_steps,
            data_path_override=args.data_path,
            finetune_days_override=args.finetune_days,
            seed=args.seed
        )
        print("Success!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
