# user_data/MDPI_TFT_PPO_Training/scripts/train_mdpi_pipeline.py
"""
MDPI 스타일 TFT + PPO 통합 학습 파이프라인
- train_pipeline.py를 MDPI_TFT_PPO_Training 폴더 구조에 맞게 수정
- MDPI 정규화 + TFT 사전학습 + PPO 파인튜닝
"""
from __future__ import annotations

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import math
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.trading_env import TradingEnv
from TFT_PPO_modules.checkpoint import ModelCheckpoint
from TFT_PPO_modules.performance_metrics import performance_metrics
from TFT_PPO_Training.scripts.utils import setup_device, set_seed, ensure_dir
from TFT_PPO_Training.scripts.optuna_tuning import tune_ppo
from TFT_PPO_Training.scripts.wrappers import PriceTapWrapper
from gymnasium.wrappers import TimeLimit

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from torch.utils.data import DataLoader
from TFT_PPO_modules.multi_task_tft import MultiTaskTFT, create_multi_horizon_targets, compute_downstream_metrics
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# MDPI 정규화 모듈 import
from mdpi_normalization import MDPIStandardizer


def ensure_dir(p):
    """디렉토리 생성"""
    os.makedirs(p, exist_ok=True)
    return p


def load_mdpi_config(config_path="user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft.yml"):
    """MDPI 설정 파일 로드 및 기본값 설정"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 기본값 설정
    defaults = {
        "seed": 42,
        "data": {
            "ohlcv_path": "user_data/datasets/ohlcv.csv",
            "asset": "BTC/USDT",
            "finetune_days": 180  # PPO 파인튜닝용 최근 데이터 일수
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


def build_timeseries_dataset(df, config):
    """TimeSeriesDataSet 구성"""
    feature_cols = [c for c in df.columns if c not in [
        "date", "asset", "open", "high", "low", "close", 
        "return_24h", "return_48h", "return_96h", 
        "direction", "volatility", "time_idx", "group_id"
    ]]
    
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="return_24h",
        group_ids=["group_id"],
        min_encoder_length=config["tft"]["enc_len"],
        max_encoder_length=config["tft"]["enc_len"],
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols + ["return_24h"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )


def train_mdpi_tft_ppo(config_path="user_data/MDPI_TFT_PPO_Training/configs/mdpi_tft.yml"):
    """MDPI 스타일 TFT + PPO 통합 학습"""
    
    # ============================
    # 1) Load Config & Setup
    # ============================
    config = load_mdpi_config(config_path)
    
    device = setup_device(verbose=True)
    set_seed(config.get("seed", 42), deterministic=True)
    
    ensure_dir("user_data/models")
    ensure_dir("user_data/models/best")
    print("[PATH] models dir =", os.path.abspath("user_data/models"))
    
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
    
    # 피처 엔지니어링 (FutureWarning 방지)
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
    
    # 멀티-호라이즌 타깃 생성 (FutureWarning 방지)
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
    
    # === Direction 라벨 디버깅 ===
    print("\n=== Direction Label Analysis ===")
    direction_labels = df["direction"].dropna()
    print(f"Direction label distribution:")
    print(f"  Total samples: {len(direction_labels)}")
    print(f"  Positive rate: {direction_labels.mean():.3f}")
    print(f"  Unique values: {sorted(direction_labels.unique())}")
    print(f"  Value counts: {direction_labels.value_counts().to_dict()}")
    
    # 시차 확인: return_24h와 direction의 관계
    r24_sample = df["return_24h"].dropna().head(100)
    dir_sample = df["direction"].dropna().head(100)
    print(f"  Return_24h sample: mean={r24_sample.mean():.4f}, std={r24_sample.std():.4f}")
    print(f"  Direction sample: mean={dir_sample.mean():.3f}")
    
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
    
    # Train/Val split by date
    cutoff = pd.to_datetime(config["split"]["train_val_cut"])
    # 시간대 문제 해결: 데이터가 UTC이면 cutoff도 UTC로 변환
    if df["date"].dt.tz is not None:
        cutoff = cutoff.tz_localize('UTC')
    
    df_tft = df[df["date"] < cutoff].copy()
    df_ppo = df[df["date"] >= cutoff].copy()
    
    # PPO용 최근 데이터 제한 (finetune_days)
    if config["data"]["finetune_days"] > 0:
        cut_date = df_ppo["date"].max() - pd.Timedelta(days=config["data"]["finetune_days"])
        df_ppo = df_ppo[df_ppo["date"] >= cut_date].reset_index(drop=True)
    
    print(f"TFT training data: {len(df_tft)} samples")
    print(f"PPO training data: {len(df_ppo)} samples")
    
    # ============================
    # 3) MDPI Normalization (피처만, 라벨 제외)
    # ============================
    print("Applying MDPI normalization...")
    feature_cols = [c for c in df_tft.columns if c not in [
        "date", "asset", "open", "high", "low", "close", 
        "return_24h", "return_48h", "return_96h", 
        "direction", "volatility", "log_return"
    ]]
    
    print(f"Normalizing {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    # MDPI 정규화 (train에만 fit, 피처만 변환)
    scaler = MDPIStandardizer(group_cols=["asset"])
    df_tft[feature_cols] = scaler.fit_transform(df_tft, feature_cols)[feature_cols]
    df_ppo[feature_cols] = scaler.transform(df_ppo, feature_cols)[feature_cols]
    
    # NaN 처리 (피처만)
    for d in (df_tft, df_ppo):
        d[feature_cols] = d[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    print("Normalization completed. Labels (returns, direction, volatility) remain unchanged.")
    
    # ============================
    # 4) TFT Pretrain
    # ============================
    print("Training TFT encoder...")
    
    # TFT용 인덱싱
    for d in (df_tft, df_ppo):
        d["time_idx"] = (d.groupby("asset")["date"].rank(method="first").astype(int) - 1)
        d["group_id"] = d["asset"]
    
    # Train/Validation 분할
    split_idx = int(len(df_tft) * config["tft"]["train_split"])
    train_df = df_tft.iloc[:split_idx].copy()
    val_df = df_tft.iloc[split_idx:].copy()
    
    # 데이터셋 구성
    train_ds = build_timeseries_dataset(train_df, config)
    val_ds = build_timeseries_dataset(val_df, config)
    
    train_loader = train_ds.to_dataloader(
        train=True, 
        batch_size=config["tft"]["batch_size"], 
        num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, 
        batch_size=config["tft"]["batch_size"], 
        num_workers=0
    )
    
    # 타깃 구성 함수
    def _build_targets_from_x(x: dict, horizons, dataset):
        """x에서 직접 타깃 컬럼을 추출하여 멀티태스크 타깃 구성"""
        enc = x["encoder_cont"]
        feats = dataset.reals
        idx = {f: feats.index(f) for f in feats}
        
        last = enc[:, -1, :]
        targets = {}
        
        # 멀티-호라이즌 수익률 타깃
        for h in horizons:
            col = f"return_{h}h"
            if col in idx:
                targets[col] = last[:, idx[col]]
        
        # 방향성 타깃
        r24 = targets["return_24h"]
        targets["direction"] = (r24 > 0).float()
        
        # 변동성 타깃
        vol_col = "volatility"
        if vol_col in idx:
            vol_raw = last[:, idx[vol_col]]
            targets["volatility"] = torch.abs(vol_raw)
        else:
            targets["volatility"] = torch.abs(r24)
        
        return {k: v.to(r24.device) for k, v in targets.items()}
    
    # 멀티태스크 TFT 모델 초기화
    tft = MultiTaskTFT(
        dataset=train_ds,
        hidden_size=config["tft"]["hidden_size"],
        attention_head_size=config["tft"]["attention_heads"],
        dropout=config["tft"]["dropout"],
        horizons=horizons,
        loss_weights=config["tft"]["loss_weights"],
        device=device
    )
    
    # Optimizer 설정 - AdamW로 교체 (weight decay 포함)
    optimizer = torch.optim.AdamW(
        tft.parameters(), 
        lr=float(config["tft"]["learning_rate"]),
        weight_decay=1e-4  # 정규화 효과
    )
    
    # Early stopping 설정
    best_val = float("inf")
    best_auc = -float("inf")
    patience = config["tft"]["early_stopping"]["patience"]
    wait = 0
    max_epochs = config["tft"]["max_epochs"]
    
    print(f"Training TFT with early stopping (patience={patience}, max_epochs={max_epochs})")
    
    for epoch in range(1, max_epochs + 1):
        # Training
        tft.train()
        train_losses = {"total": 0.0, "returns": 0.0, "direction": 0.0, "volatility": 0.0}
        
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move batch data to device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            y = y.to(device) if isinstance(y, torch.Tensor) else y
            
            # 멀티태스크 타깃 준비
            targets = _build_targets_from_x(x, horizons, train_ds)
            
            # Forward pass
            predictions = tft(x)
            
            # 멀티태스크 손실 계산
            losses = tft.compute_loss(predictions, targets)
            
            # Backward pass
            losses["total"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(tft.parameters(), max_norm=config["tft"]["grad_clip"])
            optimizer.step()
            
            # 손실 누적
            for key, loss in losses.items():
                if key in train_losses:
                    train_losses[key] += float(loss.item())
                else:
                    train_losses[key] = float(loss.item())
        
        # Validation
        tft.eval()
        val_losses = {"total": 0.0, "returns": 0.0, "direction": 0.0, "volatility": 0.0}
        downstream_metrics = {"directional_auc": 0.0, "ic_top20": 0.0, "vol_calibration": 0.0}
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                y = y.to(device) if isinstance(y, torch.Tensor) else y
                
                # 멀티태스크 타깃 준비
                targets = _build_targets_from_x(x, horizons, val_ds)
                
                # Forward pass
                predictions = tft(x)
                
                # 손실 계산
                losses = tft.compute_loss(predictions, targets)
                for key, loss in losses.items():
                    if key in val_losses:
                        val_losses[key] += float(loss.item())
                    else:
                        val_losses[key] = float(loss.item())
                
                # 다운스트림 메트릭 계산
                metrics = compute_downstream_metrics(predictions, targets)
                for key, value in metrics.items():
                    downstream_metrics[key] += value
        
        # 평균 계산
        for key in train_losses:
            train_losses[key] /= max(len(train_loader), 1)
        for key in val_losses:
            val_losses[key] /= max(len(val_loader), 1)
        for key in downstream_metrics:
            downstream_metrics[key] /= max(len(val_loader), 1)
        
        # 로깅
        auc_val = downstream_metrics.get('directional_auc', 0.5)
        ic_val = downstream_metrics.get('ic_top20', 0.0)
        vol_val = downstream_metrics.get('vol_calibration', 0.0)
        
        print(f"[TFT] epoch={epoch} "
              f"train_loss={train_losses['total']:.6f} val_loss={val_losses['total']:.6f} "
              f"AUC={auc_val:.3f} IC@20={ic_val:.3f} VolCorr={vol_val:.3f}")
        
        # 저장/조기종료 로직
        auc = float(downstream_metrics.get('directional_auc', float('nan')))
        val = float(val_losses['total'])
        
        is_auc_finite = math.isfinite(auc)
        loss_min_delta = float(config["tft"]["early_stopping"]["min_delta"])
        auc_min_delta = 1e-4
        
        first_save = (best_val == float("inf"))
        save_by_auc = (is_auc_finite and (best_auc == -float("inf") or auc > best_auc + auc_min_delta))
        save_by_loss = (val + loss_min_delta < best_val)
        
        should_save = first_save or (save_by_auc or (not is_auc_finite and save_by_loss))
        
        if should_save:
            best_val = min(best_val, val)
            if is_auc_finite:
                best_auc = auc
            wait = 0
            save_path = os.path.abspath("user_data/models/tft_encoder.pt")
            torch.save(tft.state_dict(), save_path)
            print(f"[TFT] Saved best encoder (val_loss={val:.6f}, AUC={auc if is_auc_finite else float('nan'):.3f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"[TFT] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    # 백업 안전장치
    if not os.path.exists("user_data/models/tft_encoder.pt"):
        torch.save(tft.state_dict(), "user_data/models/tft_encoder.pt")
        print("[TFT] WARNING: No best checkpoint during training. Saved last state.")
    
    # MDPI 스타일 저장 (추가)
    ensure_dir("user_data/models/best")
    torch.save(tft.state_dict(), "user_data/models/best/mdpi_tft.pt")
    
    # 스케일러 저장
    import pickle, json
    with open("user_data/models/best/mdpi_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("user_data/models/best/mdpi_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("TFT encoder saved to user_data/models/tft_encoder.pt")
    print("MDPI assets saved to user_data/models/best/")
    
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
            df_subset = df_ppo.iloc[offset:].reset_index(drop=True)
        else:
            df_subset = df_ppo
        
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
    best_params = tune_ppo(
        make_env, 
        config, 
        optuna_cfg_path="user_data/TFT_PPO_Training/configs/optuna_config.yml",
        tft_model=tft,
        df_ppo=df_ppo,
        feature_pipeline=fp
    )
    
    # 최종 학습
    env = DummyVecEnv([make_env])
    ppo = PPO("MlpPolicy", env, **best_params, seed=config.get("seed", 42), device="cpu", verbose=1)
    
    checkpoint = ModelCheckpoint(save_dir="user_data/models/best")
    
    # 학습 진행
    set_seed(config.get("seed", 42), deterministic=False)
    
    total_steps = int(config["ppo"]["timesteps"])
    trained = 0
    log_every = max(10_000, total_steps // 20)
    
    while trained < total_steps:
        step = min(log_every, total_steps - trained)
        ppo.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False)
        trained += step
        try:
            ent = float(getattr(ppo.logger.name_to_value, "train/entropy_loss", np.nan))
        except Exception:
            ent = np.nan
        print(f"[Train] PPO trained_steps={trained}/{total_steps} | entropy={ent:.4f}")
    
    ppo.save("user_data/models/mdpi_ppo_policy.zip")
    print("PPO policy saved to user_data/models/ppo_policy.zip")
    
    # ============================
    # 6) Evaluation
    # ============================
    obs = env.reset()
    rewards = []
    done = [False]
    while not done[0]:
        action, _ = ppo.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward[0]))
    
    metrics = performance_metrics(
        np.asarray(rewards, dtype=np.float64),
        freq="1h",
        is_log_return=True
    )
    
    stop = checkpoint.update(ppo, metrics={"sharpe": metrics["sharpe"]})
    
    print(
        "Final Metrics | "
        f"Sharpe={metrics['sharpe']:.4f} | "
        f"Sortino={metrics.get('sortino', np.nan):.4f} | "
        f"Calmar={metrics.get('calmar', np.nan):.4f} | "
        f"MDD={metrics['mdd']:.4f} | "
        f"WinRate={metrics['win_rate']:.2%}"
    )
    print("Training complete. Best model stored under user_data/models/best/")
    
    return {
        "tft_model": tft,
        "ppo_model": ppo,
        "scaler": scaler,
        "config": config,
        "metrics": metrics
    }


if __name__ == "__main__":
    train_mdpi_tft_ppo()
