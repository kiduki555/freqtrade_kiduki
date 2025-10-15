# user_data/TFT_PPO_Training/scripts/train_pipeline.py
from __future__ import annotations

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

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


# ============================
# 1) Load Config & Setup
# ============================
CONFIG_PATH = "user_data/TFT_PPO_Training/configs/model_config.yml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = setup_device(verbose=True)  # text-only logs
set_seed(config.get("seed", 42), deterministic=True)   # ⚠️ TFT는 안정성 우선

ensure_dir("user_data/models", verbose=True)
ensure_dir("user_data/models/best", verbose=True)

# ============================
# 2) Data Preparation
# ============================
print("Loading data...")
df = pd.read_feather(config["data"]["path"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Build features
fp = FeaturePipeline()
df = fp.add_features(df)

# Use log-returns (numeric stable; avoid torch ops in pandas)
df["log_return"] = np.log(df["close"]).diff().fillna(0.0)

# Split for TFT pretrain (long history) vs PPO finetune (recent n days)
cut_date = df["date"].max() - pd.Timedelta(days=config["data"]["finetune_days"])
df_tft = df.copy()
df_ppo = df[df["date"] >= cut_date].reset_index(drop=True)

# ============================
# 3) TFT Pretrain (e.g., 5y)
# ============================
print("Training TFT encoder...")
# time_idx: integer index (e.g., hourly steps). Use uniform step index to avoid gaps.
df_tft = df_tft.copy()
df_tft["time_idx"] = np.arange(len(df_tft), dtype=np.int64)
df_tft["group_id"] = "ASSET"

# Train/Validation 분할
split_idx = int(len(df_tft) * config["tft"]["train_split"])
train_df = df_tft.iloc[:split_idx].copy()
val_df = df_tft.iloc[split_idx:].copy()

# Train dataset
train_ds = TimeSeriesDataSet(
    train_df, 
    time_idx="time_idx", 
    target="log_return",
    group_ids=["group_id"],
    time_varying_known_reals=[],
    time_varying_unknown_reals=fp.features,
    max_encoder_length=config["tft"]["encoder_length"],
    max_prediction_length=1,
    target_normalizer=None,
)

# Validation dataset
val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df, stop_randomization=True)

# Data loaders
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

# TFT 모델 초기화
tft = TemporalFusionTransformer.from_dataset(
    train_ds,
    hidden_size=config["tft"]["hidden_size"],
    attention_head_size=config["tft"]["attention_heads"],
    dropout=config["tft"]["dropout"],
    output_size=1,
).to(device)

# Optimizer와 Scheduler 설정
optimizer = torch.optim.Adam(tft.parameters(), lr=float(config["tft"]["learning_rate"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="min", 
    factor=config["tft"]["scheduler"]["factor"], 
    patience=config["tft"]["scheduler"]["patience"]
)

# Early stopping 설정
best_val = float("inf")
patience = config["tft"]["early_stopping"]["patience"]
wait = 0
max_epochs = config["tft"]["max_epochs"]

print(f"Training TFT with early stopping (patience={patience}, max_epochs={max_epochs})")

for epoch in range(1, max_epochs + 1):
    # Training
    tft.train()
    train_loss_sum = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        # Move batch data to device
        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        target = y[0].to(device) if isinstance(y, (tuple, list)) else y.to(device)
        
        out = tft(x)
        prediction = out.prediction if hasattr(out, 'prediction') else out
        loss = torch.mean(torch.abs(prediction - target))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(tft.parameters(), max_norm=config["tft"]["grad_clip"])
        optimizer.step()
        train_loss_sum += float(loss.item())

    # Validation
    tft.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            target = y[0].to(device) if isinstance(y, (tuple, list)) else y.to(device)
            
            out = tft(x)
            prediction = out.prediction if hasattr(out, 'prediction') else out
            vloss = torch.mean(torch.abs(prediction - target))
            val_loss_sum += float(vloss.item())

    train_loss = train_loss_sum / max(len(train_loader), 1)
    val_loss = val_loss_sum / max(len(val_loader), 1)
    print(f"[TFT] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    # Learning rate scheduler
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"[TFT] Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

    # Early stopping 체크
    if val_loss + float(config["tft"]["early_stopping"]["min_delta"]) < best_val:
        best_val = val_loss
        wait = 0
        torch.save(tft.state_dict(), "user_data/models/tft_encoder.pt")
        print(f"[TFT] Saved best encoder (val_loss={val_loss:.6f})")
    else:
        wait += 1
        if wait >= patience:
            print(f"[TFT] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

print("TFT encoder saved to user_data/models/tft_encoder.pt")

# ============================
# 4) PPO Training (e.g., 6m)
# ============================
print("PPO fine-tuning...")

def make_env(offset=0):
    """환경 생성 함수 - offset 지원"""
    # offset이 있으면 해당 지점부터 시작하는 데이터 사용
    if offset > 0:
        df_subset = df_ppo.iloc[offset:].reset_index(drop=True)
    else:
        df_subset = df_ppo
    
    # 포지션 인식 TradingEnv - PnL 기반 보상 사용
    env = TradingEnv(
        df_subset, 
        tft_model=tft.eval(), 
        features=fp.features,
        reward_mode="pnl",  # PnL 기반 보상 사용
        fee_bps=10,         # 0.10% 수수료
        slippage_bps=5      # 0.05% 슬리피지
    )
    
    # 1) 가격 info 주입
    env = PriceTapWrapper(env)
    
    # 2) 남은 길이에 맞춰 max_episode_steps를 동적으로 설정
    try:
        total_len = env.get_wrapper_attr("length")
    except Exception:
        total_len = getattr(getattr(env, "unwrapped", env), "length", None)
    
    # 최소 1000, 최대 5000 스텝 사이에서, 남은 길이를 넘지 않게
    min_steps = 1000
    max_steps = 5000
    if total_len is not None and isinstance(offset, int):
        remaining = max(0, int(total_len) - int(offset))
        steps = max(min_steps, min(max_steps, remaining - 2))
    else:
        steps = 2000
    if steps < 50:
        steps = 50
    
    # 이미 TimeLimit가 있으면 그대로, 없으면 감싸기
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, max_episode_steps=int(steps))
    
    return env

# Optuna 튜닝
best_params = tune_ppo(make_env, config, optuna_cfg_path="user_data/TFT_PPO_Training/configs/optuna_config.yml")

# 최종 학습(선택): 튜닝된 파라미터 + timesteps 크게
env = DummyVecEnv([make_env])
ppo = PPO("MlpPolicy", env, **best_params, seed=config.get("seed", 42), device="cpu", verbose=1)

checkpoint = ModelCheckpoint(save_dir="user_data/models/best")

# Optional: W&B logging (guarded import)
use_wandb = bool(config.get("logging", {}).get("use_wandb", False))
if use_wandb:
    import wandb
    wandb.init(
        project=config["logging"]["wandb_project"],
        entity=config["logging"]["wandb_entity"],
        config=config,
    )

# === PPO 학습 전: 탐색을 위해 결정성 해제 ===
from TFT_PPO_Training.scripts.utils import set_seed as _set_seed
_set_seed(config.get("seed", 42), deterministic=False)  # ✅ 학습은 비결정적으로

# 학습 진행 로그 추가
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

ppo.save("user_data/models/ppo_policy.zip")
print("PPO policy saved to user_data/models/ppo_policy.zip")

# ============================
# 5) Evaluation
# ============================
# Simple rollout on the same env (for quick sanity). For unbiased eval, use holdout env.
obs = env.reset()
rewards = []
done = [False]
while not done[0]:
    action, _ = ppo.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    # reward is vector; take first env
    rewards.append(float(reward[0]))

metrics = performance_metrics(np.asarray(rewards, dtype=np.float64), freq=config.get("eval", {}).get("freq", "daily"))
if use_wandb:
    import wandb
    wandb.log(metrics)
    wandb.finish()

# Update checkpoint with primary metric (default: sharpe)
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
