# user_data/TFT_PPO_Training/scripts/train_pipeline.py
from __future__ import annotations

import os
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from TFT_PPO_Modules.feature_pipeline import FeaturePipeline
from TFT_PPO_Modules.trading_env import TradingEnv
from TFT_PPO_Modules.checkpoint import ModelCheckpoint
from TFT_PPO_Modules.performance_metrics import performance_metrics
from TFT_PPO_Training.scripts.utils import setup_device, set_seed, ensure_dir
from TFT_PPO_Training.scripts.optuna_tuning import tune_ppo

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from torch.utils.data import DataLoader


# ============================
# 1) Load Config & Setup
# ============================
CONFIG_PATH = "user_data/TFT_PPO_Training/configs/model_config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

device = setup_device(verbose=True)  # text-only logs
set_seed(config.get("seed", 42), deterministic=True)

ensure_dir("user_data/models", verbose=True)
ensure_dir("user_data/models/best", verbose=True)

# ============================
# 2) Data Preparation
# ============================
print("Loading data...")
df = pd.read_csv(config["data"]["path"])
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

# Minimal TFT dataset: target is log_return; unknown reals = features
tft_dataset = TimeSeriesDataSet(
    df_tft,
    time_idx="time_idx",
    target="log_return",
    group_ids=["group_id"],
    time_varying_known_reals=[],
    time_varying_unknown_reals=fp.features,
    max_encoder_length=config["tft"]["encoder_length"],  # e.g., 72
    max_prediction_length=1,
    # Keep target as-is; many setups use GroupNormalizer/NaN handling explicitly
    target_normalizer=None,
)

train_loader: DataLoader = tft_dataset.to_dataloader(
    train=True,
    batch_size=config["tft"]["batch_size"],
    num_workers=0,
)

tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    hidden_size=config["tft"]["hidden_size"],
    attention_head_size=config["tft"]["attention_heads"],
    dropout=config["tft"]["dropout"],
    output_size=1,
)
tft.to(device)
tft.train()

optimizer = torch.optim.Adam(tft.parameters(), lr=float(config["tft"]["learning_rate"]))

for epoch in range(int(config["tft"]["epochs"])):
    total_loss = 0.0
    for batch in train_loader:
        # pytorch_forecasting returns a dict-like batch
        x, y = batch
        optimizer.zero_grad()
        # y can be tuple(target, weight) depending on dataset; take first item if tuple
        target = y[0] if isinstance(y, (tuple, list)) else y
        out = tft(x.to(device))
        # out shape [B, pred_len], target shape [B, pred_len]
        loss = torch.mean(torch.abs(out - target.to(device)))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    avg_loss = total_loss / max(len(train_loader), 1)
    print(f"[TFT] epoch={epoch+1} loss={avg_loss:.6f}")

torch.save(tft.state_dict(), "user_data/models/tft_encoder.pt")
print("TFT encoder saved to user_data/models/tft_encoder.pt")

# ============================
# 4) PPO Training (e.g., 6m)
# ============================
print("PPO fine-tuning...")

def make_env():
    # TradingEnv is gymnasium-compatible; SB3 DummyVecEnv expects a callable returning an Env
    return TradingEnv(df_ppo, tft_model=tft.eval(), features=fp.features)

# Optuna tuning (returns best_params). Our tune_ppo expects env_fn, not env instance
best_params = tune_ppo(make_env, config)

# Build vectorized env with best hyperparameters
env = DummyVecEnv([make_env])
ppo = PPO("MlpPolicy", env, **best_params, verbose=1, seed=config.get("seed", 42))

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

ppo.learn(total_timesteps=int(config["ppo"]["timesteps"]))
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
