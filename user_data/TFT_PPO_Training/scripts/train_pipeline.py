# user_data/TFT_PPO_Training/scripts/train_pipeline.py
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
print("[PATH] models dir =", os.path.abspath("user_data/models"))

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

# 멀티-호라이즌 타깃 생성
horizons = config["tft"].get("horizons", [24, 48, 96])
df = create_multi_horizon_targets(df, horizons=horizons, price_col="close")

# 기존 log_return도 유지 (호환성)
df["log_return"] = df["return_24h"].fillna(0.0)  # 24h를 기본으로 사용

# 데이터 품질 검증
print("Data quality check:")
for col in [f"return_{h}h" for h in horizons] + ["direction", "volatility", "log_return"]:
    if col in df.columns:
        na_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        print(f"  {col}: {na_count} NA, {inf_count} infinite values")
        
        # 최종 안전장치: 모든 NA/무한대 값을 0으로 대체
        if na_count > 0 or inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            print(f"    -> Fixed: replaced with 0.0")

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

# Train dataset - 멀티태스크용으로 수정 (타깃 컬럼 포함)
extra_targets = [f"return_{h}h" for h in horizons] + ["direction", "volatility"]
all_unknowns = list(dict.fromkeys(fp.features + extra_targets))  # 중복 제거

train_ds = TimeSeriesDataSet(
    train_df, 
    time_idx="time_idx", 
    target="return_24h",  # 기본 타깃 (24h)
    group_ids=["group_id"],
    time_varying_known_reals=[],
    time_varying_unknown_reals=all_unknowns,  # 타깃 컬럼도 함께 포함
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

# 타깃을 x에서 직접 구성하는 함수 (올바른 방법)
def _build_targets_from_x(x: dict, horizons, dataset):
    """x에서 직접 타깃 컬럼을 추출하여 멀티태스크 타깃 구성"""
    # x["encoder_cont"] shape: (B, T_enc, F)
    enc = x["encoder_cont"]  # (B, T, F)
    feats = dataset.reals  # TimeSeriesDataSet이 보유한 실수 피처 이름 순서
    idx = {f: feats.index(f) for f in feats}

    last = enc[:, -1, :]  # (B, F) 마지막 타임스텝
    targets = {}
    
    # 멀티-호라이즌 수익률 타깃 (미리 shift된 미래 수익)
    for h in horizons:
        col = f"return_{h}h"
        if col in idx:
            targets[col] = last[:, idx[col]]
    
    # 방향성 타깃 = 24h 미래수익의 부호 → {0,1}
    r24 = targets["return_24h"]
    targets["direction"] = (r24 > 0).float()
    
    # 변동성 타깃 = 양수 보장
    vol_col = "volatility"
    if vol_col in idx:
        vol_raw = last[:, idx[vol_col]]
        targets["volatility"] = torch.abs(vol_raw)  # 안전하게 양수화
    else:
        targets["volatility"] = torch.abs(r24)  # fallback
    
    return {k: v.to(r24.device) for k, v in targets.items()}

# 멀티태스크 TFT 모델 초기화
tft = MultiTaskTFT(
    dataset=train_ds,
    hidden_size=config["tft"]["hidden_size"],
    attention_head_size=config["tft"]["attention_heads"],
    dropout=config["tft"]["dropout"],
    horizons=horizons,
    loss_weights=config["tft"].get("loss_weights", {
        "returns": 1.0,
        "direction": 0.5,
        "volatility": 0.25
    }),
    device=device
)

# Optimizer 설정
optimizer = torch.optim.Adam(tft.parameters(), lr=float(config["tft"]["learning_rate"]))

# Learning rate scheduler 설정
lr_schedule = config["tft"].get("lr_schedule", "cosine")
if lr_schedule == "cosine":
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config["tft"]["max_epochs"],
        eta_min=float(config["tft"].get("min_lr", 1e-5))
    )
elif lr_schedule == "onecycle":
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config["tft"]["learning_rate"]),
        epochs=config["tft"]["max_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
else:
    # 기본 ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=config["tft"]["scheduler"]["factor"], 
        patience=config["tft"]["scheduler"]["patience"]
    )

# Early stopping 설정
best_val = float("inf")
best_r2 = -float("inf")  # R² 메트릭 초기화
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
        
        # 멀티태스크 타깃 준비 - x에서 직접 추출 (올바른 방법)
        targets = _build_targets_from_x(x, horizons, train_ds)
        
        # 디버그: 타깃 스케일 확인 (첫 배치만)
        if epoch == 1 and batch_idx == 0:
            print(f"[DEBUG] === Targets ===")
            for k, v in targets.items():
                a = v.detach().cpu().numpy()
                print(f"[DEBUG] target[{k}] mean={a.mean():.4g} std={a.std():.4g} min={a.min():.4g} max={a.max():.4g}")
            
            # 타깃 유효성 검증
            direction_vals = targets["direction"].detach().cpu().numpy()
            unique_dirs = np.unique(direction_vals)
            print(f"[DEBUG] direction unique values: {unique_dirs}")
            
            vol_vals = targets["volatility"].detach().cpu().numpy()
            print(f"[DEBUG] volatility has negative values: {(vol_vals < 0).sum()}")
        
        # 타깃이 이미 텐서이므로 디바이스만 확인
        for key, value in targets.items():
            if not isinstance(value, torch.Tensor):
                targets[key] = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                targets[key] = value.to(device)
        
        # Forward pass
        predictions = tft(x)
        
        # 디버그: 예측 dict 키/텐서 분산 출력 (첫 배치만)
        if epoch == 1 and batch_idx == 0:
            print(f"[DEBUG] === Epoch {epoch}, Batch {batch_idx} ===")
            if isinstance(predictions, dict):
                for k, v in predictions.items():
                    if isinstance(v, dict):
                        print(f"[DEBUG] pred[{k}] is nested dict:")
                        for k2, v2 in v.items():
                            arr = v2.detach().cpu().numpy()
                            print(f"[DEBUG]   pred[{k}][{k2}] shape={arr.shape} mean={arr.mean():.4g} std={arr.std():.4g}")
                    else:
                        arr = v.detach().cpu().numpy()
                        print(f"[DEBUG] pred[{k}] shape={arr.shape} mean={arr.mean():.4g} std={arr.std():.4g}")
            else:
                arr = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
                print(f"[DEBUG] pred shape={arr.shape} mean={arr.mean():.4g} std={arr.std():.4g}")
        
        # 멀티태스크 손실 계산
        losses = tft.compute_loss(predictions, targets)
        
        # 디버그: 그래디언트 흐름 확인 (첫 배치만)
        if epoch == 1 and batch_idx == 0:
            total_before = sum(p.detach().abs().sum().item() for p in tft.parameters())
            print(f"[DEBUG] param_sum_before={total_before:.4g}")
        
        # Backward pass
        losses["total"].backward()
        
        # 디버그: 그래디언트 확인 (첫 배치만)
        if epoch == 1 and batch_idx == 0:
            grad_sum = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in tft.parameters())
            print(f"[DEBUG] grad_sum={grad_sum:.4g}")
        
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
            
            # 멀티태스크 타깃 준비 - x에서 직접 추출 (올바른 방법)
            targets = _build_targets_from_x(x, horizons, val_ds)
            
            # 타깃이 이미 텐서이므로 디바이스만 확인
            for key, value in targets.items():
                if not isinstance(value, torch.Tensor):
                    targets[key] = torch.tensor(value, dtype=torch.float32, device=device)
                else:
                    targets[key] = value.to(device)
            
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
    
    # 로깅 (안전한 값으로 출력)
    auc_val = downstream_metrics.get('directional_auc', 0.5)
    ic_val = downstream_metrics.get('ic_top20', 0.0)
    vol_val = downstream_metrics.get('vol_calibration', 0.0)
    
    print(f"[TFT] epoch={epoch} "
          f"train_loss={train_losses['total']:.6f} val_loss={val_losses['total']:.6f} "
          f"AUC={auc_val:.3f} IC@20={ic_val:.3f} VolCorr={vol_val:.3f}")

    # Learning rate scheduler
    old_lr = optimizer.param_groups[0]['lr']
    if lr_schedule in ["cosine", "onecycle"]:
        scheduler.step()
    else:
        scheduler.step(val_losses['total'])
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"[TFT] Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")

    # ===== 개선된 저장/조기종료 로직 =====
    # NaN 안전 처리
    auc = float(downstream_metrics.get('directional_auc', float('nan')))
    val = float(val_losses['total'])

    is_auc_finite = math.isfinite(auc)
    loss_min_delta = float(config["tft"]["early_stopping"]["min_delta"])
    auc_min_delta = 1e-4  # AUC 개선 최소폭

    # 최초 저장 여부 (아직 저장 안 됐으면 1회 보장)
    first_save = (best_val == float("inf"))

    save_by_auc  = (is_auc_finite and (best_r2 == -float("inf") or auc > best_r2 + auc_min_delta))
    save_by_loss = (val + loss_min_delta < best_val)

    # 저장 조건:
    # 1) AUC가 유효하면 AUC 개선 시 저장
    # 2) AUC가 NaN/비유효면 loss 개선 시 저장
    # 3) 첫 에폭이면 무조건 1회 저장
    should_save = first_save or (save_by_auc or (not is_auc_finite and save_by_loss))

    if should_save:
        best_val = min(best_val, val)
        if is_auc_finite:
            best_r2 = auc  # 여기선 AUC를 1차 메트릭으로 사용
        wait = 0
        save_path = os.path.abspath("user_data/models/tft_encoder.pt")
        # state_dict 저장 (전략 로더 호환)
        torch.save(tft.state_dict(), save_path)
        print(f"[TFT] Saved best encoder (val_loss={val:.6f}, AUC={auc if is_auc_finite else float('nan'):.3f}) -> {save_path}")
    else:
        wait += 1
        if wait >= patience:
            print(f"[TFT] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

# 학습 루프 바깥에서 백업 안전장치
if not os.path.exists("user_data/models/tft_encoder.pt"):
    torch.save(tft.state_dict(), "user_data/models/tft_encoder.pt")
    print("[TFT] WARNING: No best checkpoint during training. Saved last state to user_data/models/tft_encoder.pt")

print("TFT encoder saved to user_data/models/tft_encoder.pt")

# ============================
# 4) PPO Training (e.g., 6m)
# ============================
print("PPO fine-tuning...")

def make_env(offset=0, trial_seed=None):
    """환경 생성 함수 - offset 지원 및 trial별 랜덤성 추가"""
    # trial_seed가 있으면 랜덤 오프셋 추가
    if trial_seed is not None:
        np.random.seed(trial_seed)
        # 데이터 길이의 10% 범위에서 랜덤 오프셋 추가
        max_random_offset = min(500, len(df_ppo) // 10)
        random_offset = np.random.randint(0, max_random_offset)
        offset += random_offset
        print(f"[Env] Trial seed {trial_seed}: added random offset {random_offset}")
    
    # offset이 있으면 해당 지점부터 시작하는 데이터 사용
    if offset > 0:
        df_subset = df_ppo.iloc[offset:].reset_index(drop=True)
    else:
        df_subset = df_ppo
    
    # 포지션 인식 TradingEnv - 학습용 제약 완화
    env = TradingEnv(
        df_subset, 
        tft_model=tft.eval(), 
        features=fp.features,
        reward_mode="pnl_delta",  # 포지션 델타형 보상 (step당)
        fee_bps=5,               # 탐색용 수수료 완화 (0.05%)
        slippage_bps=0           # 탐색용 슬리피지 제거
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

# Optuna 튜닝 (TFT 임베딩 캐시 포함)
best_params = tune_ppo(
    make_env, 
    config, 
    optuna_cfg_path="user_data/TFT_PPO_Training/configs/optuna_config.yml",
    tft_model=tft,
    df_ppo=df_ppo,
    feature_pipeline=fp
)

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

metrics = performance_metrics(
    np.asarray(rewards, dtype=np.float64),
    freq=config.get("timeframe", "1h"),   # ← 전략 타임프레임과 일치
    is_log_return=True                    # ← Env 보상이 로그수익이면 True
)
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
