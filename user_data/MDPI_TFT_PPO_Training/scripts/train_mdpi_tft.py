# user_data/TFT_PPO_Training/scripts/train_mdpi_tft.py
"""
Train MultiTaskTFT in **MDPI style**:
- target: next-24h return (log return) as main head
- optional aux heads: direction (BCE), realized volatility (Huber)
- normalization: z / log / logit (fit on train only)
- export best checkpoint + scaler stats for reproducible inference
"""
import os, yaml, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from TFT_PPO_modules.feature_pipeline import FeaturePipeline
from TFT_PPO_modules.multi_task_tft import MultiTaskTFT, create_multi_horizon_targets
from TFT_PPO_Training.scripts.mdpi_normalization import MDPIStandardizer

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def build_timeseries_dataset(df, config):
    # Build TimeSeriesDataSet using return_24h as "target"
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
        time_varying_unknown_reals=config["tft"]["feature_cols"] + ["return_24h"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

def train_mdpi(config_path="user_data/TFT_PPO_Training/configs/mdpi_tft.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load OHLCV
    df = pd.read_csv(cfg["data"]["ohlcv_path"])
    # expect columns: date, open, high, low, close, volume, asset(optional)
    if "asset" not in df.columns:
        df["asset"] = cfg["data"].get("asset", "ASSET")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset","date"]).reset_index(drop=True)

    # 2) Feature engineering
    fp = FeaturePipeline()
    feat_df = df.groupby("asset", group_keys=False).apply(lambda g: fp.compute_all(g)).reset_index(drop=True)

    # 3) Targets (24h, 48h, 96h log-returns + direction/volatility)
    feat_df = feat_df.groupby("asset", group_keys=False).apply(lambda g: create_multi_horizon_targets(g, horizons=[24,48,96])).reset_index(drop=True)

    # 4) Train/Val split by date
    cutoff = pd.to_datetime(cfg["split"]["train_val_cut"])
    train = feat_df[feat_df["date"] < cutoff].copy()
    val   = feat_df[feat_df["date"] >= cutoff].copy()

    # 5) MDPI normalization (fit on train only), per asset
    feature_cols = [c for c in feat_df.columns if c not in ["date","asset","open","high","low","close","return_24h","return_48h","return_96h","direction","volatility"]]
    scaler = MDPIStandardizer(group_cols=["asset"])
    train[feature_cols] = scaler.fit_transform(train, feature_cols)[feature_cols]
    val[feature_cols]   = scaler.transform(val, feature_cols)[feature_cols]

    # Fill NaNs after transforms
    for d in (train, val):
        d[feature_cols] = d[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # 6) Indexing for TFT
    for d in (train, val):
        d["time_idx"] = (d.groupby("asset")["date"].rank(method="first").astype(int) - 1)
        d["group_id"] = d["asset"]

    # 7) Build datasets
    tsd_train = build_timeseries_dataset(train, cfg)
    tsd_val   = build_timeseries_dataset(val, cfg)
    dl_train  = tsd_train.to_dataloader(train=True,  batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])
    dl_val    = tsd_val.to_dataloader(train=False, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"]["num_workers"])

    # 8) Model
    tft = MultiTaskTFT(
        dataset=tsd_train,
        hidden_size=cfg["tft"]["hidden_size"],
        attention_head_size=cfg["tft"]["attention_heads"],
        dropout=cfg["tft"]["dropout"],
        horizons=[24,48,96],
        loss_weights={"returns":1.0, "direction":0.5, "volatility":0.25},
        device=device
    )

    # 9) Training loop (simple, uses internal loss dict)
    logger = CSVLogger(save_dir="user_data/TFT_PPO_Training/logs", name="mdpi_tft")
    callbacks = [EarlyStopping(monitor="val_total_loss", min_delta=1e-4, patience=cfg["train"]["early_stop_patience"], mode="min")]
    trainer = Trainer(
        max_epochs=cfg["train"]["max_epochs"],
        accelerator="gpu" if device=="cuda" else "cpu",
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=10,
        callbacks=[]
    )

    best_state = None
    best_val = float("inf")

    for epoch in range(1, cfg["train"]["max_epochs"]+1):
        tft.train()
        tr_losses = {"returns_total":0.0,"direction":0.0,"volatility":0.0,"total_loss":0.0}
        for batch in dl_train:
            # Prepare inputs
            x = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            # Build targets dict from batch
            targets = {
                "returns": { "horizon_24": batch["return_24h"].to(device).unsqueeze(-1),
                             "horizon_48": batch["return_48h"].to(device).unsqueeze(-1),
                             "horizon_96": batch["return_96h"].to(device).unsqueeze(-1)},
                "direction": (batch["direction"].to(device).unsqueeze(-1)),
                "volatility": (batch["volatility"].to(device).unsqueeze(-1)),
            }
            preds = tft(x)
            losses = tft.compute_losses(preds, targets)
            loss = sum([losses[k]*w for k,w in zip(["returns_total","direction","volatility"], [1.0,0.5,0.25])])
            tft.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tft.parameters(), cfg["train"]["grad_clip"])
            for p in tft.parameters():
                if p.grad is not None: p.data -= cfg["train"]["lr"] * p.grad  # simple SGD
            for k in tr_losses: tr_losses[k] += float(losses.get(k,0.0))

        # Validation
        tft.eval()
        vl_losses = {"returns_total":0.0,"direction":0.0,"volatility":0.0,"total_loss":0.0}
        with torch.no_grad():
            for batch in dl_val:
                x = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                targets = {
                    "returns": { "horizon_24": batch["return_24h"].to(device).unsqueeze(-1),
                                 "horizon_48": batch["return_48h"].to(device).unsqueeze(-1),
                                 "horizon_96": batch["return_96h"].to(device).unsqueeze(-1)},
                    "direction": (batch["direction"].to(device).unsqueeze(-1)),
                    "volatility": (batch["volatility"].to(device).unsqueeze(-1)),
                }
                preds = tft(x)
                losses = tft.compute_losses(preds, targets)
                for k in vl_losses: vl_losses[k] += float(losses.get(k,0.0))

        ntr = max(len(dl_train),1); nvl = max(len(dl_val),1)
        tr = {k: v/ntr for k,v in tr_losses.items()}
        vl = {k: v/nvl for k,v in vl_losses.items()}
        val_total = vl["returns_total"] + 0.5*vl["direction"] + 0.25*vl["volatility"]
        print(f"[E{epoch:03}] train={tr} | val={vl}")

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu() for k,v in tft.state_dict().items()}

        # Early stop
        if epoch - 1 >= cfg["train"]["early_stop_patience"] and val_total > best_val:
            pass

    # Save best
    ensure_dir("user_data/models/best")
    ckpt_path = "user_data/models/best/mdpi_tft.pt"
    torch.save(best_state, ckpt_path)

    # Save scaler stats
    import pickle, json
    with open("user_data/models/best/mdpi_scaler.pkl","wb") as f:
        pickle.dump(scaler, f)
    with open("user_data/models/best/mdpi_config.json","w") as f:
        json.dump(cfg, f, indent=2)

    print("[OK] saved:", ckpt_path)
    return ckpt_path

if __name__ == "__main__":
    train_mdpi()
