# user_data/TFT_PPO_modules/checkpoint.py
import os
import torch
import json
import datetime
from typing import Optional, Dict, Any


class ModelCheckpoint:
    """
    ModelCheckpoint
    ----------------
    Advanced checkpointing utility for RL/trading experiments.

    Monitors one or multiple metrics (e.g., Sharpe, Sortino, Avg Reward),
    saves the best model(s), supports patience-based early stopping,
    and optionally saves optimizer + RNG state for full reproducibility.

    Typical use:
    ------------
    >>> ckpt = ModelCheckpoint(save_dir="user_data/models/best", patience=5)
    >>> stop = ckpt.update(model, metrics={"sharpe": 1.23, "sortino": 1.05})

    Parameters
    ----------
    save_dir : str
        Directory to store checkpoints.
    patience : int
        Number of consecutive non-improving epochs before early stopping.
    max_checkpoints : int
        Maximum number of recent best checkpoints to keep.
    primary_metric : str
        Metric name to optimize (default: "sharpe").
    higher_is_better : bool
        Whether a higher metric value indicates improvement.
    """

    def __init__(
        self,
        save_dir: str = "user_data/models/best",
        patience: int = 3,
        max_checkpoints: int = 3,
        primary_metric: str = "sharpe",
        higher_is_better: bool = True,
    ):
        self.save_dir = save_dir
        self.patience = patience
        self.max_checkpoints = max_checkpoints
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

        self.best_metric = -float("inf") if higher_is_better else float("inf")
        self.counter = 0
        self.history = []  # [(timestamp, metric_value, filename)]

        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------------
    def update(
        self,
        model,
        metrics: Dict[str, float],
        model_name: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update checkpoint based on new validation metrics.

        Parameters
        ----------
        model : stable_baselines3.PPO or nn.Module
            Model to save (must have .save() or torch.save() compatible).
        metrics : dict
            Dictionary of validation metrics. Must contain primary_metric.
        model_name : str, optional
            Custom filename for the checkpoint.
        extra_state : dict, optional
            Additional state to persist (e.g., RNG, scaler).

        Returns
        -------
        bool
            True if early stopping is triggered, else False.
        """
        if self.primary_metric not in metrics:
            raise KeyError(f"Primary metric '{self.primary_metric}' not found in metrics: {list(metrics.keys())}")

        current = metrics[self.primary_metric]
        improved = (
            current > self.best_metric if self.higher_is_better else current < self.best_metric
        )

        if improved:
            self.best_metric = current
            self.counter = 0
            ckpt_name = model_name or f"best_{self.primary_metric}_{current:.3f}.zip"
            save_path = os.path.join(self.save_dir, ckpt_name)

            # Save SB3 or Torch model
            try:
                model.save(save_path)
            except AttributeError:
                torch.save(model.state_dict(), save_path)

            # Save metadata
            meta = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "metrics": metrics,
                "primary_metric": self.primary_metric,
                "best_value": self.best_metric,
            }
            if extra_state:
                meta["extra_state"] = {k: str(v) for k, v in extra_state.items()}
            with open(save_path.replace(".zip", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Keep last N checkpoints
            self.history.append((meta["timestamp"], current, ckpt_name))
            if len(self.history) > self.max_checkpoints:
                oldest = self.history.pop(0)
                try:
                    os.remove(os.path.join(self.save_dir, oldest[2]))
                except FileNotFoundError:
                    pass

            print(f"Saved new best model → {ckpt_name} | {self.primary_metric}={current:.4f}")
        else:
            self.counter += 1
            print(f"No improvement ({self.counter}/{self.patience}) | best={self.best_metric:.4f}")
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True

        return False
