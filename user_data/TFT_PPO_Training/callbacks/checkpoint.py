# user_data/TFT_PPO_Training/callbacks/checkpoint.py

import os
import json
from datetime import datetime
from typing import Dict, Any


class ModelCheckpoint:
    """
    ModelCheckpoint
    ----------------
    Monitors a chosen validation metric (e.g., Sharpe, Sortino, reward mean)
    and automatically saves the best model weights.

    Compatible with Stable-Baselines3 PPO or any model exposing `.save()`.

    Example
    -------
    >>> ckpt = ModelCheckpoint(save_dir="user_data/models/best", metric_name="sharpe", min_delta=0.001)
    >>> improved = ckpt.update(model, metrics)
    >>> if improved:
    ...     print("New best model saved.")
    """

    def __init__(
        self,
        save_dir: str = "user_data/models/best",
        metric_name: str = "sharpe",
        mode: str = "max",
        min_delta: float = 1e-4,
        higher_is_better: bool | None = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        save_dir : str
            Directory to store model checkpoints.
        metric_name : str
            Name of metric to monitor (e.g. 'sharpe', 'sortino', 'avg_reward').
        mode : {'max', 'min'}
            Whether higher or lower metric values are better.
        min_delta : float
            Minimum change in metric to qualify as improvement.
        higher_is_better : bool | None
            Deprecated alias for mode (auto-resolved if provided).
        verbose : bool
            Print log messages when True.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.verbose = verbose

        # Backward compatibility for older argument
        if higher_is_better is not None:
            self.mode = "max" if higher_is_better else "min"

        self.best_score = -float("inf") if self.mode == "max" else float("inf")
        self.last_checkpoint_path: str | None = None

    def _is_improved(self, current_score: float) -> bool:
        if self.mode == "max":
            return current_score > self.best_score + self.min_delta
        return current_score < self.best_score - self.min_delta

    def update(self, model: Any, metrics: Dict[str, float]) -> bool:
        """
        Evaluate current metric and save the model if improved.

        Parameters
        ----------
        model : PPO or torch.nn.Module
            Trained model (must implement `.save()` or have `state_dict()`).
        metrics : dict
            Dictionary of validation metrics.

        Returns
        -------
        bool
            True if model was saved, False otherwise.
        """
        if self.metric_name not in metrics:
            if self.verbose:
                print(f"[Checkpoint] Metric '{self.metric_name}' not found in metrics dict.")
            return False

        current_score = float(metrics[self.metric_name])
        if not self._is_improved(current_score):
            if self.verbose:
                print(
                    f"[Checkpoint] {self.metric_name} did not improve: "
                    f"{current_score:.6f} (best={self.best_score:.6f})"
                )
            return False

        # Update best score and save model
        self.best_score = current_score
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.metric_name}_{current_score:.4f}_{timestamp}.zip"
        path = os.path.join(self.save_dir, filename)

        try:
            if hasattr(model, "save"):
                model.save(path)
            elif hasattr(model, "state_dict"):
                import torch
                torch.save(model.state_dict(), path)
            else:
                raise AttributeError("Model does not support save() or state_dict().")
        except Exception as e:
            print(f"[Checkpoint] Error saving model: {e}")
            return False

        # Save accompanying metadata
        meta = {
            "timestamp": timestamp,
            "metric": self.metric_name,
            "best_score": current_score,
            "mode": self.mode,
        }
        with open(path.replace(".zip", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"[Checkpoint] New best model saved: {path} ({self.metric_name}={current_score:.4f})")

        # Optional: remove previous checkpoint if exists (to save disk)
        if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
            try:
                os.remove(self.last_checkpoint_path)
                os.remove(self.last_checkpoint_path.replace(".zip", "_meta.json"))
            except OSError:
                pass

        self.last_checkpoint_path = path
        return True
