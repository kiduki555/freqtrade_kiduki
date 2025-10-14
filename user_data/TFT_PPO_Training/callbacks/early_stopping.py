# user_data/TFT_PPO_Training/callbacks/early_stopping.py

import numpy as np
import time
from typing import Optional, Dict


class EarlyStopping:
    """
    EarlyStopping
    -------------
    Generic early stopping utility for RL or time-series model training.

    Terminates training when a monitored validation metric (e.g., Sharpe ratio)
    fails to improve beyond `min_delta` for `patience` consecutive evaluations.

    Example
    -------
    >>> stopper = EarlyStopping(patience=5, min_delta=0.001)
    >>> for epoch in range(100):
    ...     metric = validate_model(...)
    ...     if stopper.should_stop(metric):
    ...         break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "max",
        metric_name: str = "sharpe",
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        patience : int
            Number of evaluations without improvement before stopping.
        min_delta : float
            Minimum change in metric value to qualify as improvement.
        mode : {"max", "min"}
            Whether higher or lower metric values are better.
        metric_name : str
            Name of the metric being monitored (for logging).
        verbose : bool
            If True, prints progress messages.
        """
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.metric_name = metric_name
        self.verbose = verbose

        self.best_metric = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.start_time = time.time()

    def should_stop(self, current_metric: Optional[float]) -> bool:
        """
        Evaluate current metric and decide whether to stop training.

        Parameters
        ----------
        current_metric : float
            Latest validation metric.

        Returns
        -------
        bool
            True if training should stop.
        """
        if current_metric is None or np.isnan(current_metric) or np.isinf(current_metric):
            if self.verbose:
                print(f"Warning: Invalid metric value ({current_metric}); skipping check.")
            return False

        improved = (
            current_metric > self.best_metric + self.min_delta
            if self.mode == "max"
            else current_metric < self.best_metric - self.min_delta
        )

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] New best {self.metric_name}: {current_metric:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement in {self.metric_name} "
                    f"({self.counter}/{self.patience}). Best={self.best_metric:.6f}"
                )

        if self.counter >= self.patience:
            elapsed = time.time() - self.start_time
            if self.verbose:
                print(f"[EarlyStopping] Triggered after {elapsed:.1f}s — no improvement in {self.metric_name}.")
            return True

        return False

    def state_dict(self) -> Dict[str, float]:
        """Return current stopping state."""
        return {
            "best_metric": float(self.best_metric),
            "counter": int(self.counter),
            "patience": int(self.patience),
            "elapsed_sec": float(time.time() - self.start_time),
        }

    def reset(self) -> None:
        """Reset stopping state (useful for new training runs)."""
        self.best_metric = -np.inf if self.mode == "max" else np.inf
        self.counter = 0
        self.start_time = time.time()
