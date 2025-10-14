# user_data/TFT_PPO_modules/ppo_agent.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from stable_baselines3 import PPO


class PPOAgent:
    """
    PPOAgent
    --------
    Production-friendly wrapper around Stable-Baselines3 PPO for trading inference.

    Key features:
      - Device-aware loading (CPU/GPU)
      - Safe input shaping & dtype control (float32)
      - Optional observation normalization (running mean/std) for stability at inference
      - Recurrent policy compatibility (sb3-contrib RecurrentPPO-style .predict)
      - Batch prediction API
      - Action probabilities for diagnostics
      - Episode boundary handling (for recurrent policies)

    Notes
    -----
    - If you trained with VecNormalize, prefer loading the VecNormalize wrapper
      and calling model.predict through that wrapper instead of ad-hoc normalization.
    """

    def __init__(
        self,
        model_path: str,
        device: Union[str, "torch.device"] = "auto",
        obs_dim: Optional[int] = None,
        use_running_norm: bool = False,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to SB3 PPO .zip checkpoint.
        device : str or torch.device
            "auto" | "cpu" | "cuda" (delegated to SB3).
        obs_dim : Optional[int]
            Expected observation dimension (for sanity checks). If None, no check.
        use_running_norm : bool
            If True, applies an internal running mean/std normalizer at inference.
            Use only when you did not export VecNormalize with the policy.
        eps : float
            Numerical stability epsilon for normalization.
        """
        self.model = PPO.load(model_path, device=device)
        self.obs_dim = obs_dim
        self.use_running_norm = use_running_norm
        self.eps = eps

        # Running normalization state (for non-VecNormalize deployments)
        self._count = 0
        self._mean = None
        self._m2 = None  # sum of squares of diffs (Welford)

        # Recurrent policy support (sb3-contrib RecurrentPPO-compatible)
        self._rnn_state = None
        self._episode_start = True  # must be True at the beginning of each episode

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def reset(self):
        """
        Reset any internal recurrent state between episodes.
        Call this at env.reset().
        """
        self._rnn_state = None
        self._episode_start = True

    def predict(
        self,
        state: np.ndarray,
        deterministic: Optional[bool] = None,
        episode_start: Optional[bool] = None,
    ) -> int:
        """
        Predict a single action from a 1D observation.

        - Handles both feed-forward PPO and (if available) recurrent PPO signatures.
        - Ensures correct shape [1, obs_dim] and dtype float32.

        Parameters
        ----------
        state : np.ndarray
            Observation vector [obs_dim].
        deterministic : Optional[bool]
            If None, defaults to True for evaluation-like inference.
        episode_start : Optional[bool]
            Set True at the first step of an episode (for recurrent policies).

        Returns
        -------
        int
            Discrete action id (e.g., 0=Hold, 1=Buy, 2=Sell)
        """
        x = self._prepare_obs(state)
        det = True if deterministic is None else deterministic

        if episode_start is not None:
            self._episode_start = bool(episode_start)

        # Try recurrent signature first (sb3-contrib RecurrentPPO)
        try:
            action, self._rnn_state = self.model.predict(
                x, state=self._rnn_state, episode_start=np.array([self._episode_start]), deterministic=det
            )
        except TypeError:
            # Fallback to standard SB3 PPO
            action, _ = self.model.predict(x, deterministic=det)

        # After first call in an episode, mark as continuation
        self._episode_start = False
        return int(action)

    def predict_batch(
        self,
        states: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Vectorized prediction for a batch of observations.

        Parameters
        ----------
        states : np.ndarray
            Array of shape [batch, obs_dim].
        deterministic : Optional[bool]
            See `predict`.

        Returns
        -------
        np.ndarray
            Actions of shape [batch], dtype=int.
        """
        X = self._prepare_obs_batch(states)
        det = True if deterministic is None else deterministic

        # Batch: recurrent API typically expects per-env episode_starts; use zeros
        try:
            actions, _ = self.model.predict(
                X, state=None, episode_start=np.zeros((X.shape[0],), dtype=bool), deterministic=det
            )
        except TypeError:
            actions, _ = self.model.predict(X, deterministic=det)

        return actions.astype(int)

    def action_proba(self, state: np.ndarray) -> np.ndarray:
        """
        Return action probabilities/logits for debugging/analytics if available.

        Note: SB3 exposes distribution via policy; here we run a forward pass.

        Returns
        -------
        np.ndarray
            Probabilities over discrete actions if accessible, else empty array.
        """
        try:
            x = self._prepare_obs(state)
            policy = self.model.policy
            with policy.noise_context(False):
                dist = policy.get_distribution(policy.obs_to_tensor(x)[0])
                # For discrete actions, dist.distribution.probs exists
                probs = getattr(dist.distribution, "probs", None)
                if probs is not None:
                    return probs.detach().cpu().numpy().squeeze()
        except Exception:
            pass
        return np.array([])

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _prepare_obs(self, state: np.ndarray) -> np.ndarray:
        """Validate/reshape/normalize a single observation to [1, obs_dim], float32."""
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if self.obs_dim is not None and arr.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {arr.shape[0]}")
        if self.use_running_norm:
            arr = self._apply_running_norm(arr)
        return arr.reshape(1, -1)

    def _prepare_obs_batch(self, states: np.ndarray) -> np.ndarray:
        """Validate/normalize a batch of observations to [B, obs_dim], float32."""
        X = np.asarray(states, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.obs_dim is not None and X.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {X.shape[1]}")
        if self.use_running_norm:
            X = np.vstack([self._apply_running_norm(row) for row in X])
        return X

    # ---------------- Running normalization (Welford) -------------------- #
    def _apply_running_norm(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None:
            # init
            self._mean = np.zeros_like(x, dtype=np.float64)
            self._m2 = np.zeros_like(x, dtype=np.float64)
            self._count = 0

        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

        var = (self._m2 / max(self._count - 1, 1)).astype(np.float32)
        std = np.sqrt(np.maximum(var, 0.0)) + self.eps
        return ((x - self._mean.astype(np.float32)) / std).astype(np.float32)
