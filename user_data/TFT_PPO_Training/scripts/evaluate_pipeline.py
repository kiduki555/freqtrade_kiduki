# user_data/TFT_PPO_Training/scripts/evaluate_pipeline.py
from __future__ import annotations

import numpy as np
from typing import Optional, Callable, Union

from stable_baselines3 import PPO
from TFT_PPO_modules.performance_metrics import performance_metrics


def evaluate_pipeline(
    env: Union[object, Callable[[], object]],
    model_path: str = "user_data/models/ppo_policy.zip",
    n_episodes: int = 1,
    deterministic: bool = True,
    freq: str = "daily",
    risk_free_rate: float = 0.0,
    render: bool = False,
) -> dict:
    """
    Evaluate a trained PPO policy on a given environment.

    Supports both Gymnasium and legacy Gym step/reset signatures.
    Prints aggregated metrics across episodes and returns the last-episode metrics.

    Parameters
    ----------
    env : Env instance or callable
        If callable, it must return a fresh env per episode (recommended).
        If an env instance is passed, it will be reused and reset each episode.
    model_path : str
        Path to the saved Stable-Baselines3 PPO model (.zip).
    n_episodes : int
        Number of evaluation episodes to run.
    deterministic : bool
        Deterministic policy evaluation flag.
    freq : str
        Data frequency for annualization in performance_metrics (e.g., "daily", "hourly").
    risk_free_rate : float
        Annualized risk-free rate for Sharpe/Sortino calculations.
    render : bool
        If True and env supports it, calls env.render() each step.

    Returns
    -------
    dict
        Performance metrics for the last episode evaluated.
    """
    model = PPO.load(model_path)

    def _reset(_env):
        # Gymnasium: obs, info
        out = _env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        # Legacy Gym: obs
        return out

    def _step(_env, action):
        # Gymnasium: obs, reward, terminated, truncated, info
        out = _env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        # Legacy Gym: obs, reward, done, info
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    # For aggregated reporting
    all_episode_metrics = []

    for ep in range(n_episodes):
        # Fresh env per episode if factory is provided
        _env = env() if callable(env) else env
        obs = _reset(_env)
        rewards = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = _step(_env, action)
            rewards.append(float(reward))
            if render and hasattr(_env, "render"):
                _env.render()

        returns = np.asarray(rewards, dtype=np.float64)
        metrics = performance_metrics(returns, freq=freq, risk_free_rate=risk_free_rate)
        all_episode_metrics.append(metrics)

        print(
            f"Episode {ep+1}/{n_episodes} | "
            f"Sharpe={metrics['sharpe']:.4f} | "
            f"Sortino={metrics['sortino']:.4f} | "
            f"Calmar={metrics['calmar']:.4f} | "
            f"MDD={metrics['mdd']:.4f} | "
            f"WinRate={metrics['win_rate']:.2%}"
        )

        # If we created a fresh env, close it
        if callable(env) and hasattr(_env, "close"):
            _env.close()

    # Aggregate summary (mean across episodes)
    if n_episodes > 1:
        keys = all_episode_metrics[0].keys()
        agg = {k: float(np.nanmean([m[k] for m in all_episode_metrics])) for k in keys}
        print(
            "Aggregate (mean across episodes) | "
            f"Sharpe={agg['sharpe']:.4f} | Sortino={agg['sortino']:.4f} | "
            f"Calmar={agg['calmar']:.4f} | MDD={agg['mdd']:.4f} | WinRate={agg['win_rate']:.2%}"
        )

    return all_episode_metrics[-1]
