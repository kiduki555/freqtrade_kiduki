# user_data/TFT_PPO_Training/scripts/optuna_tuning.py
from __future__ import annotations
import optuna
import numpy as np
import yaml
from typing import Callable, Dict, Any
from stable_baselines3 import PPO
from TFT_PPO_modules.performance_metrics import performance_metrics
from TFT_PPO_Training.scripts.utils import set_seed

def _build_sampler(scfg: Dict[str, Any]):
    t = scfg.get("type", "TPESampler")
    if t == "TPESampler":
        return optuna.samplers.TPESampler(seed=scfg.get("seed", 42))
    return optuna.samplers.TPESampler(seed=42)

def _build_pruner(pcfg: Dict[str, Any]):
    t = pcfg.get("type", "MedianPruner")
    if t == "MedianPruner":
        return optuna.pruners.MedianPruner(
            n_startup_trials=pcfg.get("n_startup_trials", 5),
            n_warmup_steps=pcfg.get("n_warmup_steps", 2),
        )
    return optuna.pruners.MedianPruner()

def _suggest_params(trial: optuna.trial.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    p = {}
    for k, spec in space.items():
        t = spec["type"]
        if t == "loguniform":
            p[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
        elif t == "uniform":
            p[k] = trial.suggest_float(k, spec["low"], spec["high"])
        elif t == "categorical":
            p[k] = trial.suggest_categorical(k, spec["choices"])
        else:
            raise ValueError(f"Unknown search type: {t}")
    return p

def _evaluate(model: PPO, make_env: Callable[[], Any], episodes: int, freq: str) -> float:
    # 간단한 평가 루프: 마지막 에피소드 Sharpe 반환
    # 더 보수적으로 하려면 평균 Sharpe를 반환해도 됨
    metrics_list = []
    for _ in range(episodes):
        env = make_env()
        obs = env.reset()
        # Gymnasium 호환
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]
        done = False
        if isinstance(done, list):
            done = done[0]
        rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _info = step_out
            rewards.append(float(reward if np.ndim(reward) == 0 else reward[0]))
        m = performance_metrics(np.asarray(rewards, dtype=np.float64), freq=freq)
        metrics_list.append(m)
        if hasattr(env, "close"):
            env.close()
    # 평균 Sharpe 반환
    return float(np.nanmean([m["sharpe"] for m in metrics_list]))

def tune_ppo(
    env_fn: Callable[[], Any],
    main_config: Dict[str, Any],
    optuna_cfg_path: str = "user_data/TFT_PPO_Training/configs/optuna_config.yml",
) -> Dict[str, Any]:
    with open(optuna_cfg_path, "r") as f:
        ocfg = yaml.safe_load(f)["optuna"]

    sampler = _build_sampler(ocfg.get("sampler", {}))
    pruner = _build_pruner(ocfg.get("pruner", {}))
    study = optuna.create_study(
        study_name=ocfg.get("study_name", "ppo_tuning"),
        direction=ocfg.get("direction", "maximize"),
        storage=ocfg.get("storage", None),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    eval_freq = int(ocfg.get("eval_freq", 50000))
    eval_episodes = int(ocfg.get("eval_episodes", 2))
    n_trials = int(ocfg.get("n_trials", 20))
    timeout = ocfg.get("timeout", None)
    n_jobs = int(ocfg.get("n_jobs", 1))
    data_freq = main_config.get("eval", {}).get("freq", "hourly")  # 혹은 "daily"로 변경

    def objective(trial: optuna.trial.Trial) -> float:
        set_seed(main_config.get("seed", 42) + trial.number)

        params = _suggest_params(trial, ocfg["search_space"])

        # PPO 기본 하이퍼파라미터(정합성)
        # CPU 강제 (SB3 권고: MLP 정책은 CPU가 유리)
        ppo_kwargs = dict(
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            device="cpu",
            verbose=0,
            seed=main_config.get("seed", 42),
            n_steps=main_config["ppo"].get("n_steps", 2048),
        )

        env = env_fn()
        model = PPO("MlpPolicy", env, **ppo_kwargs)

        total_ts = int(main_config.get("optuna", {}).get("timesteps", 200_000))
        trained = 0
        best_score = -np.inf

        while trained < total_ts:
            step = min(eval_freq, total_ts - trained)
            model.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False)
            trained += step

            score = _evaluate(model, env_fn, episodes=eval_episodes, freq=data_freq)
            trial.report(score, step=trained)

            if score > best_score:
                best_score = score

            if trial.should_prune():
                env.close()
                raise optuna.TrialPruned()

        env.close()
        return best_score

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, gc_after_trial=True)

    print("Best parameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params
