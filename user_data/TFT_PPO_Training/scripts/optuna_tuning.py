# user_data/TFT_PPO_Training/scripts/optuna_tuning.py

import optuna
import numpy as np
from stable_baselines3 import PPO
from TFT_PPO_Modules.performance_metrics import performance_metrics
from TFT_PPO_Training.scripts.utils import set_seed


def tune_ppo(env_fn, config, seed: int = 42):
    """
    Hyperparameter tuning for PPO using Optuna.

    Parameters
    ----------
    env_fn : callable
        Function that returns a fresh instance of the training environment.
    config : dict
        Configuration dictionary containing tuning parameters, e.g.:
        {"optuna": {"trials": 20, "timesteps": 300_000}}
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """

    def objective(trial):
        # Reproducibility per trial
        set_seed(seed + trial.number)

        # Suggest hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.99),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }

        # Instantiate new environment for each trial
        env = env_fn()

        try:
            model = PPO("MlpPolicy", env, **params, verbose=0, seed=seed)
            model.learn(total_timesteps=config["optuna"]["timesteps"])

            # Evaluate policy performance
            returns = getattr(env, "episode_returns", None)
            if returns is None or len(returns) == 0:
                returns = np.array(env.rewards) if hasattr(env, "rewards") else np.array([])

            metrics = performance_metrics(np.array(returns)) if len(returns) > 0 else {"sharpe": 0}
            score = metrics.get("sharpe", 0)

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            score = 0
        finally:
            env.close()

        return score

    # Study setup
    study = optuna.create_study(direction="maximize", study_name="ppo_tuning")
    study.optimize(objective, n_trials=config["optuna"]["trials"])

    print("Best parameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    return study.best_params
