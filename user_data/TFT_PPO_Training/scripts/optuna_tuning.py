# user_data/TFT_PPO_Training/scripts/optuna_tuning.py
from __future__ import annotations

import math
from typing import Callable, Dict, Any

import numpy as np
import optuna
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from TFT_PPO_modules.performance_metrics import performance_metrics  # (남겨둠: 필요시 사용)
from TFT_PPO_modules.trading_env import TradingEnv                  # (타입 힌트 용도)
from TFT_PPO_Training.scripts.utils import set_seed


# =========================
# Sampler / Pruner builders
# =========================
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
    elif t == "SuccessiveHalvingPruner":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pcfg.get("min_resource", 1),
            reduction_factor=pcfg.get("reduction_factor", 3),
        )
    return optuna.pruners.MedianPruner()


# =========================
# Small utilities
# =========================
def _as_float(x):
    """문자열이나 다른 타입을 float로 변환하고 유효성 검증"""
    try:
        return float(x)
    except Exception:
        raise ValueError(f"search_space value must be float-compatible. got={x} ({type(x)})")


def _suggest_params(trial: optuna.trial.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """하이퍼파라미터 제안 함수 - 분포 안전장치 포함"""
    p = {}
    for k, spec in space.items():
        t = spec["type"]
        if t in ("loguniform", "uniform"):
            low = _as_float(spec["low"])
            high = _as_float(spec["high"])
            if low > high:
                low, high = high, low
            if t == "loguniform":
                if low <= 0:
                    raise ValueError(f"{k}: loguniform low must be > 0 (got {low})")
                p[k] = trial.suggest_float(k, low, high, log=True)
            else:
                p[k] = trial.suggest_float(k, low, high)
        elif t == "categorical":
            p[k] = trial.suggest_categorical(k, spec["choices"])
        else:
            raise ValueError(f"Unknown search type: {t}")
    return p


def make_env_with_offset(df, tft, features, offset, **kwargs):
    """서로 다른 시작 오프셋을 가진 환경 생성"""
    sub = df.iloc[offset:].reset_index(drop=True)
    return lambda: TradingEnv(sub, tft, features, **kwargs)


def rolling_tft_encode(tft_model, X, win=96):
    """
    TFT 모델로 롤링 임베딩 계산 (캐시용)
    X: (N, F) float32 (fp.features 순서)
    반환: (N, d_model) enc_last를 롤링으로 생성. 부족한 앞구간은 0.
    """
    import torch

    tft_model.eval()
    d_model = tft_model.hidden_size
    embs = np.zeros((len(X), d_model), dtype=np.float32)

    with torch.no_grad():
        for i in range(win, len(X)):
            window = X[i - win : i]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(tft_model.device)  # (1, win, F)
            try:
                if hasattr(tft_model.tft, "encoder"):
                    encoder_output = tft_model.tft.encoder({"encoder_cont": window_tensor})  # (1, win, d_model)
                    enc_last = encoder_output[:, -1, :].squeeze(0).cpu().numpy()
                    embs[i] = enc_last
                else:
                    tft_output = tft_model.tft({"encoder_cont": window_tensor})
                    if hasattr(tft_output, "encoder_output"):
                        enc_last = tft_output.encoder_output[:, -1, :].squeeze(0).cpu().numpy()
                        embs[i] = enc_last
                    else:
                        raise Exception("No encoder output found")
            except Exception:
                # Fallback: 간단한 선형 투영
                if not hasattr(tft_model, "_cache_proj"):
                    tft_model._cache_proj = torch.nn.Linear(window.shape[1], d_model).to(tft_model.device)
                proj_in = torch.FloatTensor(window.mean(axis=0)).unsqueeze(0).to(tft_model.device)
                embs[i] = tft_model._cache_proj(proj_in).squeeze(0).cpu().numpy()

    print(f"[TFT Cache] Generated embeddings: {embs.shape}, mean={embs.mean():.6f}, std={embs.std():.6f}")
    return embs


def max_drawdown_equity(eq):
    """에쿼티 시계열에서 최대 드로우다운 계산"""
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    return float(np.max(dd)) if dd.size else 0.0


# =========================
# 평가 유틸 (속도모드 포함)
# =========================
def _evaluate_vectorized(model, make_env_fn, episodes: int = 3, fee_bps: float = 10.0, ann_factor: float = 365.0):
    """
    간단한 환경 보상 기반 평가 (필터 적용된 환경 사용)
    """
    from gymnasium.wrappers import TimeLimit

    scores = []

    for episode in range(episodes):
        env = make_env_fn(offset=episode * 500)  # 서로 다른 시작점
        eval_max_steps = 1000
        if not isinstance(env, TimeLimit):
            env = TimeLimit(env, max_episode_steps=eval_max_steps)

        obs = env.reset()[0]
        rewards = []
        trade_count = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rewards.append(float(reward))
            
            # ✅ 실제 거래 카운트
            if isinstance(info, dict):
                trade_count += int(info.get("trade", 0))

        env.close()
        
        # 보상 기반 점수 계산
        if len(rewards) < 10:
            scores.append(-1.0)
            continue
            
        score = _score_rewards(rewards, freq="1h", trades_override=trade_count)
        scores.append(score)

    score = float(np.mean(scores))
    print(f"[EvalDebug] Evaluation score: {score:.6f}")
    return score


def _score_rewards(rewards, freq="1h", trades_override=None):
    """보상 시퀀스를 KPI 점수로 변환 - 상세 로그 포함(스피드모드에서 사용)"""
    if not rewards or len(rewards) < 10:
        print(f"[EvalDebug] Too short rewards: len={len(rewards) if rewards else 0} -> penalty -1.0")
        return -1.0

    rew = np.array(rewards, dtype=float)
    mu = float(np.mean(rew))
    sd = float(np.std(rew, ddof=1))
    if sd == 0 or not np.isfinite(sd):
        sharpe = -1.0
    else:
        ann = 24*365 if freq in ["1h", "h", "hour", "hourly"] else 365
        sharpe = (mu / sd) * math.sqrt(ann)

    winrate = float(np.mean(rew > 0))
    equity = np.exp(np.cumsum(rew))
    peak = np.maximum.accumulate(equity)
    dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))

    # ✅ 거래 횟수는 override가 있으면 그것을 사용
    if trades_override is not None:
        trades = int(trades_override)
    else:
        trades = 0  # 안전 기본값

    # 과매매 페널티 기준을 "스텝 대비 비율"로 변경
    steps = len(rew)
    overtrading_ratio = trades / max(1, steps)
    penalty_reason = "none"

    if trades == 0:
        score = -2.0
        penalty_reason = "trades=0"
    elif trades < 5:
        score = -1.5 + (trades / 5.0) * 0.5
        penalty_reason = f"trades={trades}<5"
    elif overtrading_ratio > 0.5:              # 스텝의 50% 초과 거래면 과매매
        score = -1.0
        penalty_reason = f"overtrading={trades}"
    else:
        score = (
            0.6 * np.clip(sharpe, -2.0, 5.0)
            + 0.3 * (winrate - 0.5) * 2.0
            - 0.1 * np.clip(dd, 0.0, 0.5) * 5
            + 0.1 * np.clip(trades / 50.0, 0.0, 1.0)
        )
        penalty_reason = "none"

    print(f"[EvalDebug] len={len(rew)} | trades={trades} | sharpe={sharpe:.3f} | winrate={winrate:.3f} | mdd={dd:.3f} | penalty={penalty_reason} -> score={score:.3f}")
    print(f"[EvalDebug] reward_stats: mean={mu:.6f} | std={sd:.6f} | min={rew.min():.6f} | max={rew.max():.6f}")
    return float(score)


def _evaluate(model: PPO, make_env_fn, episodes: int, freq: str) -> float:
    ann_factor = 24 * 365 if freq in ["1h", "h", "hour", "hourly"] else 365
    score = _evaluate_vectorized(model, make_env_fn=make_env_fn, episodes=episodes, fee_bps=10.0, ann_factor=ann_factor)
    return score


# =========================
# 안정화 래퍼
# =========================
class StickyActionWrapper(gym.Wrapper):
    """초기 탐색 안정화를 위한 sticky action"""
    def __init__(self, env, prob: float = 0.25):
        super().__init__(env)
        self.prob = prob
        self._last = None

    def step(self, action):
        import random

        if self._last is not None and random.random() < self.prob:
            action = self._last
        self._last = action
        return self.env.step(action)


# =========================
# Optuna main
# =========================
def tune_ppo(
    env_fn: Callable[[], Any],
    main_config: Dict[str, Any],
    optuna_cfg_path: str = "user_data/TFT_PPO_Training/configs/optuna_config.yml",
    tft_model=None,
    df_ppo=None,
    feature_pipeline=None,
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

    # 전역/기본 평가 설정
    n_trials = int(ocfg.get("n_trials", 20))
    timeout = ocfg.get("timeout", None)
    n_jobs = int(ocfg.get("n_jobs", 1))
    data_freq = main_config.get("timeframe", "1h")

    def objective(trial: optuna.trial.Trial) -> float:
        # ✅ 학습은 비결정 (평가만 결정)
        set_seed(main_config.get("seed", 42) + trial.number, deterministic=False)
        
        # 디버그: 파일 경로/시간 확인
        import os, time
        print(f"[DEBUG] using optuna_tuning.py at {__file__} mtime={time.ctime(os.path.getmtime(__file__))}")

        # 1) 탐색공간 & 안전 클램프
        params = _suggest_params(trial, ocfg["search_space"])
        # SB3 일반 권장 범위에 맞춰 보수적으로 (탐색 과잉 방지)
        params["ent_coef"] = float(np.clip(params.get("ent_coef", 0.0), 0.0, 0.05))
        params["clip_range"] = float(np.clip(params.get("clip_range", 0.2), 0.10, 0.25))
        params["gamma"] = float(np.clip(params.get("gamma", 0.99), 0.98, 0.995))  # 감마 범위 현실적으로 조정
        params["batch_size"] = int(params.get("batch_size", 256))
        print(f"[PARAM-OVERRIDE] ent_coef={params['ent_coef']:.4f} clip_range={params['clip_range']:.2f} gamma={params['gamma']:.3f} batch={params['batch_size']}")

        print("[Optuna] TFT embedding cache disabled for trial diversity")

        # 2) PPO kwargs
        ppo_kwargs = dict(
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            device="cpu",
            verbose=0,
            seed=main_config.get("seed", 42),
            n_steps=main_config["ppo"].get("n_steps", 2048),
        )

        # 3) 학습용 VecNormalize + StickyAction + MinHoldCooldown 적용
        def make_trial_env():
            trial_seed = main_config.get("seed", 42) + trial.number
            env = env_fn(trial_seed=trial_seed)  # 사용자가 넘기는 TradingEnv factory
            
            # ⚙️ 환경 생성 시 옵션 강제 통일: 보상/비용/모드
            env.reward_mode = "pnl_delta"
            env.fee_rate = 3 / 1e4  # fee_bps=3
            env.slippage_rate = 1 / 1e4  # slippage_bps=1
            if hasattr(env, "sanity_mode"):
                env.sanity_mode = False  # 학습은 False로
            
            env = StickyActionWrapper(env, prob=0.25)
            # 액션 안정화를 위한 최소 보유시간 + 쿨다운 래퍼 추가
            from TFT_PPO_Training.scripts.wrappers import MinHoldCooldownWrapper
            env = MinHoldCooldownWrapper(env, min_hold=3, cooldown=2)  # 과매매 방지 완화
            if hasattr(env, "reset"):
                env.reset(seed=trial_seed)
            return env

        venv = DummyVecEnv([make_trial_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)

        model = PPO("MlpPolicy", venv, **ppo_kwargs)
        if hasattr(model.policy, "reset_parameters"):
            model.policy.reset_parameters()
        print(f"[Optuna] Trial {trial.number} - PPO model initialized with fresh weights")

        total_ts = int(main_config.get("optuna", {}).get("timesteps", 200_000))

        # 평가 설정 (speed mode 기본)
        eval_config = main_config.get("eval", {})
        speed_mode = eval_config.get("speed_mode", True)
        eval_warmup_steps = max(100_000, eval_config.get("warmup_steps", 100_000))  # 최소 100k
        eval_every = eval_config.get("every", 50_000)
        eval_max_steps = eval_config.get("max_steps", 1000)   # ← 1000으로 증가
        eval_episodes = eval_config.get("episodes", 2)        # ← 2개 에피소드로 증가
        eval_offsets = eval_config.get("offsets", [0, 800])   # ← 서로 다른 시작점 2개

        # 평가용 VecNormalize venv 생성자 (학습 통계 공유)
        def _make_eval_venv():
            e = DummyVecEnv([make_trial_env])
            ev = VecNormalize(e, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
            ev.obs_rms = venv.obs_rms
            ev.ret_rms = venv.ret_rms
            ev.training = False
            ev.norm_reward = False
            
            # ⚙️ 평가 환경도 동일한 옵션 적용
            for env in ev.envs:
                if hasattr(env, 'env'):
                    env = env.env
                env.reward_mode = "pnl_delta"
                env.fee_rate = 3 / 1e4  # fee_bps=3
                env.slippage_rate = 1 / 1e4  # slippage_bps=1
                if hasattr(env, "sanity_mode"):
                    env.sanity_mode = False
            
            return ev

        def _evaluate_with_vecnorm(model, steps=1000) -> float:
            ev = _make_eval_venv()
            # TimeLimit: VecNormalize 내부 env에 적용 (reset 전에)
            from gymnasium.wrappers import TimeLimit
            if not isinstance(ev.envs[0], TimeLimit):
                ev.envs[0] = TimeLimit(ev.envs[0], max_episode_steps=int(steps))
            obs = ev.reset()[0]
            rewards, done = [], False
            trade_count = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # DummyVecEnv는 배열을 기대하므로 액션을 배열로 감싸기
                if not isinstance(action, (list, np.ndarray)) or (isinstance(action, np.ndarray) and action.ndim == 0):
                    action = [action]
                elif isinstance(action, np.ndarray) and action.ndim == 1 and len(action) == 1:
                    action = action.tolist()
                obs, reward, dones, infos = ev.step(action)
                done = bool(dones[0])
                rewards.append(float(reward[0]))
                
                # ✨ 환경 info에서 실제 거래수 합산 (VecNormalize 환경 고려)
                if isinstance(infos, (list, tuple)) and len(infos) > 0:
                    info = infos[0]
                    if isinstance(info, dict):
                        trade_flag = int(info.get("trade", 0))
                        trade_count += trade_flag
                        if trade_flag > 0:  # 디버깅용
                            print(f"[DEBUG] Trade detected: {trade_flag}")
                    elif hasattr(info, 'get'):
                        trade_flag = int(info.get("trade", 0))
                        trade_count += trade_flag
                        if trade_flag > 0:  # 디버깅용
                            print(f"[DEBUG] Trade detected: {trade_flag}")
            ev.close()
            return _score_rewards(rewards, freq=data_freq, trades_override=trade_count)

        learned = 0
        best_score = -np.inf
        print(f"[Optuna] Trial {trial.number} start | total_ts={total_ts} | speed_mode={speed_mode} | params={params}")

        while learned < total_ts:
            chunk = min(10_000, total_ts - learned)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            learned += chunk
            print(f"[Optuna] Trial {trial.number} learned {learned}/{total_ts} timesteps ({learned/total_ts*100:.1f}%)")

            # 워ーム업: 100k 이전엔 평가 skip
            if learned < eval_warmup_steps:
                print(f"[Optuna] Trial {trial.number} warmup phase - skipping evaluation")
                continue
            if learned % eval_every != 0:
                continue

            # 평가 실행 (여러 오프셋으로 평균)
            if speed_mode:
                scores = []
                for offset in eval_offsets:
                    # 오프셋별 환경 생성 (df_ppo와 feature_pipeline이 있는 경우에만)
                    if df_ppo is not None and feature_pipeline is not None:
                        def make_offset_env():
                            trial_seed = main_config.get("seed", 42) + trial.number
                            env = make_env_with_offset(df_ppo, tft_model, feature_pipeline.features, offset)()
                            env = StickyActionWrapper(env, prob=0.25)
                            from TFT_PPO_Training.scripts.wrappers import MinHoldCooldownWrapper
                            env = MinHoldCooldownWrapper(env, min_hold=3, cooldown=2)  # 과매매 방지 완화
                            if hasattr(env, "reset"):
                                env.reset(seed=trial_seed)
                            return env
                    else:
                        # 기존 방식 사용
                        def make_offset_env():
                            trial_seed = main_config.get("seed", 42) + trial.number
                            env = env_fn()
                            
                            # ⚙️ 환경 생성 시 옵션 강제 통일: 보상/비용/모드
                            env.reward_mode = "pnl_delta"
                            env.fee_rate = 3 / 1e4  # fee_bps=3
                            env.slippage_rate = 1 / 1e4  # slippage_bps=1
                            if hasattr(env, "sanity_mode"):
                                env.sanity_mode = False
                            
                            env = StickyActionWrapper(env, prob=0.25)
                            from TFT_PPO_Training.scripts.wrappers import MinHoldCooldownWrapper
                            env = MinHoldCooldownWrapper(env, min_hold=3, cooldown=2)  # 과매매 방지 완화
                            if hasattr(env, "reset"):
                                env.reset(seed=trial_seed)
                            return env
                    
                    # 오프셋별 평가용 환경 생성
                    def _make_offset_eval_venv():
                        e = DummyVecEnv([make_offset_env])
                        ev = VecNormalize(e, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
                        ev.obs_rms = venv.obs_rms
                        ev.ret_rms = venv.ret_rms
                        ev.training = False
                        ev.norm_reward = False
                        return ev
                    
                    def _evaluate_offset(model, steps=eval_max_steps) -> float:
                        ev = _make_offset_eval_venv()
                        # TimeLimit: VecNormalize 내부 env에 적용 (reset 전에)
                        from gymnasium.wrappers import TimeLimit
                        if not isinstance(ev.envs[0], TimeLimit):
                            ev.envs[0] = TimeLimit(ev.envs[0], max_episode_steps=int(steps))
                        obs = ev.reset()[0]
                        rewards, done = [], False
                        trade_count = 0
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            # DummyVecEnv는 배열을 기대하므로 액션을 배열로 감싸기
                            if not isinstance(action, (list, np.ndarray)) or (isinstance(action, np.ndarray) and action.ndim == 0):
                                action = [action]
                            elif isinstance(action, np.ndarray) and action.ndim == 1 and len(action) == 1:
                                action = action.tolist()
                            obs, reward, dones, infos = ev.step(action)
                            done = bool(dones[0])
                            rewards.append(float(reward[0]))
                            
                            # ✨ 환경 info에서 실제 거래수 합산
                            if isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
                                trade_count += int(infos[0].get("trade", 0))
                        ev.close()
                        return _score_rewards(rewards, freq=data_freq, trades_override=trade_count)
                    
                    score = _evaluate_offset(model, steps=eval_max_steps)
                    scores.append(score)
                
                # 여러 오프셋의 평균 점수 사용
                score = float(np.mean(scores))
            else:
                # 더 정확한 평가가 필요하면 아래를 확장
                score = _evaluate_with_vecnorm(model, steps=eval_max_steps * 2)

            trial.report(score, step=learned)
            print(f"[Optuna] Trial {trial.number} evaluation: score={score:.6f} at step={learned}")
            print(
                f"[Optuna] Trial {trial.number} params: "
                f"lr={params['learning_rate']:.2e}, gamma={params['gamma']:.3f}, ent_coef={params['ent_coef']:.4f}"
            )

            if score > best_score:
                best_score = score

            if trial.should_prune():
                venv.close()
                raise optuna.TrialPruned()

        venv.close()
        return best_score

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, gc_after_trial=True)

    print("Best parameters found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params
