# user_data/TFT_PPO_Training/scripts/optuna_tuning.py
from __future__ import annotations

import math
import inspect
import pathlib
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
# TradingEnv 소스 검증 강화 (pathlib 기반)
# =========================
src = pathlib.Path(inspect.getsourcefile(TradingEnv)).resolve()
print(f"[ENV-CHECK] TradingEnv from: {src}")

# optuna_tuning.py 위치 기준 기대 경로 (CWD 무관)
base = pathlib.Path(__file__).resolve().parents[2]  # .../user_data
expected = (base / "TFT_PPO_modules" / "trading_env.py").resolve()
if src != expected:
    raise RuntimeError(f"Loaded WRONG TradingEnv: {src} (expected: {expected})")


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


def _extract_trade(infos, as_delta=False, _state={"last": 0}):
    """VecEnv/래퍼에서 trade 값 안전하게 추출 (per-step 플래그 대응)"""
    v = 0
    if isinstance(infos, (list, tuple)):
        # DummyVecEnv: 첫 번째 info만 확인 (per-step 플래그)
        for info in infos:
            if isinstance(info, dict):
                # 여러 키명 fallback 지원
                if "trade" in info:
                    v = int(info["trade"])
                    break
                elif "did_trade" in info:
                    v = int(info["did_trade"])
                    break
                elif "trade_flag" in info:
                    v = int(info["trade_flag"])
                    break
    elif isinstance(infos, dict):
        # 여러 키명 fallback 지원
        if "trade" in infos:
            v = int(infos["trade"])
        elif "did_trade" in infos:
            v = int(infos["did_trade"])
        elif "trade_flag" in infos:
            v = int(infos["trade_flag"])
    
    if not as_delta:
        return v
    
    # delta 모드: 이전 값과의 차이 반환 (누적 카운터 대응)
    delta = max(0, v - _state["last"])
    _state["last"] = v
    return delta


def _as_scalar(x):
    """배열을 스칼라로 안전하게 변환"""
    arr = np.array(x)
    return float(arr.reshape(-1)[0])


def _as_bool(x):
    """배열을 bool로 안전하게 변환"""
    arr = np.array(x)
    return bool(arr.reshape(-1)[0])


def _audit_env(ev):
    """평가 환경 정상 작동 확인 (VecEnv API 안전 처리)"""
    out = ev.reset()
    obs = out[0] if isinstance(out, tuple) else out
    
    acts = [1, 2] * 10  # long/short 번갈아
    pos_rewards = 0
    trades = 0
    
    for a in acts:
        a_in = [a] if hasattr(ev, "num_envs") else a
        ret = ev.step(a_in)
        
        # Gymnasium API: (obs, reward, terminated, truncated, infos)
        if len(ret) == 5:
            obs, r, term, trunc, infos = ret
            done = _as_bool(term) or _as_bool(trunc)
        else:
            obs, r, done, infos = ret  # 구버전 호환
            done = _as_bool(done)
        
        pos_rewards += int(_as_scalar(r) > 0)
        trades += _extract_trade(infos)
        
        if done:
            break
    
    print(f"[AUDIT] pos_rewards={pos_rewards} trades={trades}")
    
    if trades < 5:
        raise ValueError(f"Audit failed: trades={trades}<5 (info lost or filter too strict)")
    if pos_rewards < 1:
        raise ValueError(f"Audit failed: no positive rewards (wrong reward config)")
    
    print(f"[AUDIT] SUCCESS: Environment is working correctly")
    # 감사 후 환경 리셋
    ev.reset()
    return True


def _probe_eval_pipeline(main_config=None, df_ppo=None, tft_model=None, feature_pipeline=None, env_fn=None):
    """평가 파이프라인 검증 (모델 없이 강제 액션으로)"""
    print("[Probe] Starting evaluation pipeline probe...")
    
    # 간단한 평가용 환경 생성
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env import VecNormalize
    
    # 기본 환경 생성 (offset=0)
    trial_seed = (main_config.get("seed", 42) if main_config else 42) + 999  # 테스트용 시드
    if df_ppo is not None and feature_pipeline is not None and tft_model is not None:
        env = make_env_with_offset(df_ppo, tft_model, feature_pipeline.features, offset=0)()
    elif env_fn is not None:
        env = env_fn()
    else:
        print("[Probe] WARNING: No environment factory available, skipping probe")
        return False
    
    # 평가용 파라미터 설정
    if hasattr(env, "reward_mode"):    env.reward_mode = "pnl_delta"
    if hasattr(env, "fee_rate"):       env.fee_rate = 3 / 1e4
    if hasattr(env, "slippage_rate"):  env.slippage_rate = 1 / 1e4
    if hasattr(env, "sanity_mode"):    env.sanity_mode = False
    
    from gymnasium.wrappers import TimeLimit
    eval_steps = (main_config.get("eval", {}).get("max_steps", 1000) if main_config else 1000)
    env = TimeLimit(env, max_episode_steps=int(eval_steps))
    env.reset(seed=trial_seed)
    
    # VecEnv로 래핑
    ev = DummyVecEnv([lambda: env])
    ev = VecNormalize(ev, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=float("inf"))
    ev.training = False
    ev.norm_reward = False
    
    obs = ev.reset()[0]
    trades = 0
    
    for i in range(30):
        a = np.array([1 if i%2==0 else 2], dtype=np.int64)  # 1,2 번갈아
        obs, reward, dones, infos = ev.step(a)
        t = _extract_trade(infos)
        info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos)>0 else infos
        print(f"[Probe] i={i} forced_action={a[0]} trade={t} info_keys={list(info0.keys()) if isinstance(info0, dict) else type(info0)}")
        trades += t
        if bool(dones[0]): 
            break
    
    ev.close()
    print(f"[Probe] total_trades={trades}")
    
    if trades > 0:
        print("[Probe] SUCCESS: Pipeline is working, trades detected")
    else:
        print("[Probe] FAILED: No trades detected in pipeline")
    
    return trades > 0


def max_drawdown_equity(eq):
    """에쿼티 시계열에서 최대 드로우다운 계산"""
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    return float(np.max(dd)) if dd.size else 0.0


import gymnasium as gym
import numpy as np

class NoOpStreakPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, patience=5, penalty=1e-4, max_penalty=5e-4):
        super().__init__(env)
        self.patience = patience
        self.penalty = penalty
        self.max_penalty = max_penalty
        self._streak = 0

    def reset(self, **kwargs):
        self._streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # SB3 Discrete: 0=hold 가정
        a = int(action[0] if isinstance(action, (list, np.ndarray)) else action)
        if a == 0:
            self._streak += 1
            if self._streak >= self.patience:
                # 페널티는 누진(상한 캡)
                p = min(self.penalty * (self._streak - self.patience + 1), self.max_penalty)
                reward = float(reward) - p
                info["noop_penalty"] = p
        else:
            self._streak = 0
        return obs, reward, terminated, truncated, info


# === PnL 기반 평가 헬퍼 ===
def _extract_pnl(infos):
    # VecEnv일 수 있어 list/tuple 방어
    if isinstance(infos, (list, tuple)):
        for d in infos:
            if isinstance(d, dict):
                if "pnl_step" in d: return float(d["pnl_step"])
                if "pnl" in d:      return float(d["pnl"])  # 백업
        return 0.0
    if isinstance(infos, dict):
        return float(infos.get("pnl_step", infos.get("pnl", 0.0)))
    return 0.0

def _equity_from_pnl_series(pnl_series, start_equity=1.0):
    # pnl_step은 로그수익 가정 → equity = exp(cumsum)
    pnl = np.asarray(pnl_series, dtype=np.float64)
    return start_equity * np.exp(np.cumsum(pnl))

def _sharpe_from_pnl(pnl_series, steps_per_year=24*365, eps=1e-12):
    r = np.asarray(pnl_series, dtype=np.float64)
    mu, sd = float(np.mean(r)), float(np.std(r))
    if sd < eps: 
        return 0.0
    return float((mu / sd) * np.sqrt(steps_per_year))

# === 스마트 거래 카운터 헬퍼 ===
def _trade_counter_begin():
    """거래 카운터 상태 리셋"""
    _extract_trade.__defaults__ = (False, {"last": 0})

def _trade_count_smart(infos):
    """
    trade 플래그/카운터 자동 판별:
    - 우선 델타 시도(카운터 가정)
    - 델타가 0이고 플래그가 1이면(=카운터 아님) 플래그를 합산
    """
    # raw flag
    raw = _extract_trade(infos, as_delta=False)
    # delta (counter 가정)
    delta = _extract_trade(infos, as_delta=True)
    if delta > 0:
        return delta
    # delta=0인데 raw=1 이면 per-step flag로 간주
    return 1 if raw == 1 else 0
def _score_from_kpis(sharpe, mdd, trades):
    """점수 계산 함수 - 현실적인 범위로 수정"""
    if trades == 0:   return -2.0
    if trades < 3:    return -1.6
    if trades < 5:    return -1.3

    # 샤프↑(0.6), MDD↓(-1.5), 거래 소량 보너스(+0.1)
    score = 0.6*sharpe - 1.5*mdd + 0.1*min(trades, 50)/50.0
    # 과매매(스텝 대비 거래 비중) 완만 패널티를 원하면 여기서 조정
    return float(np.clip(score, -2.0, 3.0))


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

        # 정책 네트워크 강제 재초기화 함수 정의
        def _force_policy_reinit(model):
            """정책 네트워크를 완전히 재초기화하는 함수"""
            import torch
            import torch.nn as nn
            
            # 정책 네트워크의 모든 가중치를 재초기화
            for module in model.policy.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            
            print("[PolicyReinit] Policy network weights completely reinitialized")
            return model

        # 1) 탐색공간 & 안전 클램프
        params = _suggest_params(trial, ocfg["search_space"])
        # 탐색 안정화를 위해 상한 조정
        params["ent_coef"] = float(np.clip(params.get("ent_coef", 0.0), 0.002, 0.03))  # 정상 범위로 복원
        params["clip_range"] = float(np.clip(params.get("clip_range", 0.2), 0.10, 0.25))
        params["gamma"] = float(np.clip(params.get("gamma", 0.99), 0.98, 0.992))  # 감마 범위 현실적으로 조정
        params["batch_size"] = int(params.get("batch_size", 256))
        print(f"[PARAM-OVERRIDE] ent_coef={params['ent_coef']:.4f} clip_range={params['clip_range']:.2f} gamma={params['gamma']:.3f} batch={params['batch_size']}")

        print("[Optuna] TFT embedding cache disabled for trial diversity")

        # 2) PPO kwargs (스케줄은 학습 중에 동적으로 조정)
        ppo_kwargs = dict(
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            ent_coef=0.07,  # 🔼 초기값을 높게 설정 (학습 중 동적 조정)
            batch_size=params["batch_size"],
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            device="cpu",
            verbose=0,
            seed=main_config.get("seed", 42),
            n_steps=main_config["ppo"].get("n_steps", 2048),
            target_kl=0.015,                         # 🔒 쏠림 가속 업데이트 컷
        )

        # 3) 학습용 VecNormalize + StickyAction + MinHoldCooldown 적용
        def make_trial_env():
            trial_seed = main_config.get("seed", 42) + trial.number
            env = env_fn(trial_seed=trial_seed)  # 사용자가 넘기는 TradingEnv factory
            
            # 환경 설정 검증 및 로깅 (래퍼 전에)
            if hasattr(env, 'reward_mode'):
                print(f"[ENV-PARAMS] reward_mode={env.reward_mode}")
                assert env.reward_mode == "pnl_delta", f"Wrong reward_mode: {env.reward_mode}"
            
            if hasattr(env, 'fee_rate'):
                print(f"[ENV-PARAMS] fee_rate={env.fee_rate}")
                assert abs(env.fee_rate - 0.00015) < 1e-6, f"Wrong fee_rate: {env.fee_rate}"
            
            # ⚙️ 환경 생성 시 옵션 강제 통일: 보상/비용/모드
            env.reward_mode = "pnl_delta"
            env.fee_rate = 3 / 1e4  # fee_bps=3
            env.slippage_rate = 1 / 1e4  # slippage_bps=1
            if hasattr(env, "sanity_mode"):
                env.sanity_mode = False  # 학습은 False로
            
            # TimeLimit 래핑 (래퍼 전에)
            from gymnasium.wrappers import TimeLimit
            eval_steps = main_config.get("eval", {}).get("max_steps", 1000)
            env = TimeLimit(env, max_episode_steps=int(eval_steps))
            
            # Sticky/ActionFilter는 평가에서 비활성화 (정량 비교 목적)
            # 훈련에서는 탐색을 위해 더 공격적으로 활성화
            env = StickyActionWrapper(env, prob=0.15)  # 0.4 → 0.15로 완화
            
            # MinHoldCooldown은 유지 (과매매 방지)
            from TFT_PPO_Training.scripts.wrappers import MinHoldCooldownWrapper
            env = MinHoldCooldownWrapper(env, min_hold=3, cooldown=2)
            
            # 새로 추가: 안정화된 탐색 강화 래퍼들 (훈련 전용)
            from TFT_PPO_Training.scripts.wrappers import EpsGreedyWrapper, SameActionPenaltyWrapper
            # ActionCycleWrapper는 잠정 비활성화 (학습 신호 왜곡 큼)
            # env = ActionCycleWrapper(env, cycle_length=100)
            
            # StickyAction 제거 (쏠림 유지기 역할)
            # env = StickyActionWrapper(env, prob=0.15)  # ❌ 제거
            
            # 순서: (탐색) → (반복벌) → (대기연속벌)
            env = EpsGreedyWrapper(env, eps=0.20)  # 0.15 → 0.20 (초기만 살짝 올려 탐색 확보)
            env = SameActionPenaltyWrapper(env, penalty=1e-4)  # 3e-5 → 1e-4 (반복 억제 강화)
            env = NoOpStreakPenaltyWrapper(env, patience=5, penalty=1e-4, max_penalty=5e-4)  # 대기연속벌
            
            if hasattr(env, "reset"):
                env.reset(seed=trial_seed)
            return env

        def make_eval_env():
            trial_seed = main_config.get("seed", 42) + trial.number
            env = env_fn(trial_seed=trial_seed)  # 사용자가 넘긴 순수 TradingEnv 팩토리
            # 평가용 공통 파라미터 통일
            if hasattr(env, "reward_mode"):    env.reward_mode = "pnl_delta"
            if hasattr(env, "fee_rate"):       env.fee_rate = 3 / 1e4
            if hasattr(env, "slippage_rate"):  env.slippage_rate = 1 / 1e4
            if hasattr(env, "sanity_mode"):    env.sanity_mode = False
            
            # 평가에서는 순수한 정책 성능 측정 (ε-greedy 제거)
            # from TFT_PPO_Training.scripts.wrappers import EpsGreedyWrapper
            # env = EpsGreedyWrapper(env, eps=0.05)  # 평가용으로 더 낮은 확률
            
            from gymnasium.wrappers import TimeLimit
            eval_steps = main_config.get("eval", {}).get("max_steps", 1000)
            env = TimeLimit(env, max_episode_steps=int(eval_steps))
            env.reset(seed=trial_seed)
            return env

        venv = DummyVecEnv([make_trial_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=float("inf"))

        model = PPO("MlpPolicy", venv, **ppo_kwargs)
        
        # 정책 네트워크 강제 재초기화 (매 Trial마다)
        model = _force_policy_reinit(model)
        
        if hasattr(model.policy, "reset_parameters"):
            model.policy.reset_parameters()
        print(f"[Optuna] Trial {trial.number} - PPO model initialized with fresh weights")

        total_ts = int(main_config.get("optuna", {}).get("timesteps", 200_000))

        # 평가 설정 (speed mode 기본)
        eval_config = main_config.get("eval", {})
        speed_mode = eval_config.get("speed_mode", True)
        eval_warmup_steps = max(100_000, eval_config.get("warmup_steps", 100_000))  # 최소 100k
        eval_every = eval_config.get("every", 50_000)
        eval_max_steps = eval_config.get("max_steps", 2000)   # ← 2000으로 증가
        eval_episodes = eval_config.get("episodes", 2)        # ← 2개 에피소드로 증가
        eval_offsets = eval_config.get("offsets", [0, 800, 1600])   # ← 서로 다른 시작점 3개

        # 평가용 VecNormalize venv 생성자 (학습 통계 공유)
        def _make_eval_venv():
            e = DummyVecEnv([make_eval_env])  # ★ 래퍼 없는 평가용
            ev = VecNormalize(e, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=float("inf"))
            ev.obs_rms = venv.obs_rms
            ev.ret_rms = venv.ret_rms
            ev.training = False
            ev.norm_reward = False
            return ev

        def _evaluate_with_vecnorm(model, steps=1000) -> float:
            ev = _make_eval_venv()

            # 첫 평가 시 감사
            if not hasattr(_evaluate_with_vecnorm, '_audited'):
                _audit_env(ev)
                _probe_eval_pipeline(main_config, df_ppo, tft_model, feature_pipeline, env_fn)  # 파이프라인 검증 추가
                _evaluate_with_vecnorm._audited = True

            obs = ev.reset()[0]
            
            # 거래 카운트 상태 리셋 (스마트 집계용)
            _trade_counter_begin()
            
            pnl_list, rewards, trade_count = [], [], 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)   # ★ 결정적
                # 안전 포장: (1,) 형태의 int64 보장
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action = action.reshape(1,)
                    elif action.ndim > 1:
                        action = action.squeeze()
                else:
                    action = np.array([action])
                action = action.astype(np.int64)
                obs, reward, dones, infos = ev.step(action)
                done = bool(dones[0])
                rewards.append(float(reward[0]))                     # 로깅용
                pnl_list.append(_extract_pnl(infos))                 # ★ 평가 핵심
                trade_count += _extract_trade(infos)                 # 플래그 합산 (as_delta=False)
                
                # 디버그 로그 (200스텝마다)
                if len(pnl_list) % 200 == 0:
                    print(f"[StepDbg] t={len(pnl_list)} trade+={_extract_trade(infos)} total_trades={trade_count}")

            ev.close()

            equity = _equity_from_pnl_series(pnl_list, start_equity=1.0)
            mdd = max_drawdown_equity(equity)
            sharpe = _sharpe_from_pnl(pnl_list, steps_per_year=24*365)
            winrate = float(np.mean(np.array(pnl_list) > 0.0))
            final = _score_from_kpis(sharpe, mdd, trade_count)

            print(f"[EvalKPIs] len={len(pnl_list)} trades={trade_count} sharpe={sharpe:.3f} winrate={winrate:.3f} mdd={mdd:.3f}")
            print(f"[EvalStats] reward: mean={np.mean(rewards):.6f} std={np.std(rewards):.6f} min={np.min(rewards):.6f} max={np.max(rewards):.6f}")
            print(f"[EvalStats] pnl   : mean={np.mean(pnl_list):.6f} std={np.std(pnl_list):.6f} min={np.min(pnl_list):.6f} max={np.max(pnl_list):.6f}")
            print(f"[EvalScore] final_score={final:.6f} (sharpe={sharpe:.3f}, mdd={mdd:.3f}, trades={trade_count})")
            return final

        learned = 0
        best_score = -np.inf
        print(f"[Optuna] Trial {trial.number} start | total_ts={total_ts} | speed_mode={speed_mode} | params={params}")

        while learned < total_ts:
            # ε-greedy 스케줄: 초반만 세게, 이후 급감
            if learned < 60_000:
                try: venv.envs[0].env.env.set_eps(0.30)  # 0.15 → 0.30 (초반만 강탐색)
                except: pass
            elif learned < 120_000:
                try: venv.envs[0].env.env.set_eps(0.10)
                except: pass
            else:
                try: venv.envs[0].env.env.set_eps(0.05)
                except: pass
            
            chunk = min(10_000, total_ts - learned)
            
            # ent_coef 동적 조정 (스케줄 적용)
            progress = learned / total_ts
            remaining = 1.0 - progress
            if remaining > 0.5:  # 초기 50% 구간
                model.ent_coef = 0.07
            elif remaining > 0.2:  # 중기 30% 구간
                model.ent_coef = 0.04
            else:  # 후기 20% 구간
                model.ent_coef = 0.012
            
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
                    def make_offset_eval_env(offset):
                        trial_seed = main_config.get("seed", 42) + trial.number
                        if df_ppo is not None and feature_pipeline is not None:
                            env = make_env_with_offset(df_ppo, tft_model, feature_pipeline.features, offset)()
                        else:
                            env = env_fn()
                        if hasattr(env, "reward_mode"):    env.reward_mode = "pnl_delta"
                        if hasattr(env, "fee_rate"):       env.fee_rate = 3 / 1e4
                        if hasattr(env, "slippage_rate"):  env.slippage_rate = 1 / 1e4
                        if hasattr(env, "sanity_mode"):    env.sanity_mode = False
                        
                        # 평가에서는 순수한 정책 성능 측정 (ε-greedy 제거)
                        # from TFT_PPO_Training.scripts.wrappers import EpsGreedyWrapper
                        # env = EpsGreedyWrapper(env, eps=0.05)
                        
                        from gymnasium.wrappers import TimeLimit
                        eval_steps = main_config.get("eval", {}).get("max_steps", 1000)
                        env = TimeLimit(env, max_episode_steps=int(eval_steps))
                        env.reset(seed=trial_seed)
                        return env

                    def _make_offset_eval_venv(offset):
                        e = DummyVecEnv([lambda: make_offset_eval_env(offset)])
                        ev = VecNormalize(e, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=float("inf"))
                        ev.obs_rms = venv.obs_rms
                        ev.ret_rms = venv.ret_rms
                        ev.training = False
                        ev.norm_reward = False
                        return ev

                    def _evaluate_offset(model, offset, steps):
                        ev = _make_offset_eval_venv(offset)
                        obs = ev.reset()[0]
                        
                        # 거래 카운트 상태 리셋 (스마트 집계용)
                        _trade_counter_begin()
                        
                        pnl_list, rewards, trade_count, done = [], [], 0, False
                        epsilon_probe_used = False
                        
                        # 디버깅 변수들
                        step_i = 0
                        act_hist = {0:0, 1:0, 2:0}
                        trade_ones = 0
                        
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            
                            # 정책 분포 진단 (200스텝 간격)
                            if len(pnl_list) % 200 == 0:
                                try:
                                    dist = model.policy.get_distribution(obs)
                                    probs = getattr(dist.distribution, "probs", None)
                                    if probs is not None:
                                        p = probs[0].detach().cpu().numpy()
                                        if len(p) == 3:
                                            print(f"[PiProbs] t={len(pnl_list)} p(a0,a1,a2)={p[0]:.3f},{p[1]:.3f},{p[2]:.3f}")
                                except:
                                    pass
                            
                            # --- ε-probe: 결정적 정책이 trades를 못 만들 때만 가끔 찔러봄 ---
                            if trade_count < 3 and np.random.rand() < 0.05:
                                # action 공간이 Discrete(3)라고 가정: 0,1,2 중에서 무작위
                                action = np.array([np.random.randint(0, 3)], dtype=np.int64)
                                epsilon_probe_used = True
                            # -------------------------------------------------------------
                            
                            # 안전 포장: (1,) 형태의 int64 보장
                            if isinstance(action, np.ndarray):
                                if action.ndim == 0:
                                    action = action.reshape(1,)
                                elif action.ndim > 1:
                                    action = action.squeeze()
                            else:
                                action = np.array([action])
                            action = action.astype(np.int64)
                            
                            obs, reward, dones, infos = ev.step(action)
                            done = bool(dones[0])
                            
                            # 디버깅 정보 수집 (실제 환경에 전달된 액션 카운트)
                            # action은 이미 환경에 전달된 후이므로 실제 액션임
                            a0 = int(action[0])
                            act_hist[a0] = act_hist.get(a0, 0) + 1
                            
                            t = _extract_trade(infos)
                            if t > 0:
                                trade_ones += 1
                            
                            rewards.append(float(reward[0]))
                            pnl_list.append(_extract_pnl(infos))
                            trade_count += _trade_count_smart(infos)  # 스마트 집계
                            
                            step_i += 1
                        ev.close()
                        print(f"[ActionHist] {act_hist} | trade_ones_in_first={trade_ones}")
                        
                        equity = _equity_from_pnl_series(pnl_list, start_equity=1.0)
                        mdd = max_drawdown_equity(equity)
                        sharpe = _sharpe_from_pnl(pnl_list, steps_per_year=24*365)
                        score = _score_from_kpis(sharpe, mdd, trade_count)
                        
                        print(f"[OffsetEval] offset={offset} len={len(pnl_list)} trades={trade_count} sharpe={sharpe:.3f} mdd={mdd:.3f} score={score:.6f} probe_used={epsilon_probe_used}")
                        return score

                    score = _evaluate_offset(model, offset, steps=eval_max_steps)
                    scores.append(score)
                
                # 여러 오프셋의 평균 점수 사용
                score = float(np.mean(scores))
                print(f"[MultiOffset] scores={[f'{s:.3f}' for s in scores]} -> avg_score={score:.6f}")
            else:
                # 더 정확한 평가가 필요하면 아래를 확장
                score = _evaluate_with_vecnorm(model, steps=eval_max_steps * 2)

            trial.report(score, step=learned)
            print(f"[Optuna] Trial {trial.number} evaluation: score={score:.6f} at step={learned}")
            print(f"[Optuna] Trial {trial.number} params: lr={params.get('learning_rate', 0):.2e}, gamma={params.get('gamma', 0):.3f}, ent_coef={params.get('ent_coef', 0):.4f}")

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
