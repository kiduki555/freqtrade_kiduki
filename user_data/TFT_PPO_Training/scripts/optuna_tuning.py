# user_data/TFT_PPO_Training/scripts/optuna_tuning.py
from __future__ import annotations
import optuna
import numpy as np
import math
import yaml
from typing import Callable, Dict, Any
from stable_baselines3 import PPO
from TFT_PPO_modules.performance_metrics import performance_metrics
from TFT_PPO_modules.trading_env import TradingEnv
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
            # 자동 정렬 (뒤집혀 들어오면 바로 교체)
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

def max_drawdown_equity(eq):
    """에쿼티 시계열에서 최대 드로우다운 계산"""
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    return float(np.max(dd)) if dd.size else 0.0

def _evaluate_vectorized(model, make_env_fn, episodes: int = 3, fee_bps: float = 10.0, ann_factor: float = 365.0):
    """
    모델의 행동으로 전이기반 엔트리/엑싯을 만들고, 간단 롱온리 PnL/Sharpe를 계산.
    - fee_bps: 왕복 수수료 기준 bps(=0.01% -> 1 bps). 여기선 편의상 진입/청산에 절반씩 부과.
    - ann_factor: 일간 수익률 기준 연율화 계수(crypto 일일 365 추천; 분/시봉이면 사용 주기에 맞게 수정)
    반환: 평균 Sharpe (여러 오프셋/에피소드 평균)
    """
    scores = []
    
    # 안전한 offset 산출
    # 임시 env 하나 열어 길이 추정
    probe_env = make_env_fn(offset=0)
    try:
        length_hint = None
        try:
            length_hint = probe_env.get_wrapper_attr("length")
        except Exception:
            pass
        if length_hint is None:
            length_hint = getattr(getattr(probe_env, "unwrapped", probe_env), "length", None)
        if length_hint is None:
            length_hint = getattr(probe_env, "n_steps", None)
        if length_hint is None:
            # 러프하게 10_000 스텝 가정, 이후 too short면 penalty
            length_hint = 10_000
        print(f"[EvalDebug] Detected data length: {length_hint}")
    except Exception as e:
        print(f"[EvalDebug] Failed to detect data length: {e}, using default 10000")
        length_hint = 10_000
    finally:
        probe_env.close()
    
    # 전체 길이의 50% 내에서 에피소드 시작 (더 안전하게)
    safe_max_offset = max(100, int(length_hint * 0.5))  # 최소 100 스텝 보장
    raw_offsets = [0, min(1000, safe_max_offset // 2), min(2000, safe_max_offset)]
    offsets = raw_offsets[:max(1, episodes)]
    print(f"[EvalDebug] Using offsets: {offsets} (data_length={length_hint})")

    for off in offsets:
        # --- 1) 평가용 env/데이터 준비
        env = make_env_fn(offset=off)  # 반드시 새 env
        out = env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        closes = []
        acts = []

        done = False
        while not done:
            # 결정적으로 예측 (평가)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            # 가격(or 로그수익) 시퀀스 확보
            if isinstance(info, dict) and "close" in info:
                closes.append(info["close"])
            elif isinstance(info, dict) and "price" in info:
                closes.append(info["price"])
            # 마지막 방어막: info에 없으면 PriceTapWrapper가 못 찾은 케이스 → env에서 직접 추출 시도
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "prices"):
                p = env.unwrapped.prices
                idx = min(len(closes), len(p)-1)
                closes.append(float(p[idx]))
            # 액션 기록
            acts.append(int(action))

        env.close()
        if len(closes) < 50:  # 최소 길이 기준 상향 (너무 짧은 에피소드는 버림)
            print("[EvalDebug] too short episode; returning penalty")
            scores.append(-1.0)
            continue

        # --- 2) 액션 시퀀스 → 전이기반 엔트리/엑싯
        a = np.array(acts, dtype=int)                     # 0=Hold, 1=Buy, 2=Sell
        prev = np.roll(a, 1); prev[0] = 0
        enter_long = (a == 1) & (prev != 1)
        exit_long  = (prev == 1) & (a != 1)

        # 상태 기반 마스킹(약식): 빈 포지션에서 Sell은 무시
        pos = np.zeros_like(a, dtype=int)
        open_flag = 0
        for t in range(len(a)):
            if open_flag == 0 and enter_long[t]:
                open_flag = 1
            elif open_flag == 1 and exit_long[t]:
                open_flag = 0
            pos[t] = open_flag

        # ✅ 1) 마지막에 포지션이 열려있으면 강제 청산
        force_exit = False
        if open_flag == 1:
            exit_long[-1] = True
            force_exit = True

        # ✅ 2) 트레이드 페어 구성 (엔트리-엑싯)
        ent_idx = np.where(enter_long)[0]
        ex_idx = np.where(exit_long)[0]
        pairs = []
        j = 0
        for i in ent_idx:
            while j < len(ex_idx) and ex_idx[j] <= i:
                j += 1
            if j < len(ex_idx):
                pairs.append((i, ex_idx[j]))
                j += 1

        trades = len(pairs)
        if trades == 0:
            print("[EvalDebug] trades=0 → penalty score")
            scores.append(-1.0)
            continue

        # --- 3) 벡터화 PnL 계산 (log-return + 수수료)
        c = np.array(closes, dtype=float)
        ret = np.zeros_like(c)
        ret[1:] = np.log(c[1:] / c[:-1])                  # 로그수익
        # 포지션은 직전 시점 보유분이 수익에 기여
        pnl = pos * ret

        # 수수료: 진입/청산 시점에 half-half 부과
        delta_pos = np.zeros_like(pos); delta_pos[1:] = pos[1:] - pos[:-1]
        # 진입/청산 모두 절댓값 1 변화 → half_fee  each
        half_fee = (fee_bps / 1e4) / 2.0                 # bps → 비율, 절반
        fees = np.abs(delta_pos) * half_fee
        pnl_after_fee = pnl - fees

        # === 복합 KPI 계산 ===
        # 1) 샤프 비율 (연율화)
        if len(pnl_after_fee) < 2:
            sharpe = -1.0
        else:
            mu = float(np.mean(pnl_after_fee))
            sd = float(np.std(pnl_after_fee, ddof=1))
            if sd == 0 or not np.isfinite(sd):
                sharpe = -1.0
            else:
                sharpe = (mu / sd) * math.sqrt(ann_factor)

        # 2) 승률 계산 (트레이드별 승패)
        wins = 0
        losses = 0
        equity = [1.0]
        cur = 1.0
        
        # 에쿼티 시계열 생성
        for t in range(1, len(c)):
            cur *= np.exp(pnl_after_fee[t])  # 로그수익 누적
            equity.append(cur)
        
        # ✅ 3) 승률/트레이드 계산 (강제청산 포함)
        for i, j in pairs:
            trade_logret = np.sum(pnl_after_fee[i+1:j+1])
            if trade_logret > 0:
                wins += 1
            else:
                losses += 1
        
        actual_trades = len(pairs)
        winrate = wins / max(1, actual_trades)
        
        # 3) 최대 드로우다운
        maxdd = max_drawdown_equity(equity)
        
        # 4) 복합 스코어 계산 (Sharpe 0.6, Winrate 0.3, DD 0.1)
        kpi = {
            "sharpe": sharpe,
            "winrate": winrate,
            "maxdd": maxdd
        }
        
        score_ep = (
            0.6 * np.clip(kpi["sharpe"], -2.0, 5.0) +
            0.3 * (kpi["winrate"] - 0.5) * 2.0 +        # 50% 기준 정규화
            -0.1 * np.clip(kpi["maxdd"], 0.0, 0.5) * 5  # DD 50%-> -0.5 패널티
        )

        act_vals, act_cnts = np.unique(a, return_counts=True)
        act_dist = {int(v): int(c) for v, c in zip(act_vals, act_cnts)}
        print(f"[EvalDebug] off={off} | acts={act_dist} | len={len(a)} | trades={actual_trades} (forced_exit={force_exit}) | "
              f"sharpe={sharpe:.4f} | winrate={winrate:.3f} | maxdd={maxdd:.3f} | score={score_ep:.4f}")
        scores.append(float(score_ep))

    # 여러 오프셋 평균
    score = float(np.mean(scores))
    print(f"[EvalDebug] Evaluation score: {score:.6f}")
    return score

def _evaluate(model: PPO, make_env_fn, episodes: int, freq: str) -> float:
    """벡터화된 백테스트를 사용한 평가 함수"""
    # 벡터화된 평가 함수 사용
    score = _evaluate_vectorized(model, make_env_fn=make_env_fn, episodes=episodes, fee_bps=10.0, ann_factor=365.0)
    return score

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
        # ✅ 학습은 비결정 (평가만 결정)
        set_seed(main_config.get("seed", 42) + trial.number, deterministic=False)

        params = _suggest_params(trial, ocfg["search_space"])

        # PPO 기본 하이퍼파라미터(정합성)
        # CPU 강제 (SB3 권고: MLP 정책은 CPU가 유리)
        ppo_kwargs = dict(
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            batch_size=params["batch_size"],
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
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
        print(f"[Optuna] Trial {trial.number} start | total_ts={total_ts} | params={params}")

        while trained < total_ts:
            step = min(eval_freq, total_ts - trained)
            model.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False)
            trained += step
            # 진행 로그 (엔트로피/업데이트 수 등은 로거에 안 뜰 수 있어 간략 로그)
            print(f"[Optuna] Trial {trial.number} learned {trained}/{total_ts} timesteps")

            # 벡터화된 평가 함수 사용 - 정책 행동 반영
            score = _evaluate(model, env_fn, episodes=eval_episodes, freq=data_freq)
            trial.report(score, step=trained)
            
            # 평가 직후 요약 로그 (KPI 포함)
            print(f"[Optuna] Trial {trial.number} summary: score={score:.6f} at step={trained}")
            print(f"[Optuna] Trial {trial.number} params: lr={params['learning_rate']:.2e}, "
                  f"gamma={params['gamma']:.3f}, ent_coef={params['ent_coef']:.4f}")

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
