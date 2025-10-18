# user_data/TFT_PPO_modules/performance_metrics.py
import numpy as np

def _infer_periods_per_year(freq: str) -> int:
    """시간프레임에 따른 연간 주기 수 계산"""
    f = (freq or "").lower()
    if f in ("1h", "h", "hour", "hourly"): 
        return 24 * 365
    if f in ("4h", "4hour"): 
        return 6 * 365
    if f in ("15m", "15min"): 
        return 4 * 24 * 365
    if f in ("1d", "d", "day", "daily"): 
        return 365
    # 안전 기본값: 1시간봉 가정
    return 24 * 365

def performance_metrics(
    rewards: np.ndarray,
    freq: str = "1h",
    is_log_return: bool = True,
    risk_free_rate: float = 0.0
) -> dict:
    """
    RL 평가용 성능 메트릭 계산 - 시간프레임 정합 연율화
    
    Parameters
    ----------
    rewards : np.ndarray
        보상 배열 (로그수익 또는 단순수익)
    freq : str
        데이터 주기 ("1h", "daily" 등)
    is_log_return : bool
        True면 로그수익, False면 단순수익으로 가정
    risk_free_rate : float
        연간 무위험 수익률
    """
    r = np.asarray(rewards, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "mdd": 0.0, "win_rate": 0.0}

    # 로그수익이면 그대로, 단순수익이면 로그로 변환해서 합산 일관화
    if not is_log_return:
        r = np.log1p(r)  # 단순 -> 로그

    periods = _infer_periods_per_year(freq)
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    sd_down = float(np.std(np.clip(r, a_max=0.0, a_min=None), ddof=1))  # 하방 변동만

    # 방어적 계산
    eps = 1e-12
    sharpe = (mu / max(sd, eps)) * np.sqrt(periods)
    sortino = (mu / max(sd_down, eps)) * np.sqrt(periods)

    # 에쿼티/최대낙폭
    eq = np.exp(np.cumsum(r))  # 로그수익 누적 → 에쿼티
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, eps)
    mdd = float(np.max(dd)) if dd.size else 0.0

    # 간단 승률(양의 구간 비중)
    win_rate = float(np.mean(r > 0.0))

    return {
        "sharpe": sharpe, 
        "sortino": sortino, 
        "calmar": (mu / (mdd + eps)) if mdd > 0 else np.nan,
        "mdd": mdd, 
        "win_rate": win_rate,
        "avg_return": mu,
        "volatility": sd
    }
