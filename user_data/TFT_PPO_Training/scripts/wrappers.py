# user_data/TFT_PPO_Training/scripts/wrappers.py
import gymnasium as gym
import numpy as np

class MinHoldCooldownWrapper(gym.Wrapper):
    """액션 안정화를 위한 최소 보유시간 + 쿨다운 래퍼"""
    def __init__(self, env, min_hold=3, cooldown=2):
        super().__init__(env)
        self.min_hold = int(min_hold)
        self.cooldown = int(cooldown)
        self._hold = 0
        self._cool = 0
        self._last_action = 0  # 0=Hold, 1=Buy, 2=Sell

    def reset(self, **kwargs):
        self._hold = 0
        self._cool = 0
        self._last_action = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        a = int(action)

        # 쿨다운 중엔 진입/청산 제한
        if self._cool > 0:
            if a in (1,2):  # 방향 전환성 액션
                a = 0

        # 최소 보유시간: 직전이 Buy 상태였다면 즉시 Sell 금지
        # env에 현재 포지션을 노출한다면 활용(없으면 _last_action으로 보수적 제한)
        if self._hold < self.min_hold and self._last_action == 1 and a == 2:
            a = 0

        obs, reward, terminated, truncated, info = self.env.step(a)

        # 타이머 갱신
        if a == 1:  # 진입
            self._hold = 1
            self._cool = self.cooldown
        elif a == 0 and self._hold > 0:  # 유지
            self._hold += 1
            self._cool = max(0, self._cool - 1)
        elif a == 2:  # 청산
            self._hold = 0
            self._cool = self.cooldown

        self._last_action = a
        return obs, reward, terminated, truncated, info

class PriceTapWrapper(gym.Wrapper):
    """
    step()의 info에 'close'를 항상 넣어주는 래퍼.
    env가 내부에 price 시계열을 가지고 있으면 자동으로 참조.
    """
    def __init__(self, env):
        super().__init__(env)
        self._price_accessor = None
        self._t = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # gymnasium: (obs, info), gym: obs
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._t = 0
        self._price_accessor = self._detect_price_accessor()
        if self._price_accessor is not None:
            price = float(self._price_accessor(self._t))
            info = dict(info or {})
            info["close"] = price
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if self._price_accessor is None:
            self._price_accessor = self._detect_price_accessor()
        if self._price_accessor is not None:
            try:
                price = float(self._price_accessor(self._t))
                info = dict(info or {})
                info["close"] = price
            except Exception:
                pass
        return obs, reward, terminated, truncated, info

    # 내부 가격 탐색(환경마다 다름 → 여러 후보 지원)
    def _detect_price_accessor(self):
        env = self.env
        # wrapper 경유 변수는 get_wrapper_attr로 조회 (경고 해결)
        for attr in ["prices", "closes", "close", "price"]:
            try:
                seq = env.get_wrapper_attr(attr)
            except Exception:
                seq = getattr(getattr(env, "unwrapped", env), attr, None)
            if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) > 0:
                return lambda t, _seq=seq: _seq[min(t, len(_seq)-1)]
        # pandas DataFrame 보관 패턴
        for attr in ["df", "data", "dataset", "_data", "_dataset"]:
            try:
                obj = env.get_wrapper_attr(attr)
            except Exception:
                obj = getattr(getattr(env, "unwrapped", env), attr, None)
            if obj is not None:
                try:
                    import pandas as pd  # noqa
                    if hasattr(obj, "iloc") and "close" in getattr(obj, "columns", []):
                        return lambda t, _df=obj: _df["close"].iloc[min(t, len(_df)-1)]
                except Exception:
                    pass
        return None
