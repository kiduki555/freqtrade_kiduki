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


class EpsGreedyWrapper(gym.Wrapper):
    """ε-greedy 액션 선택 래퍼 (훈련 전용)"""
    def __init__(self, env, eps=0.10):
        super().__init__(env)
        self.eps = eps
    
    def step(self, action):
        if np.random.rand() < self.eps:
            action = self.action_space.sample()
        return self.env.step(action)


class ActionCycleWrapper(gym.Wrapper):
    """액션 순환을 강제하여 단일 액션 고착을 방지하는 래퍼"""
    def __init__(self, env, cycle_length=100):  # 더 긴 순환
        super().__init__(env)
        self.cycle_length = cycle_length
        self.step_count = 0
        # 더 균등한 순환: 각 액션을 동일하게 반복
        self.forced_cycle = []
        for i in range(cycle_length):
            self.forced_cycle.append(i % 3)  # 0,1,2를 균등하게 반복
        
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        # 처음 cycle_length 스텝은 강제 순환
        if self.step_count < self.cycle_length:
            forced_action = self.forced_cycle[self.step_count]
            obs, reward, terminated, truncated, info = self.env.step(forced_action)
            self.step_count += 1
            return obs, reward, terminated, truncated, info
        else:
            # 이후에는 정상 액션 사용
            return self.env.step(action)


class SameActionPenaltyWrapper(gym.Wrapper):
    """같은 액션 반복 시 미세 페널티 (훈련 전용)"""
    def __init__(self, env, penalty=1e-5):
        super().__init__(env)
        self.penalty = penalty
        self._prev_action = None
    
    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._prev_action is not None and int(action) == int(self._prev_action):
            reward = float(reward) - self.penalty
        self._prev_action = int(action)
        return obs, reward, terminated, truncated, info