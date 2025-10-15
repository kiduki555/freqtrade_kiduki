# user_data/TFT_PPO_Training/scripts/action_filter.py
import numpy as np

class ActionFilter:
    """
    액션 붕괴 방지를 위한 필터
    - 인퍼런스 마스킹: 빈 포지션에서 Sell 금지
    - 히스테리시스: 같은 액션이 N캔들 연속일 때만 전이 인정
    - 쿨다운: 진입/청산 후 M캔들 동안 재전이 금지
    """
    def __init__(self, hysteresis=2, cooldown=5):
        self.last_a = 0
        self.rep = 0
        self.cd = 0
        self.pos = 0
        self.hysteresis = hysteresis
        self.cooldown = cooldown

    def step(self, a):
        """
        액션 필터링 적용
        Args:
            a: 원본 액션 (0=Hold, 1=Buy, 2=Sell)
        Returns:
            필터링된 액션
        """
        # 쿨다운 중이면 Hold 강제
        if self.cd > 0:
            self.cd -= 1
            a = 0
        
        # 빈 포지션에서 Sell 금지
        if self.pos == 0 and a == 2:
            a = 0
        
        # 히스테리시스: 같은 액션이 연속되지 않으면 이전 액션 유지
        self.rep = self.rep + 1 if a == self.last_a else 1
        if self.rep < self.hysteresis:
            a = self.last_a

        # 전이 시 포지션/쿨다운 갱신
        if self.last_a != a:
            if a == 1 and self.pos == 0:  # 진입
                self.pos = 1
                self.cd = self.cooldown
            elif a != 1 and self.pos == 1:  # 청산
                self.pos = 0
                self.cd = self.cooldown
        
        self.last_a = a
        return a

    def reset(self):
        """필터 상태 초기화"""
        self.last_a = 0
        self.rep = 0
        self.cd = 0
        self.pos = 0
