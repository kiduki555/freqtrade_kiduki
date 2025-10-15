import gymnasium as gym
import numpy as np
import torch
from TFT_PPO_modules.reward_function import custom_reward
from TFT_PPO_Training.scripts.action_filter import ActionFilter


class TradingEnv(gym.Env):
    """
    Position-aware Trading Environment
    ---------------------------------
    Reinforcement Learning environment for algorithmic trading with proper position management.
    
    Key Features:
      - Position state management (-1, 0, +1)
      - Entry price tracking and equity calculation
      - Transaction costs and slippage on position changes
      - PnL-based reward system with optional custom reward
      - Proper mark-to-market accounting
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        tft_model,
        features,
        window: int = 72,
        fee_bps: float = 10,              # 1bp = 0.01% → 10bps = 0.10%
        slippage_bps: float = 5,          # 체결 슬리피지
        reward_mode: str = "pnl",         # "pnl" | "custom"
        reward_func=custom_reward,
    ):
        super().__init__()

        # === Core dataset & model ===
        self.df = df.reset_index(drop=True)
        self.tft = tft_model
        self.features = features
        self.window = window
        self.fee = fee_bps / 1e4
        self.slip = slippage_bps / 1e4
        self.reward_mode = reward_mode
        self.reward_func = reward_func
        self.device = torch.device("cpu")  # MLP 정책은 CPU 권장
        
        # 전체 길이 캐시
        self.length = int(len(self.df)) if hasattr(self.df, "__len__") else None

        # === Environment state ===
        self.current_step = window
        self.position = 0           # -1, 0, +1
        self.entry_price = None
        self.equity = 1.0
        self.prev_action = 0
        
        # === KPI 정렬 보상 파라미터 ===
        self.fee_rate = fee_bps / 1e4 / 2.0  # 편측 수수료 (진입/청산 각각)
        self.dd_coef = 0.1                   # 드로우다운 페널티 계수
        self.switch_coef = 0.005             # 포지션 변경 페널티 계수
        self.peak_equity = 1.0               # 최고 에쿼티 추적
        
        # === 항상-롱 붕괴 방지 파라미터 ===
        self.vol_floor = 5e-4               # 변동성 하한선
        self.flat_coef = 0.1                # 정체 페널티 계수
        self.streak_hyst = 8                # 연속 액션 허용 스텝
        self.streak_coef = 1e-3             # 연속 액션 페널티 계수
        
        # === 상태 추적 ===
        self.vol_ewm = 0.0                  # 지수 가중 변동성
        self.streak = 0                     # 연속 액션 카운트
        self.last_action = 0                # 이전 액션
        
        # === 액션 필터 (개선) ===
        self.action_filter = ActionFilter(hysteresis=2, cooldown=5)

        # === Spaces (Gymnasium style) ===
        self.action_space = gym.spaces.Discrete(3)  # 0=Flat, 1=Long, 2=Short
        
        # TFT 임베딩 크기 가정: 64 (실제 모델에 맞게 조정)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )

    # -----------------------------
    # State extraction
    # -----------------------------
    def _encode_state(self):
        """상태 인코딩 - TFT 학습된 표현 활용 (옵션)"""
        window_df = self.df[self.features].iloc[self.current_step - self.window : self.current_step]
        
        # 옵션 1: TFT 학습된 표현 사용 (복잡하지만 강력)
        if hasattr(self, 'use_tft_encoding') and self.use_tft_encoding:
            try:
                x = torch.tensor(window_df.values, dtype=torch.float32).unsqueeze(0)
                # TFT의 encoder 부분만 사용 (간단한 방법)
                with torch.no_grad():
                    # TFT의 encoder 부분만 추출 (전체 모델 대신)
                    if hasattr(self.tft, 'encoder'):
                        encoded = self.tft.encoder(x)
                        return encoded.squeeze().cpu().numpy().astype(np.float32)
                    else:
                        # TFT 전체 모델 사용 시 필요한 입력 구성
                        tft_input = {
                            "encoder_cont": x,
                            "encoder_lengths": torch.tensor([self.window], dtype=torch.long),
                            "decoder_lengths": torch.tensor([1], dtype=torch.long),
                        }
                        state = self.tft(tft_input)
                        return state.squeeze().cpu().numpy().astype(np.float32)
            except Exception as e:
                print(f"[Warning] TFT encoding failed: {e}, falling back to simple encoding")
        
        # 옵션 2: 간단한 특성 추출 (기본값 - 안정적이고 빠름)
        # --- 안전 슬라이스: 인덱스 보정 ---
        start = max(0, self.current_step - self.window)
        end = max(start + 1, self.current_step)
        
        try:
            window_df = self.df[self.features].iloc[start:end]
            x = window_df.values.astype(np.float32).reshape(-1)
        except Exception:
            # 슬라이스 실패 시 기본값
            x = np.zeros((self.window * len(self.features),), dtype=np.float32)
            window_df = self.df[self.features].iloc[:1]  # 최소 1행
        
        # 기술적 지표 기반 특성 추출 (안전하게)
        try:
            recent_values = window_df.iloc[-5:].values.flatten()
        except Exception:
            recent_values = np.zeros((5 * len(self.features),), dtype=np.float32)
        
        # 빈 배열 방어 코드
        def safe_max_min_diff(arr):
            if len(arr) == 0:
                return 0.0
            return float(np.max(arr) - np.min(arr))
        
        def safe_mean(arr):
            if len(arr) == 0:
                return 0.0
            return float(np.mean(arr))
        
        def safe_std(arr):
            if len(arr) == 0:
                return 0.0
            return float(np.std(arr))
        
        stats_features = np.array([
            safe_mean(x),                    # 전체 평균
            safe_std(x),                     # 전체 변동성
            safe_mean(recent_values),        # 최근 평균
            safe_std(recent_values),         # 최근 변동성
            safe_max_min_diff(x),            # 전체 범위
            safe_max_min_diff(recent_values),  # 최근 범위
        ], dtype=np.float32)
        
        # 기본 특성과 통계 특성 결합
        combined_state = np.concatenate([x, stats_features])
        
        # 상태 크기를 64로 맞추기
        if len(combined_state) < 64:
            padded_state = np.pad(combined_state, (0, 64 - len(combined_state)), 'constant')
        else:
            padded_state = combined_state[:64]
            
        return padded_state

    def _get_prices(self):
        """현재 및 다음 가격 반환"""
        curr = float(self.df["close"].iloc[self.current_step])
        nxt = float(self.df["close"].iloc[self.current_step + 1])
        return curr, nxt

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, action):
        terminated = False
        truncated = False

        if self.current_step >= len(self.df) - 2:
            truncated = True
            return self._encode_state(), 0.0, terminated, truncated, {}

        curr_price, next_price = self._get_prices()
        
        # 로그 수익률 계산
        logret = np.log((next_price + 1e-12) / (curr_price + 1e-12))

        # 액션 필터링 적용
        filtered_action = self.action_filter.step(int(action))
        
        # 액션→목표 포지션 매핑
        target_pos = {0: 0, 1: +1, 2: -1}[filtered_action]

        # === KPI 정렬 보상 계산 ===
        # 1) 기본 수익률 보상
        reward_return = float(self.position) * float(logret)
        
        # 2) 수수료 (진입/청산 시에만)
        fee = 0.0
        position_changed_to_open = False
        position_changed_to_close = False
        
        if target_pos != self.position:
            if target_pos != 0 and self.position == 0:  # 진입
                fee -= self.fee_rate
                position_changed_to_open = True
            elif target_pos == 0 and self.position != 0:  # 청산
                fee -= self.fee_rate
                position_changed_to_close = True
            
            self.position = target_pos
            self.entry_price = curr_price if self.position != 0 else None

        # 3) 에쿼티 업데이트 및 드로우다운 페널티
        self.equity *= np.exp(reward_return + fee)
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        pen_dd = -self.dd_coef * dd

        # 4) 변동성/정체 페널티: 수익 없는데 변동만 큰 구간에서 포지션 유지 페널티
        self.vol_ewm = 0.9 * self.vol_ewm + 0.1 * abs(logret)  # 지수 가중 이동평균
        flat_pen = -self.flat_coef * self.position * max(0.0, self.vol_ewm - self.vol_floor)

        # 5) 과매매/행동단조 페널티(같은 액션 연속 K스텝 초과 시)
        self.streak = self.streak + 1 if int(action) == self.last_action else 1
        pen_streak = -self.streak_coef * max(0, self.streak - self.streak_hyst)
        self.last_action = int(action)

        # 6) 포지션 변경 페널티 (과매매 방지)
        position_changed = int(target_pos != self.position)
        pen_switch = -self.switch_coef * position_changed

        # 7) 최종 보상
        reward = reward_return + fee + pen_dd + flat_pen + pen_streak + pen_switch
        
        # 디버그용 변수들
        trade_cost = fee
        trade_flag = int(position_changed)

        self.current_step += 1
        if self.current_step >= len(self.df) - 2:
            terminated = True

        obs = self._encode_state()
        info = {
            "pnl_step": float(self.position) * float(logret) - trade_cost,
            "equity": float(self.equity),
            "position": int(self.position),
            "trade": int(trade_flag),
        }
        return obs, float(reward), terminated, truncated, info

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- 안전한 초기화: offset이 범위를 넘지 않도록 ---
        offset = self.window  # 기본값
        if self.length is None:
            self.length = int(len(self.df)) if hasattr(self.df, "__len__") else 0
        
        # 최소로 필요한 길이: window + 2 (관측+한 스텝)
        required = max(self.window + 2, 4)
        if self.length <= required:
            # 데이터 자체가 너무 짧으면 window부터 시작
            offset = self.window
        else:
            # 남은 길이가 최소 요구 길이보다 작지 않게 오프셋 클램프
            max_offset = max(self.window, self.length - required)
            offset = int(max(self.window, min(offset, max_offset)))
        
        self.current_step = offset
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        self.peak_equity = 1.0
        self.prev_action = 0
        self.vol_ewm = 0.0
        self.streak = 0
        self.last_action = 0
        self.action_filter.reset()
        obs = self._encode_state()
        info = {}
        return obs, info

    # -----------------------------
    # Render
    # -----------------------------
    def render(self):
        """Optional visualization hook for backtesting or live tracking."""
        step_data = self.df.iloc[self.current_step]
        print(
            f"Step {self.current_step}: Price={step_data['close']:.2f}, Position={self.position}, "
            f"Drawdown={step_data['rolling_drawdown_48']:.4f}"
        )
