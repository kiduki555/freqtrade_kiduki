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
        sanity_mode: bool = False,        # sanity 모드 플래그 (기본값: False)
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
        self.equity_peak = 1.0      # 드로우다운 계산용 최고점
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
        self.hold_streak = 0                # 연속 hold 카운트 (무거래 방지)
        self.position_changes = 0           # 포지션 변경 횟수 (다양성 보너스용)
        
        # === sanity 모드 (디버깅용) - 먼저 설정 ===
        self.sanity_mode = sanity_mode
        
        # === 액션 필터 (과매매 방지 강화) ===
        # 모든 모드에서 필터 비활성화 (거래 발생 보장)
        self.action_filter = None  # 필터 완전 비활성화

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
        
        # 옵션 1: TFT 학습된 표현 사용 (강력한 방법)
        if hasattr(self, 'use_tft_encoding') and self.use_tft_encoding:
            try:
                # TFT 입력 형태로 데이터 준비
                x = torch.tensor(window_df.values, dtype=torch.float32).unsqueeze(0)  # (1, window, features)
                
                with torch.no_grad():
                    # MultiTaskTFT의 forward 메서드 사용
                    tft_input = {
                        "encoder_cont": x,
                        "encoder_lengths": torch.tensor([self.window], dtype=torch.long),
                        "decoder_lengths": torch.tensor([1], dtype=torch.long),
                    }
                    
                    # TFT 예측 수행
                    predictions = self.tft(tft_input)
                    
                    # encoder representation 추출
                    if "encoder_repr" in predictions:
                        encoded = predictions["encoder_repr"]
                    else:
                        # fallback: 수익률 예측값 사용
                        encoded = predictions["returns"]["horizon_24"]
                    
                    # 64차원으로 맞추기
                    encoded_np = encoded.squeeze().cpu().numpy().astype(np.float32)
                    if len(encoded_np) < 64:
                        # 패딩
                        padded = np.pad(encoded_np, (0, 64 - len(encoded_np)), 'constant')
                        return padded
                    elif len(encoded_np) > 64:
                        # 자르기
                        return encoded_np[:64]
                    else:
                        return encoded_np
                        
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
    
    def _drawdown(self, peak, current):
        """드로우다운 계산"""
        if peak <= 0:
            return 0.0
        return (peak - current) / peak

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

        # 액션 필터링 적용 (sanity 모드에서는 필터 비활성화)
        if self.action_filter is not None:
            filtered_action = self.action_filter.step(int(action))
        else:
            filtered_action = int(action)  # 필터 없이 그대로 사용
        
        # 액션→목표 포지션 매핑
        target_pos = {0: 0, 1: +1, 2: -1}[filtered_action]

        # === 새로운 보상 리셰이핑 ===
        # 1) PnL 델타 계산
        pnl_delta = float(self.position) * float(logret)
        
        # 2) 수수료: 항상 양수 비용으로 누적, 플립(±1↔∓1) 시 2번 청구
        fee_cost = 0.0
        position_changed = (target_pos != self.position)
        if position_changed:
            if self.position != 0:   # 기존 포지션 청산
                fee_cost += self.fee_rate
            if target_pos != 0:      # 새 포지션 진입
                fee_cost += self.fee_rate
        
        # sanity 모드에서 수수료 완화
        if getattr(self, "sanity_mode", False):
            fee_cost *= 0.1  # 수수료를 1/10로 완화

        # 포지션 적용
        if position_changed:
            self.position = target_pos
            self.entry_price = curr_price if self.position != 0 else None
            self.position_changes += 1
        
        # 3) 무거래 방지 로직
        if target_pos == 0:  # hold 액션
            self.hold_streak += 1
        else:
            self.hold_streak = 0
        
        # 4) 거래 인센티브 및 포지션 다양성 보너스
        trading_bonus = 0.0
        if position_changed:
            trading_bonus = 0.0001  # 소량의 거래 인센티브
        
        # 5) 연속 hold 페널티 (무거래 방지)
        hold_penalty = 0.0
        if self.hold_streak > 20:  # 20스텝 이상 연속 hold
            hold_penalty = 0.00005 * (self.hold_streak - 20)  # 점진적 페널티
        
        # 6) 과매매 방지 (거래 빈도 제한) + 턴오버 페널티
        overtrading_penalty = 0.0
        turnover_penalty = 0.0
        
        if position_changed:
            # 최근 10스텝 내 거래 횟수 체크
            recent_trades = getattr(self, 'recent_trades', [])
            recent_trades.append(1)  # 거래 발생
            if len(recent_trades) > 10:
                recent_trades.pop(0)  # 오래된 기록 제거
            self.recent_trades = recent_trades
            
            # 10스텝 내 5회 이상 거래 시 페널티
            if len(recent_trades) >= 5 and sum(recent_trades) >= 5:
                overtrading_penalty = 0.001  # 과매매 페널티
            
            # 턴오버 페널티 (거래 비용 추가)
            turnover_penalty = 0.0005  # 거래당 추가 비용
            
            # sanity 모드에서 페널티 완화
            if self.sanity_mode:
                overtrading_penalty *= 0.2
                turnover_penalty *= 0.2
        
        # 6) 드로우다운/에쿼티 업데이트 (수수료는 반드시 빼기)
        dd_before = self._drawdown(self.equity_peak, self.equity)
        equity_next = self.equity * np.exp(pnl_delta - fee_cost)
        self.equity = equity_next
        self.equity_peak = max(self.equity_peak, self.equity)
        dd_after = self._drawdown(self.equity_peak, self.equity)

        # 7) 보상 계산 (수수료는 마이너스)
        reward = (
            pnl_delta
            - fee_cost
            - 0.1 * max(0.0, dd_after - dd_before)
            + trading_bonus
            - hold_penalty
            - overtrading_penalty
            - turnover_penalty
        )

        # ✅ 순이익(pnl_delta - fee_cost) > 0 이면 최소 양수 보상 보장
        if (pnl_delta - fee_cost) > 0:
            if getattr(self, "sanity_mode", False):
                reward = max(reward, 0.01)  # sanity 모드: 더 강한 플로어
            else:
                reward = max(reward, 0.001)  # 정상 모드: 기본 플로어

        reward = float(np.clip(reward, -0.05, 0.05))
        
        # 8) 에피소드 보상 추적
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)
        
        # 9) 에피소드 종료 시 Sharpe 보너스 + 거래 다양성 보너스
        done = self.current_step >= len(self.df) - 2
        if done and len(self.episode_rewards) > 10:
            r = np.asarray(self.episode_rewards, dtype=np.float64)
            sharpe = float(np.mean(r) / (np.std(r) + 1e-8)) * np.sqrt(24*365)  # 1h이면 24*365
            reward += 0.01 * np.tanh(sharpe)
            
            # 거래 다양성 보너스 (포지션 변경 횟수 기반)
            if self.position_changes > 5:  # 최소 5회 이상 거래
                diversity_bonus = 0.005 * min(self.position_changes / 20, 1.0)  # 최대 0.005
                reward += diversity_bonus
        
        # 8) 거래 플래그는 '포지션 변경 여부'로!
        trade_flag = int(position_changed)
        self.prev_action = target_pos

        self.current_step += 1
        if self.current_step >= len(self.df) - 2:
            terminated = True

        obs = self._encode_state()
        info = {
            "pnl_step": float(self.position) * float(logret) - fee_cost,
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
            
            # 시드가 있으면 약간의 랜덤성 추가 (최대 100 스텝)
            if seed is not None:
                np.random.seed(seed)
                random_offset = np.random.randint(0, min(100, max_offset - offset))
                offset += random_offset
                offset = min(offset, max_offset)
        
        self.current_step = offset
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        self.equity_peak = 1.0
        self.prev_action = 0
        self.vol_ewm = 0.0
        self.streak = 0
        self.last_action = 0
        self.hold_streak = 0
        self.position_changes = 0
        self.recent_trades = []  # 과매매 방지용 거래 기록 초기화
        # 액션 필터 리셋 (필터가 있는 경우만)
        if self.action_filter is not None:
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
