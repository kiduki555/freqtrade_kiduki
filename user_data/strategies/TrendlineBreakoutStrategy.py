# user_data/strategies/TrendlineBreakoutStrategy.py
from typing import Dict, Optional, Tuple
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

class TrendlineBreakoutStrategy(IStrategy):
    """
    TradingView Pine v6 'Trendline Breakout Strategy [KedArc Quant]' 포팅
    - 피벗(HL/LH) 2점으로 추세선(상승: higher lows, 하락: lower highs) 생성 후 우측으로 연장
    - 종가가 하락추세선(downTL) 상향돌파 → 롱 / 종가가 상승추세선(upTL) 하향돌파 → 숏
    - 선택적 MA 필터: 롱 시 close > EMA(maLen), 숏 시 close < EMA(maLen)
    - ATR 기반 스탑: stop = entry ± (atrMult * ATR_entry)
    - ATR×R 기반 익절: tp = entry ± (tpR * atrMult * ATR_entry)
    - 반대 시그널 발생 시 익절(퀵 플립)

    백테스트 시 ROI는 사실상 미사용(커스텀 익절로 관리)
    """
    INTERFACE_VERSION = 3

    timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 480  # 충분히 크게(피벗, EMA, ATR 계산 여유)

    # ROI/SL은 커스텀 로직으로 제어하므로 넉넉히 설정
    minimal_roi = {"0": 100}
    stoploss = -0.99

    # 선물/마진이면 True. 스팟만이면 False로 두고 롱만 사용 권장.
    can_short = False
    use_custom_stoploss = True

    # === Pine inputs ===
    pivLR: int = 5                # Pivot left/right bars
    useMA: bool = True            # MA 필터 사용
    maLen: int = 180              # EMA 길이
    atrLen: int = 14              # ATR 길이
    atrMult: float = 1.5          # 스탑 배수
    tpR: float = 2.0              # 익절 R 배수(= 스탑 거리의 R배)
    enable_longs: bool = True
    enable_shorts: bool = True

    # (선택) 하이퍼옵트용 파라미터 껍데기 – 필요 없으면 제거해도 됩니다.
    piv_lr_hp = IntParameter(2, 10, default=pivLR)
    atr_mult_hp = DecimalParameter(0.5, 4.0, default=atrMult, decimals=1)
    tp_r_hp = DecimalParameter(1.0, 4.0, default=tpR, decimals=1)

    def informative_pairs(self):
        return []

    # --------- 유틸: 피벗 계산(확정 중심 인덱스에 표기) ---------
    @staticmethod
    def _pivots(low: pd.Series, high: pd.Series, lr: int) -> Tuple[pd.Series, pd.Series]:
        n = len(low)
        pl = pd.Series(index=low.index, dtype='float64')
        ph = pd.Series(index=high.index, dtype='float64')

        # 피벗은 중심에서 lr 좌/우 비교가 모두 끝난 시점에 확정됨.
        # 단순/명확성을 위해 루프 사용(속도가 아주 중요치 않을 때 안정적)
        for i in range(lr, n - lr):
            w_low = low.iloc[i - lr: i + lr + 1]
            w_high = high.iloc[i - lr: i + lr + 1]
            center = i
            if low.iloc[center] == w_low.min():
                pl.iloc[center] = low.iloc[center]
            if high.iloc[center] == w_high.max():
                ph.iloc[center] = high.iloc[center]
        return pl, ph

    # --------- 유틸: 두 피벗점으로 만든 추세선 y(x) ---------
    @staticmethod
    def _line_value_at(x1: int, y1: float, x2: int, y2: float, x: int) -> float:
        if x2 == x1:
            return np.nan
        m = (y2 - y1) / (x2 - x1)
        return y1 + m * (x - x1)

    # --------- 유틸: 추세선 가격 시리즈 생성(우측 연장) ---------
    def _build_trendlines(self, df: DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        up_price: 최근 'higher lows' 2점으로 만든 상승추세선의 현재바 가격
        down_price: 최근 'lower highs' 2점으로 만든 하락추세선의 현재바 가격
        """
        idx = np.arange(len(df))
        pl, ph = df['pivot_low'], df['pivot_high']

        # pivot 인덱스 → 가격 매핑
        low_pts = [(i, pl.iat[i]) for i in range(len(pl)) if pd.notna(pl.iat[i])]
        high_pts = [(i, ph.iat[i]) for i in range(len(ph)) if pd.notna(ph.iat[i])]

        up_price = pd.Series(index=df.index, dtype='float64')
        down_price = pd.Series(index=df.index, dtype='float64')

        # 스캔하면서 "마지막 2개 피벗"이 조건(HL/LH)을 만족하면 활성 추세선을 갱신
        hl_last2: list[Tuple[int, float]] = []
        lh_last2: list[Tuple[int, float]] = []

        low_ptr = 0
        high_ptr = 0
        active_up = None   # (x1, y1, x2, y2)
        active_down = None

        for t in range(len(df)):
            # 새 pivot low 확정 도달 시 수집
            while low_ptr < len(low_pts) and low_pts[low_ptr][0] <= t:
                hl_last2.append(low_pts[low_ptr])
                if len(hl_last2) > 2:
                    hl_last2.pop(0)
                # higher low인지 확인 후 활성화
                if len(hl_last2) == 2:
                    (x1, y1), (x2, y2) = hl_last2
                    if y2 > y1 and x2 > x1:
                        active_up = (x1, y1, x2, y2)
                low_ptr += 1

            # 새 pivot high 확정 도달 시 수집
            while high_ptr < len(high_pts) and high_pts[high_ptr][0] <= t:
                lh_last2.append(high_pts[high_ptr])
                if len(lh_last2) > 2:
                    lh_last2.pop(0)
                # lower high인지 확인 후 활성화
                if len(lh_last2) == 2:
                    (x1, y1), (x2, y2) = lh_last2
                    if y2 < y1 and x2 > x1:
                        active_down = (x1, y1, x2, y2)
                high_ptr += 1

            # 현재 바의 추세선 값 계산(우측 연장)
            if active_up is not None:
                x1, y1, x2, y2 = active_up
                up_price.iat[t] = self._line_value_at(x1, y1, x2, y2, t)
            if active_down is not None:
                x1, y1, x2, y2 = active_down
                down_price.iat[t] = self._line_value_at(x1, y1, x2, y2, t)

        return up_price, down_price

    # --------- 지표 생성 ---------
    def populate_indicators(self, df: DataFrame, metadata: Dict) -> DataFrame:
        # EMA, ATR
        df['ema'] = EMAIndicator(close=df['close'], window=self.maLen).ema_indicator()
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=self.atrLen)
        df['atr'] = atr.average_true_range()

        # 피벗
        pl, ph = self._pivots(df['low'], df['high'], self.piv_lr_hp.value if self.piv_lr_hp.value else self.pivLR)
        df['pivot_low'] = pl
        df['pivot_high'] = ph

        # 추세선 (우측 연장가)
        up_price, down_price = self._build_trendlines(df)
        df['up_price'] = up_price
        df['down_price'] = down_price

        # 크로스 신호
        # crossover(close, down_price): 현재 close>down & 이전 close<=이전 down
        dp = df['down_price']
        up = df['up_price']
        df['long_break'] = (
            (dp.notna()) &
            (df['close'] > dp) &
            (df['close'].shift(1) <= dp.shift(1))
        )
        df['short_break'] = (
            (up.notna()) &
            (df['close'] < up) &
            (df['close'].shift(1) >= up.shift(1))
        )

        # MA 필터 + 추가 필터들
        if self.useMA:
            df['long_break'] = df['long_break'] & (df['close'] > df['ema'])
            df['short_break'] = df['short_break'] & (df['close'] < df['ema'])
        
        # 추가 승률 향상 필터들
        # 1. 볼륨 필터: 평균 볼륨 대비 높은 볼륨에서만 진입
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_filter'] = df['volume'] > df['volume_avg'] * 1.2
        
        # 2. ATR 필터: 너무 낮은 변동성에서는 진입 금지
        df['atr_filter'] = df['atr'] > df['atr'].rolling(window=20).mean() * 0.8
        
        # 3. 추세 강도 필터: EMA 기울기 확인
        df['ema_slope'] = df['ema'].diff(5)  # 5바 전 대비 EMA 변화
        df['trend_strength_filter'] = (
            (df['ema_slope'] > 0) & (df['long_break']) |  # 상승 추세에서만 롱
            (df['ema_slope'] < 0) & (df['short_break'])    # 하락 추세에서만 숏
        )
        
        # 모든 필터 적용
        df['long_break'] = df['long_break'] & df['volume_filter'] & df['atr_filter'] & (df['ema_slope'] > 0)
        df['short_break'] = df['short_break'] & df['volume_filter'] & df['atr_filter'] & (df['ema_slope'] < 0)

        # R 계산 편의를 위해 (엔트리 시 ATR 스냅샷 조회용)
        return df

    # --------- 진입 ---------
    def populate_entry_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        df['enter_long'] = 0
        df['enter_short'] = 0

        if self.enable_longs:
            df.loc[df['long_break'] == True, 'enter_long'] = 1

        if self.enable_shorts and self.can_short:
            df.loc[df['short_break'] == True, 'enter_short'] = 1

        return df

    # --------- 표준 익절 신호(주로 반대 시그널 플립용) ---------
    def populate_exit_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        # 기본은 커스텀 TP/SL에 맡기되, 반대 시그널 시 빠른 청산
        df['exit_long'] = 0
        df['exit_short'] = 0

        # 반대 시그널 감지 → 즉시 익절(포지션 클로즈)
        df.loc[df['short_break'] == True, 'exit_long'] = 1
        df.loc[df['long_break'] == True, 'exit_short'] = 1

        return df

    # --------- 커스텀 스탑로스(ATR 기반) ---------
    def custom_stoploss(self, pair: str, trade, current_time: pd.Timestamp, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        entry 시점 ATR을 불러서
        롱: stop = entry - atrMult * ATR_entry
        숏: stop = entry + atrMult * ATR_entry
        → stoploss(%) 로 환산해 반환
        """
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
        except Exception:
            return 1.0  # 보호: 데이터 없으면 넉넉한 SL

        # 엔트리 직전/동일 캔들 찾기
        try:
            # 타입 안전성을 위한 비교
            if trade.open_date_utc in df.index:
                entry_candle = df.loc[:trade.open_date_utc].iloc[-1]
            elif len(df.index) > 0 and pd.to_datetime(trade.open_date_utc) > pd.to_datetime(df.index[0]):
                entry_candle = df.loc[:trade.open_date_utc].iloc[-1]
            else:
                entry_candle = None
        except (TypeError, ValueError, KeyError):
            entry_candle = None
        if entry_candle is None or pd.isna(entry_candle.get('atr')):
            return 1.0

        atr_entry = float(entry_candle['atr'])
        atr_mult = float(self.atr_mult_hp.value) if self.atr_mult_hp.value else self.atrMult

        if trade.is_short:
            stop_price = trade.open_rate + atr_mult * atr_entry
            stoploss_val = (stop_price - trade.open_rate) / trade.open_rate  # 양수
        else:
            stop_price = trade.open_rate - atr_mult * atr_entry
            stoploss_val = (stop_price - trade.open_rate) / trade.open_rate  # 음수

        # freqtrade는 '손절 퍼센트(음수 또는 양수)'를 반환(양수면 무시됨). 우리는 절대값만큼의 SL을 적용하도록 변환
        return abs(stoploss_val)

    # --------- 커스텀 익절(ATR×R 목표 + 반대 시그널 플립) ---------
    def custom_exit(self, pair: str, trade, current_time: pd.Timestamp, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        롱: tp = entry + tpR * atrMult * ATR_entry → current_rate >= tp 이면 익절
        숏: tp = entry - tpR * atrMult * ATR_entry → current_rate <= tp 이면 익절
        + 현재 캔들에 반대 브레이크 발생 시 익절
        """
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
        except Exception:
            return None

        # 현재 캔들 인덱스(가장 가까운 과거 캔들)
        try:
            if current_time in df.index:
                df_now = df.loc[current_time]
            else:
                # 타입 안전성을 위한 슬라이싱
                try:
                    sliced_df = df.loc[:current_time]
                    df_now = sliced_df.iloc[-1] if len(sliced_df) > 0 else None
                except (TypeError, ValueError, KeyError):
                    # 대안: 가장 최근 캔들 사용
                    df_now = df.iloc[-1] if len(df) > 0 else None
        except (TypeError, ValueError, KeyError):
            # 최후의 수단: 가장 최근 캔들 사용
            df_now = df.iloc[-1] if len(df) > 0 else None

        if df_now is None or pd.isna(df_now.get('atr')):
            return None

        # 엔트리 시 ATR
        try:
            # 타입 안전성을 위한 비교
            if trade.open_date_utc in df.index:
                entry_candle = df.loc[:trade.open_date_utc].iloc[-1]
            elif len(df.index) > 0 and pd.to_datetime(trade.open_date_utc) > pd.to_datetime(df.index[0]):
                entry_candle = df.loc[:trade.open_date_utc].iloc[-1]
            else:
                entry_candle = None
        except (TypeError, ValueError, KeyError):
            entry_candle = None
        if entry_candle is None or pd.isna(entry_candle.get('atr')):
            return None

        atr_entry = float(entry_candle['atr'])
        atr_mult = float(self.atr_mult_hp.value) if self.atr_mult_hp.value else self.atrMult
        tp_r = float(self.tp_r_hp.value) if self.tp_r_hp.value else self.tpR

        # 목표가
        if trade.is_short:
            tp_price = trade.open_rate - tp_r * atr_mult * atr_entry
            if current_rate <= tp_price:
                return "tp_r_multiple"
            # 반대 시그널(롱 브레이크) → 익절
            if bool(df_now.get('long_break', False)):
                return "flip_to_long"
        else:
            tp_price = trade.open_rate + tp_r * atr_mult * atr_entry
            if current_rate >= tp_price:
                return "tp_r_multiple"
            # 반대 시그널(숏 브레이크) → 익절
            if bool(df_now.get('short_break', False)):
                return "flip_to_short"

        return None
