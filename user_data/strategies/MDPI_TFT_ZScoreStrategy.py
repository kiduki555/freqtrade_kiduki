# pragma: no cover
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib as ta

class MDPI_TFT_ZScoreStrategy(IStrategy):
    """
    freqtrade strategy implementing the MDPI paper idea:
    - External TFT predicts next-period returns
    - Rolling z-score standardization
    - Trade only when z exceeds threshold, under regime filters
    """
    timeframe = "5m"
    informative_timeframe = "1h"
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    z_window = IntParameter(48, 288, default=96, space="buy")
    z_entry = DecimalParameter(0.4, 2.0, default=0.8, decimals=2, space="buy")
    z_exit = DecimalParameter(0.0, 1.0, default=0.2, decimals=2, space="sell")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, self.informative_timeframe) for p in pairs]

    def merge_external_signals(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Expect a CSV produced by export_freqtrade_signals.py at user_data/signals/mdpi_tft_signals.csv
        with columns: date, asset, pred_ret_24h, z_24h, long_sig, short_sig
        We left-join on nearest previous timestamp to avoid lookahead.
        """
        try:
            sig = self.custom_info.get("mdpi_sig")
            if sig is None:
                path = "user_data/signals/mdpi_tft_signals.csv"
                sig = pd.read_csv(path)
                sig["date"] = pd.to_datetime(sig["date"])
                sig = sig.rename(columns={"asset": "pair"})
                self.custom_info["mdpi_sig"] = sig
            else:
                sig = sig
            pair = metadata["pair"]
            sdf = sig[sig["pair"] == pair].copy()
            if sdf.empty:
                return df
            # Merge-asof to prevent lookahead
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df["index"] if "index" in df.columns else df["open_time"] if "open_time" in df.columns else df.index)
            m = pd.merge_asof(
                df.sort_values("date"),
                sdf.sort_values("date"),
                on="date",
                by=None,
                direction="backward",
                tolerance=pd.Timedelta(self.timeframe)
            )
            return m
        except Exception as e:
            return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        # Regime filters on 1h
        informative = self.dp.get_pair_dataframe(metadata["pair"], self.informative_timeframe)
        informative["ema200"] = ta.EMA(informative["close"], timeperiod=200)
        informative["adx"] = ta.ADX(informative["high"], informative["low"], informative["close"], timeperiod=14)
        informative["atrp"] = ta.ATR(informative["high"], informative["low"], informative["close"], timeperiod=14) / informative["close"] * 100
        informative = informative[["date","ema200","adx","atrp"]]
        df = df.merge(informative, on="date", how="left", suffixes=("", "_inf")).ffill()
        df["bull"] = (df["close"] > df["ema200"]) & (df["adx"] > 20) & (df["atrp"] < 3)

        # External TFT predictions → z-signal
        df = self.merge_external_signals(df, metadata)
        # If no external predictions, strategy will idle
        if "z_24h" not in df.columns:
            df["z_24h"] = np.nan

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, ["enter_long","enter_tag"]] = (0, "")
        cond = (
            (df["bull"] == True) &
            (df["z_24h"] > float(self.z_entry.value))
        )
        df.loc[cond, ["enter_long","enter_tag"]] = (1, "tft_z_buy")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, ["exit_long","exit_tag"]] = (0, "")
        cond = (
            (df["z_24h"] < float(self.z_exit.value)) |
            (ta.RSI(df["close"], timeperiod=14) < 45)
        )
        df.loc[cond, ["exit_long","exit_tag"]] = (1, "tft_z_exit")
        return df
