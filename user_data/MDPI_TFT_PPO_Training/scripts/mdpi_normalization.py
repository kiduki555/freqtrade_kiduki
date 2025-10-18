# user_data/TFT_PPO_Training/scripts/mdpi_normalization.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

BOUNDED_01_HINTS = {"rsi", "stoch", "kurtosis_prob", "sigmoid_", "prob_", "p_","_ratio","bandwidth_norm"}
POSITIVE_SKEW_HINTS = {"volume","tvl","oi","open_interest","fees","tx_count","active_address","gas","fee","netflow_abs"}

@dataclass
class MDPIStandardizer:
    """
    Implements MDPI-style normalization: z / log / logit per feature group.
    - Fit on TRAIN only, then apply to TRAIN/VAL/TEST to avoid leakage.
    - Asset-agnostic by default; pass group_cols=['asset'] to fit per asset.
    """
    group_cols: List[str] = field(default_factory=list)
    stats_: Dict[Tuple, Dict[str, Tuple[float,float]]] = field(default_factory=dict)

    def _is_bounded01(self, name: str) -> bool:
        n = name.lower()
        return any(h in n for h in BOUNDED_01_HINTS)

    def _is_positive_skew(self, name: str) -> bool:
        n = name.lower()
        return any(h in n for h in POSITIVE_SKEW_HINTS)

    def _logit(self, x: pd.Series, eps=1e-6) -> pd.Series:
        x = x.clip(eps, 1-eps)
        return np.log(x/(1-x))

    def _zscore(self, x: pd.Series, mean: float=None, std: float=None) -> Tuple[pd.Series, float, float]:
        if mean is None: mean = x.mean()
        if std is None: std = x.std(ddof=0)
        std = std if std not in (0, None) else 1.0
        return (x - mean) / std, mean, std

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "MDPIStandardizer":
        self.stats_.clear()
        if self.group_cols:
            for key, g in df.groupby(self.group_cols):
                self.stats_[key] = {}
                for c in feature_cols:
                    s = g[c].astype(float)
                    if self._is_bounded01(c):
                        s = self._logit(s.clip(0,1))
                    elif self._is_positive_skew(c):
                        s = np.log1p(s.clip(lower=0))
                    _, m, sd = self._zscore(s)
                    self.stats_[key][c] = (m, sd)
        else:
            self.stats_[()] = {}
            for c in feature_cols:
                s = df[c].astype(float)
                if self._is_bounded01(c):
                    s = self._logit(s.clip(0,1))
                elif self._is_positive_skew(c):
                    s = np.log1p(s.clip(lower=0))
                _, m, sd = self._zscore(s)
                self.stats_[()][c] = (m, sd)
        return self

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        if self.group_cols:
            out = []
            for key, g in df.groupby(self.group_cols):
                stats = self.stats_.get(key, {})
                for c in feature_cols:
                    s = g[c].astype(float)
                    if self._is_bounded01(c):
                        s = self._logit(s.clip(0,1))
                    elif self._is_positive_skew(c):
                        s = np.log1p(s.clip(lower=0))
                    m, sd = stats.get(c, (s.mean(), s.std(ddof=0) or 1.0))
                    g[c] = (s - m) / (sd if sd else 1.0)
                out.append(g)
            return pd.concat(out, ignore_index=False).sort_index()
        else:
            stats = self.stats_[()]
            for c in feature_cols:
                s = df[c].astype(float)
                if self._is_bounded01(c):
                    s = self._logit(s.clip(0,1))
                elif self._is_positive_skew(c):
                    s = np.log1p(s.clip(lower=0))
                m, sd = stats.get(c, (s.mean(), s.std(ddof=0) or 1.0))
                df[c] = (s - m) / (sd if sd else 1.0)
            return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df, feature_cols)
