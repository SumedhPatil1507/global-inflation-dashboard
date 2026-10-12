"""Release-time-safe MIDAS nowcasting from daily markets to monthly CPI.

Daily inputs are reduced with normalized exponential Almon weights over the
trading days available strictly before each CPI release timestamp. The model
fits monthly CPI on these mixed-frequency predictors, then can nowcast using
daily data available as of an explicit cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class MIDASNowcastResult:
    target_period: str
    as_of: str
    nowcast: float
    predictors: dict[str, float]
    training_observations: int


class MIDASNowcaster:
    """Linear MIDAS regression with an estimated exponential lag-decay kernel."""

    def __init__(self, lookback_days: int = 60, alpha: float = 1.0):
        if lookback_days < 2:
            raise ValueError("lookback_days must be at least 2")
        self.lookback_days = lookback_days
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.columns: list[str] = []
        self.decay = 0.05
        self.training_observations = 0
        self._fitted = False

    def _weighted_predictors(self, daily: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
        available = daily.loc[daily.index < cutoff, self.columns].tail(self.lookback_days)
        if available.empty:
            return np.full(len(self.columns), np.nan)
        values = available.to_numpy(dtype=float)
        # Daily rows are chronological; the latest observation receives the largest weight.
        lag = np.arange(len(values) - 1, -1, -1, dtype=float)
        weights = np.exp(-self.decay * lag)
        weighted = np.where(np.isfinite(values), values, 0.0) * weights[:, None]
        weight_sum = (np.isfinite(values) * weights[:, None]).sum(axis=0)
        result = weighted.sum(axis=0) / np.where(weight_sum > 0, weight_sum, 1.0)
        result[weight_sum == 0] = np.nan
        return result

    @staticmethod
    def _utc_timestamp(value: object) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def fit(self, daily: pd.DataFrame, monthly_cpi: pd.Series,
            release_dates: pd.Series) -> "MIDASNowcaster":
        """Fit using CPI reference periods and their actual public release timestamps.

        `monthly_cpi` is indexed by reference period; `release_dates` maps those
        same period labels to publication instants. Predictors at/after a release
        are excluded from that training observation.
        """
        if daily.empty or monthly_cpi.empty:
            raise ValueError("daily predictors and monthly CPI are required")
        daily = daily.copy()
        daily.index = pd.to_datetime(daily.index, utc=True)
        daily = daily.sort_index()
        self.columns = list(daily.select_dtypes(include=[np.number]).columns)
        if not self.columns:
            raise ValueError("daily must contain numeric predictors")
        cpi = monthly_cpi.copy()
        dates = release_dates.copy()
        cpi.index = cpi.index.map(str)
        dates.index = dates.index.map(str)
        rows, targets = [], []
        for period, target in cpi.items():
            if period not in dates.index or pd.isna(target):
                continue
            cutoff = self._utc_timestamp(dates.loc[period])
            rows.append((period, cutoff))
            targets.append(float(target))
        if len(rows) < 3:
            raise ValueError("at least three CPI observations with release dates are required")

        # Choose the decay that gives the best in-sample fit, then estimate the
        # regression on those weighted mixed-frequency observations.
        def loss(decay: float) -> float:
            self.decay = float(decay)
            x = np.vstack([self._weighted_predictors(daily, cutoff) for _, cutoff in rows])
            valid = np.isfinite(x).all(axis=1)
            if valid.sum() < 3:
                return float("inf")
            z = StandardScaler().fit_transform(x[valid])
            coef = np.linalg.lstsq(np.column_stack([np.ones(valid.sum()), z]), np.asarray(targets)[valid], rcond=None)[0]
            residual = np.asarray(targets)[valid] - np.column_stack([np.ones(valid.sum()), z]) @ coef
            return float(np.mean(residual ** 2))

        optimum = minimize_scalar(loss, bounds=(0.001, 0.5), method="bounded")
        self.decay = float(optimum.x)
        x = np.vstack([self._weighted_predictors(daily, cutoff) for _, cutoff in rows])
        y = np.asarray(targets, dtype=float)
        valid = np.isfinite(x).all(axis=1)
        if valid.sum() < 3:
            raise ValueError("not enough daily history before CPI releases")
        self.model.fit(self.scaler.fit_transform(x[valid]), y[valid])
        self.training_observations = int(valid.sum())
        self._fitted = True
        return self

    def nowcast(self, daily: pd.DataFrame, *, target_period: str,
                as_of: object) -> MIDASNowcastResult:
        if not self._fitted:
            raise RuntimeError("fit the MIDASNowcaster before requesting a nowcast")
        cutoff = self._utc_timestamp(as_of)
        data = daily.copy()
        data.index = pd.to_datetime(data.index, utc=True)
        predictors = self._weighted_predictors(data.sort_index(), cutoff)
        if not np.isfinite(predictors).all():
            raise ValueError("daily predictors are unavailable before the as_of cutoff")
        prediction = float(self.model.predict(self.scaler.transform(predictors.reshape(1, -1)))[0])
        return MIDASNowcastResult(
            target_period=str(target_period), as_of=cutoff.isoformat(), nowcast=prediction,
            predictors=dict(zip(self.columns, map(float, predictors))),
            training_observations=self.training_observations,
        )
