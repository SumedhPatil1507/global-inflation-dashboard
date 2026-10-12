"""Execution friction and institutional risk metrics for portfolio backtests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionCostModel:
    """Cost rates are decimals; ADV and order notional use the same currency."""

    half_spread_bps: float = 2.0
    base_slippage_bps: float = 1.0
    impact_eta: float = 0.1
    annualization: int = 252

    def cost_rate(self, turnover: float, *, volatility: float = 0.0,
                  order_notional: float = 0.0, adv: float | None = None,
                  slippage_bps: float | None = None) -> float:
        """Return fractional cost on traded notional.

        Slippage is allowed to vary per fill; Almgren-Chriss temporary impact
        scales with volatility and the square root of participation rate.
        """
        turnover = max(0.0, float(turnover))
        slip = self.base_slippage_bps if slippage_bps is None else max(0.0, slippage_bps)
        spread = max(0.0, self.half_spread_bps)
        participation = order_notional / max(float(adv or order_notional or 1.0), 1e-12)
        ac_impact = max(0.0, self.impact_eta) * max(0.0, volatility) * np.sqrt(max(0.0, participation))
        return turnover * ((spread + slip) / 10_000.0 + ac_impact)


def performance_metrics(equity: pd.Series, *, periods_per_year: int = 252,
                        risk_free_rate: float = 0.0) -> dict[str, float]:
    """Annualized return/risk metrics and drawdown duration (in observations)."""
    equity = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        raise ValueError("equity curve cannot be empty")
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    sharpe = float(excess.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0
    sortino = float(excess.mean() / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else 0.0
    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    max_dd = float(-drawdown.min())
    n = len(equity)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / max(n - 1, 1)) - 1) if equity.iloc[0] > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0
    # Historical expected shortfall: mean loss in the worst 5% daily returns.
    losses = -returns.to_numpy()
    var95 = float(np.quantile(losses, 0.95)) if len(losses) else 0.0
    cvar95 = float(losses[losses >= var95].mean()) if len(losses) else 0.0
    underwater = drawdown.to_numpy() < 0
    max_duration = current_duration = 0
    for is_underwater in underwater:
        current_duration = current_duration + 1 if is_underwater else 0
        max_duration = max(max_duration, current_duration)
    return {
        "sharpe": float(sharpe), "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar), "expected_shortfall_95": float(cvar95),
        "max_drawdown": max_dd, "max_drawdown_duration": int(max_duration),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if n > 1 else 0.0,
        "annualized_volatility": vol * float(np.sqrt(periods_per_year)),
    }


def rebalance_portfolio(asset_returns: pd.DataFrame, weights: Mapping[str, float],
                        *, starting_capital: float = 10_000.0,
                        cost_model: ExecutionCostModel | None = None,
                        asset_volatility: Mapping[str, float] | None = None,
                        average_daily_volume: Mapping[str, float] | None = None,
                        daily_slippage_bps: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Daily target-weight rebalance with spread, variable slippage and AC impact.

    Returns equity and the per-period transaction cost as a fraction of equity.
    Target weights are normalized to one; all input returns must be point-in-time.
    """
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty")
    cols = list(weights)
    missing = set(cols) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"missing returns for assets: {sorted(missing)}")
    target = np.asarray([float(weights[c]) for c in cols], dtype=float)
    if not np.isfinite(target).all() or np.isclose(target.sum(), 0):
        raise ValueError("weights must be finite and sum to a non-zero value")
    target /= target.sum()
    model = cost_model or ExecutionCostModel()
    returns = asset_returns[cols].fillna(0.0).astype(float)
    equity_values = [float(starting_capital)]
    costs = []
    drifted = np.zeros(len(cols), dtype=float)
    for dt, row in returns.iterrows():
        pretrade = np.zeros(len(cols), dtype=float) if len(costs) == 0 else drifted
        turnover_by_asset = np.abs(target - pretrade)
        cost = 0.0
        for i, ticker in enumerate(cols):
            slip = None
            if daily_slippage_bps is not None and ticker in daily_slippage_bps.columns and dt in daily_slippage_bps.index:
                slip = float(daily_slippage_bps.at[dt, ticker])
            notional = equity_values[-1] * turnover_by_asset[i]
            rate = model.cost_rate(
                turnover_by_asset[i], volatility=float((asset_volatility or {}).get(ticker, 0.0)),
                order_notional=notional, adv=(average_daily_volume or {}).get(ticker), slippage_bps=slip,
            )
            cost += rate
        gross = float(np.dot(target, row.to_numpy()))
        period_return = gross - cost
        costs.append(cost)
        equity_values.append(equity_values[-1] * (1.0 + period_return))
        post_return_values = target * (1.0 + row.to_numpy())
        denominator = post_return_values.sum()
        drifted = post_return_values / denominator if denominator else target.copy()
    index = returns.index
    return pd.Series(equity_values[1:], index=index, name="equity"), pd.Series(costs, index=index, name="transaction_cost")
