# backtest/metrics.py

import pandas as pd
import numpy as np

def compute_trade_returns(trades: pd.DataFrame, initial_balance: float = 1.0) -> pd.Series:
    """
    Compute per-trade return series as (balance_after - balance_before) / balance_before.
    Assumes `trades` has a 'balance_after' column and that the first trade's
    balance_before is `initial_balance`.
    """
    # balance_before for each trade
    balance_before = np.concatenate([[initial_balance], trades['balance_after'].iloc[:-1].values])
    returns = (trades['balance_after'].values - balance_before) / balance_before
    return pd.Series(returns, index=trades.index)

def sharpe_ratio(returns: pd.Series, annualize: bool = True) -> float:
    """
    Compute the Sharpe ratio of a return series.
    - returns: pd.Series of per-trade returns (decimal, e.g. 0.02 for +2%)
    - annualize: if True, multiply by sqrt(N) to annualize based on trade frequency
    """
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(len(returns))
    return sr

def max_drawdown(trades: pd.DataFrame, initial_balance: float = 1.0) -> float:
    """
    Compute the maximum drawdown of the equity curve.
    - trades: DataFrame with 'balance_after'
    """
    balances = np.concatenate([[initial_balance], trades['balance_after'].values])
    peak = np.maximum.accumulate(balances)
    drawdowns = (balances - peak) / peak
    return drawdowns.min()

def win_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of trades that were profitable.
    """
    if len(trades) == 0:
        return np.nan
    return (trades['pnl'] > 0).mean()

def expectancy(trades: pd.DataFrame, initial_balance: float = 1.0) -> float:
    """
    Average return per trade (decimal).
    """
    returns = compute_trade_returns(trades, initial_balance)
    return returns.mean() if len(returns) > 0 else np.nan

def summary(trades: pd.DataFrame, initial_balance: float = 1.0) -> dict:
    """
    Produce a dictionary of key backtest metrics:
      - total_trades
      - win_rate
      - expectancy
      - sharpe_ratio
      - max_drawdown
    """
    returns = compute_trade_returns(trades, initial_balance)
    return {
        'total_trades': len(trades),
        'win_rate':            win_rate(trades),
        'expectancy':          expectancy(trades, initial_balance),
        'sharpe_ratio':        sharpe_ratio(returns, annualize=True),
        'max_drawdown':        max_drawdown(trades, initial_balance)
    }
