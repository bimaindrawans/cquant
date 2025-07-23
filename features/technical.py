# features/technical.py

import pandas as pd
import ta  # pandas-ta or ta-lib wrapper

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to the DataFrame.
    Expects df to have columns ['h','l','c'].
    """
    df = df.copy()
    df['atr'] = ta.volatility.average_true_range(
        high=df['h'],
        low=df['l'],
        close=df['c'],
        window=period
    )
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) to the DataFrame.
    Expects df to have column 'c' (close).
    """
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(
        close=df['c'],
        window=period
    )
    return df

def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Add Stochastic Oscillator (%K and %D) to the DataFrame.
    Expects df to have columns ['h','l','c'].
    """
    df = df.copy()
    stoch = ta.momentum.StochasticOscillator(
        high=df['h'],
        low=df['l'],
        close=df['c'],
        window=k_period,
        smooth_window=d_period
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    return df

def make_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators in one pass:
      - ATR(14)
      - RSI(14)
      - Stochastic %K(14), %D(3)
    Returns a new DataFrame with added columns.
    """
    df = add_atr(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    return df
