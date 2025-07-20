# live/risk.py

from config import RISK_THRESHOLDS
from core.leverage import choose_leverage

def get_risk_frac(balance_usdt: float) -> float:
    """
    Return adaptive risk fraction based on current equity tiers defined in RISK_THRESHOLDS.
    RISK_THRESHOLDS is a list of (equity_upper_bound, risk_fraction) tuples.
    """
    for bound, frac in RISK_THRESHOLDS:
        if balance_usdt < bound:
            return frac
    # Fallback, should not happen due to infinite upper bound
    return RISK_THRESHOLDS[-1][1]

def dynamic_sl_tp(
    entry_price: float,
    atr: float,
    sl_multiplier: float = 1.2,
    tp_multiplier: float = 2.4
) -> tuple[float, float]:
    """
    Calculate dynamic stop-loss and take-profit prices based on ATR.

    Parameters:
    -----------
    entry_price : float
        Price at which the trade is entered.
    atr : float
        Average True Range at the current bar.
    sl_multiplier : float
        Multiplier on ATR for stop-loss distance (default 1.2).
    tp_multiplier : float
        Multiplier on ATR for take-profit distance (default 2.4).

    Returns:
    --------
    sl_price : float
    tp_price : float
    """
    sl_price = entry_price - sl_multiplier * atr
    tp_price = entry_price + tp_multiplier * atr
    return sl_price, tp_price

def compute_position_size(
    balance_usdt: float,
    entry_price: float,
    atr: float,
    min_notional: float = 5.0
) -> tuple[int, float, float]:
    """
    Compute adaptive leverage and position size given current equity and market volatility.

    Parameters:
    -----------
    balance_usdt : float
        Current USDT equity.
    entry_price : float
        Price at which the order will be placed.
    atr : float
        ATR at current bar.
    min_notional : float
        Exchange minimum notional size in USD (default 5 USDT).

    Returns:
    --------
    leverage : int
        Chosen leverage multiplier.
    qty_usd : float
        Notional position size in USD.
    qty_base : float
        Position size in base asset units.
    """
    # 1. Determine risk fraction based on current balance
    risk_frac = get_risk_frac(balance_usdt)

    # 2. Compute stop-loss distance (ATR-based)
    stop_dist = 1.2 * atr

    # 3. Use choose_leverage to satisfy min_notional and risk fraction
    leverage, qty_usd = choose_leverage(
        balance_usdt,
        risk_frac,
        stop_dist,
        min_notional=min_notional
    )

    # 4. Convert USD notional to base asset quantity
    qty_base = qty_usd / entry_price

    return leverage, qty_usd, qty_base
