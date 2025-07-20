import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["LeverageRule", "choose_leverage"]

@dataclass
class LeverageRule:
    """Defines a leverage tier based on account equity."""
    mode: str             # 'cross_margin' | 'isolated'
    min_cap: float        # USD equity lower-bound (inclusive)
    max_cap: float        # USD equity upper-bound (exclusive)
    max_leverage: int     # maximum allowed leverage for this tier

# Leverage tiers configuration
TIERS = [
    LeverageRule('cross_margin',   0,    200,   5),
    LeverageRule('isolated',     200,   2000,  25),
    LeverageRule('isolated',    2000,  float('inf'),  10),
]

def choose_leverage(
    balance: float,
    risk_frac: float,
    stop_usd: float,
    min_notional: float = 5.0,
    exchange_bracket: Optional[dict] = None
) -> tuple[int, float]:
    """
    Calculate adaptive leverage and position USD size.

    Parameters:
    - balance: current account equity in USD
    - risk_frac: fraction of equity to risk (e.g. 0.01 = 1%)
    - stop_usd: stop-loss distance in USD
    - min_notional: minimum trade notional required by exchange
    - exchange_bracket: optional dict with 'maxLeverage' key from exchange API

    Returns:
    - leverage (int)
    - position size in USD (float)
    """
    # target risk in USD
    target_risk = balance * risk_frac

    # compute required leverage so that qty_usd >= min_notional
    lev_needed = (min_notional * stop_usd) / max(1e-9, target_risk)

    # select appropriate tier based on balance
    tier = next(t for t in TIERS if t.min_cap <= balance < t.max_cap)

    # clamp leverage between 1 and tier.max_leverage
    lev = max(1, min(round(lev_needed), tier.max_leverage))

    # further clamp by exchange bracket if given
    if exchange_bracket and 'maxLeverage' in exchange_bracket:
        lev = min(lev, exchange_bracket['maxLeverage'])

    # compute position size in USD
    qty_usd = (target_risk * lev) / stop_usd if stop_usd > 0 else 0

    return lev, qty_usd
