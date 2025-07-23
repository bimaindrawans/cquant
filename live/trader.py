# live/trader.py

import os
import time
import logging
from datetime import datetime

from config import (
    INTERVAL,
    MAX_POSITION_USDT,
    STATIC_PAIRS,
    DYNAMIC_UNIVERSE_SIZE,
    DYNAMIC_SELECT_K,
    LIGHTGBM_PARAMS
)
from core.scheduler import Scheduler
from core.cache_manager import TempCache
from core.pair_selector import DynamicUCBSelector
from cqio.binance_client import (
    fetch_klines,
    get_account_balance,
    place_market_order
)
from cqio.sentiment import aggregate_sentiment
from features.feature_union import FeatureUnion
from models.policy import PolicyModel
from models.online_update import OnlineUpdater
from live.risk import get_risk_frac, dynamic_sl_tp, compute_position_size

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Components initialization ---
cache       = TempCache(ttl_min=45)   # temporary OHLCV & HMM cache
selector    = DynamicUCBSelector(
    k=DYNAMIC_SELECT_K,
    static_pairs=STATIC_PAIRS,
    universe_size=DYNAMIC_UNIVERSE_SIZE
)
featureer   = FeatureUnion(n_states=3)
policy      = PolicyModel(LIGHTGBM_PARAMS)
updater     = OnlineUpdater(policy)

def interval_to_seconds(interval: str) -> int:
    """Convert Binance interval (e.g. '15m', '4h') into seconds."""
    unit = interval[-1]
    val = int(interval[:-1])
    return {
        'm': val * 60,
        'h': val * 3600,
        'd': val * 86400,
        'w': val * 7 * 86400
    }.get(unit, 60)

def strategy_tick():
    """Single tick: fetch data, select pairs, decide & place orders, retrain policy."""
    try:
        # 1. fetch account balance and dynamic risk fraction
        balance = get_account_balance("USDT")
        risk_frac = get_risk_frac(balance)
        logger.info(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] Balance: {balance:.2f} USDT → risk_frac={risk_frac:.2%}")

        # 2. gather sentiment signals
        sent_df = aggregate_sentiment()
        fg = sent_df['fear_greed'].iloc[-1]
        cp = sent_df['crypto_panic'].iloc[-1]
        logger.info(f" Sentiment → FearGreed: {fg:.2f}, CryptoPanic: {cp:.2f}")

        # 3. select trading symbols (static + dynamic/UCB)
        pairs = selector.choose()
        logger.info(" Trading pairs: " + ", ".join(pairs))

        # 4. loop through each symbol
        for sym in pairs:
            # 4a. load or fetch OHLCV
            df = cache.get_df(sym)
            if df is None:
                df = fetch_klines(sym, INTERVAL, start_str="2 days ago UTC")
                cache.put_df(sym, df)
                # on first-ever fetch, train the HMM on history
                featureer.fit(df)

            # 4b. compute features & regime probabilities
            feat_df = featureer.transform(df)
            last = feat_df.iloc[-1]
            feat_cols  = [c for c in feat_df.columns if not c.startswith("state_")]
            state_cols = [c for c in feat_df.columns if c.startswith("state_")]
            feat_vec   = last[feat_cols].values.reshape(1, -1)
            prob_vec   = last[state_cols].values

            # 4c. decision from policy
            decision = policy.decide(feat_vec, prob_vec, risk_aversion=risk_frac)
            side, size_frac = decision["side"], decision["size"]
            logger.info(f"  {sym}: decision={side}, size_frac={size_frac:.3f}")

            # 4d. if entry signal, compute SL/TP, leverage, qty, and place order
            if side != "flat" and size_frac > 0:
                price = last["c"]
                atr   = last["atr"]

                # calculate stop-loss and take-profit prices
                sl_price, tp_price = dynamic_sl_tp(price, atr)

                # compute adaptive leverage & position sizing
                lev, qty_usd, qty_asset = compute_position_size(
                    balance_usdt=balance,
                    entry_price=price,
                    atr=atr,
                    min_notional=MAX_POSITION_USDT
                )

                # place market order
                order = place_market_order(sym, side, qty_asset)
                logger.info(
                    f"   → {side.upper()} {sym}: qty={qty_asset:.6f}, lev={lev}x, "
                    f"SL≈{sl_price:.2f}, TP≈{tp_price:.2f}, orderId={order.get('orderId')}"
                )

                # (optional) record for online update later:
                # updater.add_observations(pd.DataFrame(feat_vec), pd.Series([label]))

        # 5. retrain policy model if scheduled
        if updater.should_retrain():
            updater.retrain()
            updater.save_model()
            logger.info(" Policy model retrained and saved.")

    except Exception:
        logger.exception("Error during strategy_tick")

def main_loop(paper: bool = True):
    """
    Start the trading scheduler loop.
    - paper=True: no real orders (monkey-patch place_market_order)
    - paper=False: live trading
    """
    if paper:
        logger.info("=== PAPER MODE: no real orders will be placed ===")

    sched = Scheduler()
    interval_sec = interval_to_seconds(INTERVAL)
    sched.add_interval_job(strategy_tick, seconds=interval_sec, job_id="strategy_tick")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        sched.shutdown()

if __name__ == "__main__":
    paper_mode = os.getenv("PAPER", "1") == "1"
    main_loop(paper=paper_mode)
