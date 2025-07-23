# backtest/backtester.py

import pandas as pd
import numpy as np

class VectorBacktester:
    """
    A simple vectorized backtester for intra-day/swing strategies.
    Simulates market orders with fixed SL/TP based on ATR and a decision policy.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        state_cols: list[str],
        policy,
        sl_atr: float = 1.2,
        tp_atr: float = 2.4,
        initial_balance: float = 1.0
    ):
        """
        Parameters:
        -----------
        df : DataFrame
            Must contain columns:
              - 'o','h','l','c','v','atr', plus any technical features
              - one column per HMM state probability (e.g. 'state_0', 'state_1', ...)
        feature_cols : list of str
            Names of the feature columns passed to policy.decide (excluding state_cols).
        state_cols : list of str
            Names of the HMM state-probability columns.
        policy : object with decide(feat_row, prob_state) -> dict(side, size)
        sl_atr : float
            Multiplier for ATR to compute stop-loss distance.
        tp_atr : float
            Multiplier for ATR to compute take-profit distance.
        initial_balance : float
            Starting equity (e.g. 1.0 for 100%).
        """
        self.df = df.copy().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.state_cols = state_cols
        self.policy = policy
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.initial_balance = initial_balance

    def run(self) -> tuple[pd.DataFrame, float]:
        """
        Execute the backtest.

        Returns:
        --------
        trades : DataFrame
            Log of completed trades with columns:
            ['entry_time','exit_time','side','entry_price','exit_price','qty','pnl','balance_after']
        final_balance : float
            Equity after all trades.
        """
        balance = self.initial_balance
        position = None  # dict with keys: side, entry_price, qty, sl, tp, entry_idx
        trade_log = []

        for i in range(len(self.df) - 1):
            row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]
            price_next = next_row['c']

            # === Position management ===
            if position is not None:
                # check stop-loss or take-profit
                exit_price = None
                if position['side'] == 'long':
                    if price_next <= position['sl']:
                        exit_price = position['sl']
                    elif price_next >= position['tp']:
                        exit_price = position['tp']
                else:  # short
                    if price_next >= position['sl']:
                        exit_price = position['sl']
                    elif price_next <= position['tp']:
                        exit_price = position['tp']

                if exit_price is not None:
                    # close position
                    pnl = (exit_price - position['entry_price']) * position['qty'] * (1 if position['side']=='long' else -1)
                    balance += pnl
                    trade_log.append({
                        'entry_time':     self.df.loc[position['entry_idx'], :].name,
                        'exit_time':      next_row.name,
                        'side':           position['side'],
                        'entry_price':    position['entry_price'],
                        'exit_price':     exit_price,
                        'qty':            position['qty'],
                        'pnl':            pnl,
                        'balance_after':  balance
                    })
                    position = None
                # if no exit, continue to next bar
                continue

            # === New entry decision ===
            feat = row[self.feature_cols].values.reshape(1, -1)
            prob = row[self.state_cols].values
            decision = self.policy.decide(feat, prob)

            if decision['side'] != 'flat':
                size_frac = decision['size']
                atr = row['atr']
                stop_dist = self.sl_atr * atr
                take_dist = self.tp_atr * atr

                # calculate stop & take prices
                if decision['side'] == 'long':
                    sl = row['c'] - stop_dist
                    tp = row['c'] + take_dist
                else:
                    sl = row['c'] + stop_dist
                    tp = row['c'] - take_dist

                # compute quantity in base asset
                risk_amount = balance * size_frac
                qty = risk_amount / stop_dist

        position = {
                    'side':        decision['side'],
                    'entry_price': row['c'],
                    'qty':         qty,
                    'sl':          sl,
                    'tp':          tp,
                    'entry_idx':   i
                }

        # Close any open position at the final bar
        if position is not None:
            last_row = self.df.iloc[-1]
            exit_price = last_row['c']
            pnl = (
                (exit_price - position['entry_price'])
                * position['qty']
                * (1 if position['side'] == 'long' else -1)
            )
            balance += pnl
            trade_log.append({
                'entry_time':  self.df.loc[position['entry_idx'], :].name,
                'exit_time':   last_row.name,
                'side':        position['side'],
                'entry_price': position['entry_price'],
                'exit_price':  exit_price,
                'qty':         position['qty'],
                'pnl':         pnl,
                'balance_after': balance,
            })

        trades_df = pd.DataFrame(trade_log)
        return trades_df, balance
    
def run_backtest(symbol: str, hist: str = "60 days ago UTC") -> None:
    """Convenience helper to quickly backtest a single symbol.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``"BTCUSDT"``.
    hist : str, optional
        How far back to fetch historical klines (default ``"60 days ago UTC"``).
    """
    from cqio.binance_client import fetch_klines
    from features.feature_union import FeatureUnion
    from models.policy import PolicyModel
    from config import LIGHTGBM_PARAMS, INTERVAL
    from .metrics import summary

    # 1. fetch candles
    df = fetch_klines(symbol, INTERVAL, start_str=hist)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # 2. compute features & labels
    fu = FeatureUnion(n_states=3)
    feats = fu.fit_transform(df)
    ret_next = df["c"].pct_change().shift(-1)
    feats["ret_next"] = ret_next.reindex(feats.index)
    feats = feats.dropna(subset=["ret_next"])
    feats["label"] = 0
    feats.loc[feats["ret_next"] > 0, "label"] = 1
    feats.loc[feats["ret_next"] < 0, "label"] = 2

    state_cols = [c for c in feats.columns if c.startswith("state_")]
    feat_cols = [c for c in feats.columns if c not in state_cols + ["ret_next", "label"]]

    # 3. train model on the same history (demo purpose)
    X = feats[feat_cols + state_cols].values
    y = feats["label"].values
    policy = PolicyModel(LIGHTGBM_PARAMS)
    policy.train(X, y)

    # 4. run backtest
    bt = VectorBacktester(feats, feat_cols, state_cols, policy)
    trades, bal = bt.run()
    print(trades.tail())
    print("Final balance", bal)
    print(summary(trades))
