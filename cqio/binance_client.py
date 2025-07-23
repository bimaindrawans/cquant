# io/binance_client.py

import os
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT

# Load your API credentials from environment variables or config.py
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_SECRET_KEY", "")

# Initialize the Binance REST client
client = Client(api_key=API_KEY, api_secret=API_SECRET)

def fetch_klines(
    symbol: str,
    interval: str,
    start_str: str = "1 day ago UTC",
    end_str: str = None
) -> pd.DataFrame:
    """
    Download historical OHLCV bars for a given symbol/interval.
    Returns a DataFrame indexed by timestamp with columns ['o','h','l','c','v'].
    """
    klines = client.get_historical_klines(symbol, interval, start_str, end_str or "now UTC")
    df = pd.DataFrame(klines, columns=[
        "ts", "o", "h", "l", "c", "v",
        "close_time", "quote_asset_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")[["o", "h", "l", "c", "v"]].astype(float)
    return df

def get_account_balance(asset: str = "USDT") -> float:
    """
    Fetch the free balance for a given asset (e.g. USDT).
    """
    info = client.get_asset_balance(asset=asset)
    return float(info["free"]) if info and info.get("free") is not None else 0.0

def place_market_order(
    symbol: str,
    side: str,
    quantity: float
) -> dict:
    """
    Place a market order.
    side: 'BUY' or 'SELL'
    quantity: amount in base asset
    """
    order = client.create_order(
        symbol=symbol,
        side=SIDE_BUY if side.lower() == "buy" else SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return order

def place_limit_order(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    time_in_force: str = "GTC"
) -> dict:
    """
    Place a limit order.
    time_in_force: 'GTC', 'IOC', or 'FOK'
    """
    order = client.create_order(
        symbol=symbol,
        side=SIDE_BUY if side.lower() == "buy" else SIDE_SELL,
        type=ORDER_TYPE_LIMIT,
        timeInForce=time_in_force,
        quantity=quantity,
        price=str(price)
    )
    return order

def get_open_orders(symbol: str = None) -> list:
    """
    Retrieve a list of open orders. If symbol is None, returns all open orders.
    """
    if symbol:
        return client.get_open_orders(symbol=symbol)
    return client.get_open_orders()

def cancel_order(symbol: str, order_id: int) -> dict:
    """
    Cancel an order by symbol and order ID.
    """
    return client.cancel_order(symbol=symbol, orderId=order_id)
