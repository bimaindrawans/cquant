"""Default configuration for cquant package."""

import os

# Trading interval for Binance klines (e.g. "15m", "1h")
INTERVAL = os.getenv("INTERVAL", "1h")

# Maximum position size in USDT for a single trade
MAX_POSITION_USDT = float(os.getenv("MAX_POSITION_USDT", "50"))

# Comma-separated list of static trading pairs
STATIC_PAIRS = os.getenv("STATIC_PAIRS", "BTCUSDT,ETHUSDT").split(',')

# Size of the dynamic universe from which pairs are selected
DYNAMIC_UNIVERSE_SIZE = int(os.getenv("DYNAMIC_UNIVERSE_SIZE", "20"))

# How many symbols to select from the dynamic universe
DYNAMIC_SELECT_K = int(os.getenv("DYNAMIC_SELECT_K", "3"))

# LightGBM hyper parameters
LIGHTGBM_PARAMS = {
    "n_estimators": int(os.getenv("LGBM_ESTIMATORS", "200")),
    "learning_rate": float(os.getenv("LGBM_LR", "0.05")),
    "max_depth": int(os.getenv("LGBM_MAX_DEPTH", "6")),
    "subsample": float(os.getenv("LGBM_SUBSAMPLE", "0.8")),
    "colsample_bytree": float(os.getenv("LGBM_COLSAMPLE", "0.8")),
}

# Base path for persistent storage
DATA_PATH = os.getenv("DATA_PATH", "data")

# Risk thresholds for adaptive position sizing
# list of tuples: (equity_upper_bound, risk_fraction)
RISK_THRESHOLDS = [
    (1000, 0.02),     # balance < 1k USDT -> risk 2%
    (5000, 0.015),    # balance < 5k USDT -> risk 1.5%
    (float("inf"), 0.01),  # balance >= 5k USDT -> risk 1%
]
