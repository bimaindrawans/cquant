# CQuant

CQuant is a research project for developing algorithmic cryptocurrency trading strategies. It combines Hidden Markov Models (HMM), traditional machine learning, stochastic processes and sentiment analysis techniques.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the trading bot (paper mode by default, add `--live` to place real orders):

```bash
python -m utils.cli trade
```

Run a quick backtest of a single pair:

```bash
python -m utils.cli bt --pair BTCUSDT
```

## Configuration

Configuration values can be customised via environment variables or a `.env` file. The defaults reside in [`config.py`](config.py). Important variables include:

- `BINANCE_API_KEY`, `BINANCE_SECRET_KEY` – Binance credentials for live trading.
- `INTERVAL` – kline interval (default `1h`).
- `MAX_POSITION_USDT` – maximum position size (default `50`).
- `STATIC_PAIRS` – comma separated list of always traded pairs.
- `DYNAMIC_UNIVERSE_SIZE` – size of the symbol universe scanned.
- `DYNAMIC_SELECT_K` – number of dynamic pairs selected.
- `LGBM_ESTIMATORS`, `LGBM_LR`, `LGBM_MAX_DEPTH`, `LGBM_SUBSAMPLE`, `LGBM_COLSAMPLE` – LightGBM hyper parameters.
- `DATA_PATH` – base path for persistent storage.
- `LOG_LEVEL`, `LOG_PATH` – logging configuration.
- `CRYPTOPANIC_TOKEN` – optional token used for the sentiment module.

Copy `.env.example` to `.env` and edit, or export variables in your shell before running the CLI.
