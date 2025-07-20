# utils/cli.py

import click
from live.trader import main_loop
from backtest.backtester import run_backtest

@click.group()
def cli():
    """Command-line interface for the crypto HMM-ML bot."""
    pass

@cli.command()
@click.option("--paper/--live", default=True, help="Run in paper mode (no real orders) or live mode.")
def trade(paper):
    """Start the trading bot (paper-trade or live)."""
    main_loop(paper)

@cli.command()
@click.option("--pair", default="BTCUSDT", help="Ticker symbol to backtest (e.g. BTCUSDT).")
def bt(pair):
    """Run a backtest for the specified trading pair."""
    run_backtest(pair)

if __name__ == "__main__":
    cli()
