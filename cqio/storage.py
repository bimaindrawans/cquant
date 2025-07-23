# io/storage.py

import os
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_PATH

class PersistentStorage:
    """
    Fallback storage for OHLCV and feature DataFrames.
    Persists to:
      - SQLite (for OHLCV time-series)
      - Parquet files (for arbitrary DataFrames)
    """
    def __init__(self):
        # ensure base data path exists
        self.base_path = Path(DATA_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # set up SQLite for OHLCV
        self.db_path = self.base_path / "storage.db"
        self.conn = sqlite3.connect(self.db_path)
        self._init_sqlite()

    def _init_sqlite(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY(symbol, ts)
            )
        """)
        self.conn.commit()

    # --- SQLite OHLCV methods ---

    def save_ohlcv(self, symbol: str, df: pd.DataFrame):
        """
        Save an OHLCV DataFrame to SQLite.
        Assumes df.index is a DatetimeIndex named 'ts', and columns ['o','h','l','c','v'].
        """
        df2 = df.copy().reset_index().rename(columns={
            'ts': 'ts',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        # convert timestamps to integer seconds
        df2['ts'] = (df2['ts'].astype('int64') // 1_000_000_000).astype(int)
        records = [
            (symbol, int(row.ts), row.open, row.high, row.low, row.close, row.volume)
            for row in df2.itertuples()
        ]
        c = self.conn.cursor()
        c.executemany(
            "INSERT OR REPLACE INTO ohlcv (symbol, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            records
        )
        self.conn.commit()

    def load_ohlcv(
        self,
        symbol: str,
        start_ts: int = None,
        end_ts: int = None
    ) -> pd.DataFrame:
        """
        Load OHLCV for `symbol` between optional Unix timestamp bounds.
        Returns DataFrame indexed by pd.DatetimeIndex with columns ['open','high','low','close','volume'].
        """
        query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        if start_ts is not None:
            query += " AND ts >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND ts <= ?"
            params.append(end_ts)
        query += " ORDER BY ts"
        df = pd.read_sql_query(query, self.conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=['open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        return df.set_index('ts')[['open','high','low','close','volume']]

    # --- Parquet methods for arbitrary DataFrames ---

    def save_parquet(self, filename: str, df: pd.DataFrame):
        """
        Save any DataFrame to <DATA_PATH>/<filename>.pq
        """
        path = self.base_path / f"{filename}.pq"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path))

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load a DataFrame from <DATA_PATH>/<filename>.pq
        """
        path = self.base_path / f"{filename}.pq"
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        table = pq.read_table(str(path))
        return table.to_pandas()
