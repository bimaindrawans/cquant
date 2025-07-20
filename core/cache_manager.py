import os
import time
import atexit
import tempfile
import sqlite3
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["TempCache"]


class TempCache:
    """A lightweight *temporary* cache for OHLCV ``pandas`` DataFrames and
    arbitrary binary blobs (e.g. serialized model parameters).

    All artefacts are kept inside a :class:`tempfile.TemporaryDirectory`, so
    they disappear once the Python process ends (or :py:meth:`cleanup` is
    invoked).  Each entry obeys a *time‑to‑live* (``ttl_min``), after which it
    is treated as stale and automatically purged.

    Parameters
    ----------
    ttl_min : int, default 30
        Time‑to‑live in *minutes* for cached objects.
    """

    def __init__(self, ttl_min: int = 30) -> None:
        self.ttl: int = ttl_min * 60  # seconds

        # ────────────────────────────────────────────────────────────────
        # Root folder for all cached files, created in /tmp or platform
        # specific TMP dir.  The folder (and everything inside) is deleted
        # when ``self.dir.cleanup()`` is called.
        # ────────────────────────────────────────────────────────────────
        self.dir = tempfile.TemporaryDirectory()  # e.g. /tmp/tmpabcd1234

        # Single SQLite database for key–blob storage
        self._db_path = os.path.join(self.dir.name, "tmp.db")
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_db()

        # Register clean‑up handler so we *never* leave junk on disk
        atexit.register(self._cleanup)

    # ------------------------------------------------------------------
    # Public helpers for OHLCV DataFrame caching
    # ------------------------------------------------------------------
    def put_df(self, symbol: str, df: pd.DataFrame) -> None:
        """Persist *df* to Parquet.  An existing file for *symbol* will be
        silently overwritten.
        """
        file_path = self._df_path(symbol)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)

    def get_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return the cached DataFrame for *symbol* or :pydata:`None` when the
        file does not exist **or** is older than the TTL.
        """
        file_path = self._df_path(symbol)
        if not os.path.exists(file_path):
            return None

        # Check age
        if time.time() - os.path.getmtime(file_path) > self.ttl:
            try:
                os.remove(file_path)
            finally:
                return None

        table = pq.read_table(file_path)
        return table.to_pandas()

    # ------------------------------------------------------------------
    # Public helpers for arbitrary binary blobs (e.g. HMM model params)
    # ------------------------------------------------------------------
    def save_blob(self, key: str, blob: bytes) -> None:
        """Store *blob* under *key*.  Overwrites any existing entry."""
        with self._db:
            self._db.execute(
                "REPLACE INTO blobs(key, value, ts) VALUES (?, ?, strftime('%s','now'))",
                (key, blob),
            )

    def load_blob(self, key: str) -> Optional[bytes]:
        """Retrieve blob by *key* or :pydata:`None` if missing or expired."""
        row = self._db.execute(
            "SELECT value, ts FROM blobs WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None

        value, ts = row
        if time.time() - ts > self.ttl:
            # Expired – remove and signal miss
            with self._db:
                self._db.execute("DELETE FROM blobs WHERE key = ?", (key,))
            return None
        return value

    # ------------------------------------------------------------------
    # Maintenance & housekeeping
    # ------------------------------------------------------------------
    def vacuum(self) -> None:
        """Manually purge stale Parquet files *and* expired blobs."""
        now = time.time()
        # Files
        for name in os.listdir(self.dir.name):
            if not name.endswith(".parquet"):
                continue
            path = os.path.join(self.dir.name, name)
            if now - os.path.getmtime(path) > self.ttl:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Blobs & SQLite vacuum
        with self._db:
            self._db.execute(
                "DELETE FROM blobs WHERE strftime('%s','now') - ts > ?",
                (self.ttl,),
            )
            self._db.execute("VACUUM")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """Create table schema if it doesn't exist."""
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    key   TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    ts    INTEGER NOT NULL       -- POSIX seconds
                )
                """
            )

    def _df_path(self, symbol: str) -> str:
        """Return safe Parquet filename for *symbol* inside temp dir."""
        safe = symbol.replace("/", "_").upper()
        return os.path.join(self.dir.name, f"{safe}.parquet")

    # ------------------------------------------------------------------
    # Destructor / finaliser
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """Close the database and delete temporary directory (and contents)."""
        try:
            self._db.close()
        finally:
            # Remove the entire tmp directory recursively
            self.dir.cleanup()
