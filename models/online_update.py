# models/online_update.py

import os
from pathlib import Path
import pandas as pd
import pickle
from datetime import datetime, timedelta

from .policy import PolicyModel

class OnlineUpdater:
    """
    Manage online retraining of a PolicyModel.
    - Records feature vectors and true labels/outcomes after each trade.
    - Maintains a rolling window of training data.
    - Retrains the model on a schedule or on-demand.
    """

    def __init__(
        self,
        model: PolicyModel,
        training_path: str = "data/training_data.parquet",
        window_size: int = 50_000,
        retrain_interval_days: int = 1
    ):
        """
        model: instance of PolicyModel to update
        training_path: where to store accumulated training data
        window_size: max number of rows to keep for training
        retrain_interval_days: how often to retrain (in days)
        """
        self.model = model
        self.training_file = Path(training_path)
        self.window_size = window_size
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self._last_retrain = datetime.min

        # ensure directory exists
        self.training_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> pd.DataFrame:
        if self.training_file.exists():
            return pd.read_parquet(self.training_file)
        else:
            # empty DataFrame; label column will be added later
            return pd.DataFrame()

    def _save_history(self, df: pd.DataFrame):
        df.to_parquet(self.training_file, index=False)

    def add_observations(
        self,
        features: pd.DataFrame,
        labels: pd.Series
    ):
        """
        Append new training rows.
        features: DataFrame of shape (n, m)
        labels: Series of length n with values in {0,1,2}
        """
        hist = self._load_history()
        # align and merge
        new = features.copy()
        new['label'] = labels.values
        combined = pd.concat([hist, new], ignore_index=True)
        # keep only the most recent window_size rows
        if len(combined) > self.window_size:
            combined = combined.iloc[-self.window_size :].reset_index(drop=True)
        self._save_history(combined)

    def should_retrain(self) -> bool:
        """
        Returns True if enough time has passed since last retrain.
        """
        return datetime.utcnow() - self._last_retrain >= self.retrain_interval

    def retrain(self):
        """
        Retrain the model on the accumulated data.
        Updates self.model in-place and resets the retrain timer.
        """
        df = self._load_history()
        if df.empty or 'label' not in df.columns:
            # nothing to train on
            return

        # split features and labels
        X = df.drop(columns=['label']).values
        y = df['label'].values

        # train the policy model
        self.model.train(X, y)
        self._last_retrain = datetime.utcnow()

    def save_model(self, path: str = "models/policy_model.pkl"):
        """
        Persist the trained PolicyModel to disk.
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str = "models/policy_model.pkl"):
        """
        Load a persisted PolicyModel from disk.
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)
