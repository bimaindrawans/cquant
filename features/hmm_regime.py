# features/hmm_regime.py

import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle

class RegimeFilter:
    """
    Hidden Markov Model filter for market regimes.
    - n_states: number of hidden regimes (e.g., 3 for Bull/Bear/Sideways)
    - covariance_type: 'diag' for diagonal covariance
    """

    def __init__(self, n_states: int = 3, covariance_type: str = "diag", random_state: int = 42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state
        )

    def fit(self, X: np.ndarray):
        """
        Fit the HMM to the feature matrix X.
        X should be shape (n_samples, n_features), e.g., returns & volatility.
        """
        self.model.fit(X)

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely state sequence for X.
        Returns an array of shape (n_samples,) with state indices [0..n_states-1].
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the posterior probability of each state for each sample.
        Returns an array of shape (n_samples, n_states).
        """
        return self.model.predict_proba(X)

    def save(self, path: str):
        """
        Persist the trained HMM to disk via pickle.
        """
        with open(path, "wb") as f:
            pickle.dump({
                "n_states": self.n_states,
                "covariance_type": self.model.covariance_type,
                "model": self.model
            }, f)

    @classmethod
    def load(cls, path: str) -> "RegimeFilter":
        """
        Load a saved HMM from disk.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(n_states=data["n_states"], covariance_type=data["covariance_type"])
        inst.model = data["model"]
        return inst
