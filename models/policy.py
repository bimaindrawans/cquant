# models/policy.py

import lightgbm as lgb
import numpy as np

class PolicyModel:
    """
    A decision model that, given feature vectors and HMM state probabilities,
    predicts whether to go long, short, or stay flat, and how much to allocate.
    """

    def __init__(self, params: dict):
        """
        Initialize the LightGBM classifier with provided hyperparameters.
        
        params: dict of LightGBM parameters, e.g.:
            {
              'n_estimators': 300,
              'learning_rate': 0.05,
              'max_depth': 6,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
            }
        """
        self.clf = lgb.LGBMClassifier(**params)

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the policy model.
        
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of labels in {0: flat, 1: long, 2: short}
        """
        self.clf.fit(X, y)

    def decide(
        self,
        feat_row: np.ndarray,
        prob_state: np.ndarray,
        risk_aversion: float = 0.02
    ) -> dict:
        """
        Make a trading decision for one time step.

        feat_row: 2D array of shape (1, n_features) containing the feature vector.
        prob_state: 1D array of shape (n_states,) with HMM state probabilities.
        risk_aversion: maximum fraction of equity to risk on this trade.

        Returns:
            {
                "side": "long" | "short" | "flat",
                "size": float  # fraction of equity to allocate
            }
        """
        # Combine technical+sentiment features (in feat_row) with regime probs if desired
        # Here we assume feat_row already includes state probabilities appended.

        proba = self.clf.predict_proba(feat_row)[0]
        # proba indices: [flat_prob, long_prob, short_prob]
        flat_p, long_p, short_p = proba

        # If no strong edge, stay flat
        if max(long_p, short_p) < 0.55:
            return {"side": "flat", "size": 0.0}

        # Decide side by higher probability
        side = "long" if long_p > short_p else "short"
        edge = abs(long_p - short_p)

        # Position size as fraction of equity, capped by risk_aversion
        size = min(edge, risk_aversion)
        return {"side": side, "size": size}
