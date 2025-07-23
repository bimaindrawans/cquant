# features/feature_union.py

import numpy as np
import pandas as pd
from .technical import make_technical_features
from .hmm_regime import RegimeFilter

class FeatureUnion:
    """
    Combines technical indicators and HMM regime probabilities into one feature DataFrame.
    
    Workflow:
      1. fit(df): compute HMM on historical data
      2. transform(df): apply technical transforms & HMM filtering to produce features
      3. fit_transform(df): do both in sequence
    """

    def __init__(self, n_states: int = 3, covariance_type: str = "diag", random_state: int = 42):
        self.n_states = n_states
        self.regime_filter = RegimeFilter(
            n_states=n_states,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self._is_fitted = False

    def _prepare_hmm_input(self, df_tech: pd.DataFrame) -> np.ndarray:
        """
        Build the observation matrix for HMM:
          - returns (%) as first column
          - ATR as second column
        """
        # pct change of close price
        ret = df_tech['c'].pct_change().fillna(0.0).values
        # ATR (already in df_tech)
        atr = df_tech['atr'].ffill().fillna(0.0).values
        return np.column_stack([ret, atr])

    def fit(self, df: pd.DataFrame):
        """
        Fit the HMM on historical data.
        `df` must contain raw OHLCV with columns ['o','h','l','c','v'].
        """
        # 1) compute technicals
        df_tech = make_technical_features(df)
        # 2) prepare HMM observations
        X = self._prepare_hmm_input(df_tech)
        # 3) fit HMM
        self.regime_filter.fit(X)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the technical feature pipeline and HMM filter to new data.
        Returns a DataFrame with:
          - original OHLCV columns
          - 'atr', 'rsi', 'stoch_k', 'stoch_d'
          - 'state_0', ..., 'state_{n_states-1}' probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureUnion must be fitted before transform()")

        # 1) technical indicators
        df_tech = make_technical_features(df)

        # 2) HMM regime probabilities
        X = self._prepare_hmm_input(df_tech)
        probs = self.regime_filter.predict_proba(X)
        cols = [f"state_{i}" for i in range(self.n_states)]
        df_probs = pd.DataFrame(probs, index=df_tech.index, columns=cols)

        # 3) concatenate
        df_feats = pd.concat([df_tech, df_probs], axis=1)

        # 4) drop any initial NaNs and return
        return df_feats.dropna()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: fit on `df` then transform the same `df`.
        """
        return self.fit(df).transform(df)
