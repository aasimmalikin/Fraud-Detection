import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols                          # ← line 1: store cols

    def fit(self, X, y=None):
        self.freq_map_ = {                        # ← line 2: learn frequencies
            c: X[c].value_counts(normalize=True)
            for c in self.cols
        }
        return self                               # ← line 3: return self

    def transform(self, X):
        X = X.copy()                              # ← line 4: copy first
        for col in self.cols:
            X[f"{col}_freq"] = (                  # ← line 5: create new column
                X[col]
                .map(self.freq_map_[col])
                .astype(np.float32)
            )
        return X                                  # ← line 6: return result


class TransactionAggregator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        grp = X.groupby("card1")["TransactionAmt"]
        self.stats_ = grp.agg(["mean", "std", "count"])
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(
            self.stats_,
            on="card1",
            how="left",
            suffixes=("", "_card1")
        )
        return X