import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MemoryReducer(BaseEstimator, TransformerMixin):
    '''Downcasting the number dtypes to reduce memory usage'''
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes("float64").columns:
            X[col] = X[col].astype(np.float32)
        for col in X.select_dtypes("int64").columns:
            X[col] = pd.to_numeric(X[col], downcast="integer")
        return X
class NullImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.num_fill = X.select_dtypes("number").median()
        self.cat_fill = X.select_dtypes("object").mode().iloc[0]
        return self
    def transform(self, X):
        X = X.copy()
        X.fillna({**self.num_fill, **self.cat_fill}, inplace = True)
        return X
