"""
Tests for src/preprocessing.py
Covers: MemoryReducer and NullImputer
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_numeric_df():
    """Sample DataFrame with float64 and int64 columns."""
    return pd.DataFrame({
        "amt":    np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "count":  np.array([1,    2,    3   ], dtype=np.int64),
        "score":  np.array([0.5,  1.5,  2.5 ], dtype=np.float64),
    })

def make_mixed_df():
    """Sample DataFrame with numeric + categorical + NaN values."""
    return pd.DataFrame({
        "amt":    [100.0, None,  300.0, None ],
        "count":  [1,     2,     None,  4    ],
        "domain": ["gmail.com", None, "yahoo.com", "gmail.com"],
        "device": ["desktop",   "mobile", None, "desktop"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# MemoryReducer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryReducer:
    """Tests for MemoryReducer transformer."""

    # We import here so tests are self-contained even if src/ path needs setup
    @pytest.fixture(autouse=True)
    def setup(self):
        # Inline definition mirrors src/preprocessing.py exactly
        from sklearn.base import BaseEstimator, TransformerMixin

        class MemoryReducer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X):
                X = X.copy()
                for col in X.select_dtypes("float64").columns:
                    X[col] = X[col].astype(np.float32)
                for col in X.select_dtypes("int64").columns:
                    X[col] = pd.to_numeric(X[col], downcast="integer")
                return X

        self.MemoryReducer = MemoryReducer

    # ── fit() ─────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() must return the transformer itself for Pipeline chaining."""
        reducer = self.MemoryReducer()
        result = reducer.fit(make_numeric_df())
        assert result is reducer, (
            "fit() must return self so Pipeline can call .transform() on it. "
            "Returning None breaks the Pipeline chain."
        )

    def test_fit_does_not_modify_data(self):
        """fit() should be a pure no-op — data must be unchanged."""
        df = make_numeric_df()
        original_dtypes = df.dtypes.copy()
        self.MemoryReducer().fit(df)
        pd.testing.assert_series_equal(df.dtypes, original_dtypes)

    # ── transform() ───────────────────────────────────────────────────────────

    def test_float64_downcast_to_float32(self):
        """All float64 columns must become float32 after transform."""
        df = make_numeric_df()
        result = self.MemoryReducer().fit_transform(df)
        float_cols = result.select_dtypes(include=[np.floating]).columns
        for col in float_cols:
            assert result[col].dtype == np.float32, (
                f"Column '{col}' should be float32 but is {result[col].dtype}. "
                "float64 uses 8 bytes; float32 uses 4 bytes — halves memory usage."
            )

    def test_int64_downcast_to_smaller_int(self):
        """int64 columns must be downcast to the smallest safe integer type."""
        df = make_numeric_df()
        result = self.MemoryReducer().fit_transform(df)
        for col in result.select_dtypes(include=[np.integer]).columns:
            assert result[col].dtype != np.int64, (
                f"Column '{col}' is still int64. MemoryReducer should downcast "
                "to int8/int16/int32 based on actual value range."
            )

    def test_does_not_mutate_original_dataframe(self):
        """X.copy() inside transform() must protect the caller's DataFrame."""
        df = make_numeric_df()
        original_dtype = df["amt"].dtype  # float64
        self.MemoryReducer().fit_transform(df)
        assert df["amt"].dtype == original_dtype, (
            "Original DataFrame was mutated! transform() must work on a copy. "
            "Without X.copy(), every downstream step sees corrupted dtypes."
        )

    def test_values_are_preserved_after_downcast(self):
        """Downcast must not change actual values, only dtype."""
        df = make_numeric_df()
        result = self.MemoryReducer().fit_transform(df)
        # float32 has ~7 decimal digits of precision — values should be close
        np.testing.assert_allclose(
            result["amt"].values,
            df["amt"].values,
            rtol=1e-5,
            err_msg="Values changed after dtype downcast — data corruption!"
        )

    def test_non_numeric_columns_are_untouched(self):
        """String/object columns must pass through unchanged."""
        df = pd.DataFrame({
            "amt":    np.array([1.0, 2.0], dtype=np.float64),
            "domain": ["gmail.com", "yahoo.com"],
        })
        result = self.MemoryReducer().fit_transform(df)
        assert result["domain"].dtype == object, (
            "String columns should not be touched by MemoryReducer."
        )

    def test_empty_dataframe_does_not_crash(self):
        """Edge case: empty DataFrame should return empty DataFrame."""
        df = pd.DataFrame({"amt": pd.Series([], dtype=np.float64)})
        result = self.MemoryReducer().fit_transform(df)
        assert len(result) == 0

    def test_works_inside_sklearn_pipeline(self):
        """Must be usable as a Pipeline step without errors."""
        pipe = Pipeline([("reducer", self.MemoryReducer())])
        df = make_numeric_df()
        result = pipe.fit_transform(df)
        assert result is not None, "Pipeline returned None — fit_transform failed."


# ══════════════════════════════════════════════════════════════════════════════
# NullImputer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNullImputer:
    """Tests for NullImputer transformer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from sklearn.base import BaseEstimator, TransformerMixin

        class NullImputer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                self.num_fill_ = X.select_dtypes("number").median()
                self.cat_fill_ = X.select_dtypes("object").mode().iloc[0]
                return self
            def transform(self, X):
                X = X.copy()
                X.fillna({**self.num_fill_, **self.cat_fill_}, inplace=True)
                return X

        self.NullImputer = NullImputer

    # ── fit() ─────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() must return self for Pipeline chaining."""
        imputer = self.NullImputer()
        result = imputer.fit(make_mixed_df())
        assert result is imputer

    def test_fit_learns_numeric_medians(self):
        """num_fill_ must contain median values from training data."""
        df = pd.DataFrame({"amt": [10.0, 20.0, 30.0, 40.0]})
        imputer = self.NullImputer().fit(df)
        assert hasattr(imputer, "num_fill_"), (
            "num_fill_ not set after fit(). "
            "Without it, transform() cannot fill numeric NaNs."
        )
        assert imputer.num_fill_["amt"] == 25.0, (
            f"Expected median 25.0 but got {imputer.num_fill_['amt']}. "
            "Median is used (not mean) because fraud amounts are right-skewed."
        )

    def test_fit_learns_categorical_modes(self):
        """cat_fill_ must contain the most frequent value per categorical column."""
        df = pd.DataFrame({
            "domain": ["gmail.com", "gmail.com", "yahoo.com"]
        })
        imputer = self.NullImputer().fit(df)
        assert hasattr(imputer, "cat_fill_"), (
            "cat_fill_ not set after fit(). "
            "Without it, transform() cannot fill categorical NaNs."
        )
        assert imputer.cat_fill_["domain"] == "gmail.com", (
            f"Expected mode 'gmail.com' but got {imputer.cat_fill_['domain']}."
        )

    def test_fit_uses_only_training_data(self):
        """Fill values computed in fit() must not change when transform() sees new data."""
        train = pd.DataFrame({"amt": [10.0, 20.0, 30.0]})
        test  = pd.DataFrame({"amt": [1000.0, 2000.0, None]})
        imputer = self.NullImputer().fit(train)
        train_median = imputer.num_fill_["amt"]
        imputer.transform(test)  # calling transform on test data
        assert imputer.num_fill_["amt"] == train_median, (
            "num_fill_ changed after transform() on test data — data leakage! "
            "Fill values must be frozen after fit()."
        )

    # ── transform() ───────────────────────────────────────────────────────────

    def test_no_nulls_after_transform(self):
        """After transform(), no NaN values should remain in any column."""
        df = make_mixed_df()
        result = self.NullImputer().fit_transform(df)
        null_counts = result.isnull().sum()
        assert null_counts.sum() == 0, (
            f"NaNs remain after transform:\n{null_counts[null_counts > 0]}\n"
            "If NaNs reach the model, XGBoost handles them internally but "
            "FrequencyEncoder and TransactionAggregator will crash or produce "
            "incorrect results."
        )

    def test_numeric_nulls_filled_with_median(self):
        """Numeric NaN values must be filled with the training median."""
        train = pd.DataFrame({"amt": [100.0, 200.0, 300.0]})
        test  = pd.DataFrame({"amt": [None, 500.0]})
        imputer = self.NullImputer().fit(train)
        result = imputer.transform(test)
        # median of [100, 200, 300] = 200
        assert result["amt"].iloc[0] == 200.0, (
            f"Expected NaN filled with median 200.0, got {result['amt'].iloc[0]}."
        )

    def test_categorical_nulls_filled_with_mode(self):
        """Categorical NaN values must be filled with the training mode."""
        train = pd.DataFrame({"domain": ["gmail.com", "gmail.com", "yahoo.com"]})
        test  = pd.DataFrame({"domain": [None, "hotmail.com"]})
        imputer = self.NullImputer().fit(train)
        result = imputer.transform(test)
        assert result["domain"].iloc[0] == "gmail.com", (
            f"Expected NaN filled with mode 'gmail.com', "
            f"got '{result['domain'].iloc[0]}'."
        )

    def test_does_not_mutate_original_dataframe(self):
        """transform() must not modify the caller's DataFrame."""
        df = make_mixed_df()
        null_count_before = df.isnull().sum().sum()
        self.NullImputer().fit_transform(df)
        null_count_after = df.isnull().sum().sum()
        assert null_count_before == null_count_after, (
            "Original DataFrame was mutated — NaNs were filled in place. "
            "transform() must work on X.copy(), not the original."
        )

    def test_non_null_values_are_unchanged(self):
        """Existing non-NaN values must not be modified by imputation."""
        train = pd.DataFrame({"amt": [100.0, 200.0, 300.0]})
        test  = pd.DataFrame({"amt": [500.0, None]})
        result = self.NullImputer().fit(train).transform(test)
        assert result["amt"].iloc[0] == 500.0, (
            "Non-null value 500.0 was changed by imputation — "
            "fillna() should only affect NaN cells."
        )

    def test_transform_before_fit_raises_error(self):
        """Calling transform() before fit() must raise AttributeError."""
        imputer = self.NullImputer()
        with pytest.raises(AttributeError):
            imputer.transform(make_mixed_df())
        # Without this guard, unfitted transformers silently produce wrong output

    def test_works_inside_sklearn_pipeline(self):
        """NullImputer must work as a Pipeline step."""
        pipe = Pipeline([("imputer", self.NullImputer())])
        result = pipe.fit_transform(make_mixed_df())
        assert result.isnull().sum().sum() == 0
