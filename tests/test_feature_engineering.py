"""
Tests for src/feature_engineering.py
Covers: FrequencyEncoder and TransactionAggregator
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# ── Inline class definitions (mirrors src/feature_engineering.py) ─────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.freq_map_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[f"{col}_freq"] = X[col].map(self.freq_map_[col]).astype(np.float32)
        return X


class TransactionAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        grp = X.groupby("card1")["TransactionAmt"]
        self.stats_ = grp.agg(["mean", "std", "count"])
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.stats_, on="card1", how="left", suffixes=("", "_card1"))
        return X


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def email_df():
    """Training DataFrame with known email domain frequencies."""
    return pd.DataFrame({
        "P_emaildomain": [
            "gmail.com", "gmail.com", "gmail.com",
            "yahoo.com", "yahoo.com",
            "hotmail.com"
        ]
    })
    # Frequencies: gmail=0.5, yahoo=0.333, hotmail=0.167


@pytest.fixture
def transaction_df():
    """Training DataFrame with card1 and TransactionAmt for aggregation."""
    return pd.DataFrame({
        "card1":          [1,     1,     1,     2,     2    ],
        "TransactionAmt": [100.0, 200.0, 300.0, 50.0,  150.0],
        "isFraud":        [0,     0,     1,     0,     1    ],
    })
    # card1=1: mean=200, std~100, count=3
    # card1=2: mean=100, std~70,  count=2


# ══════════════════════════════════════════════════════════════════════════════
# FrequencyEncoder Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyEncoder:

    # ── __init__() ────────────────────────────────────────────────────────────

    def test_cols_stored_as_attribute(self):
        """cols passed to __init__ must be stored as self.cols."""
        enc = FrequencyEncoder(cols=["P_emaildomain"])
        assert enc.cols == ["P_emaildomain"], (
            "self.cols not set in __init__. Without it, fit() and transform() "
            "have no way to know which columns to encode."
        )

    def test_get_params_works(self, email_df):
        """BaseEstimator.get_params() must return __init__ parameters."""
        enc = FrequencyEncoder(cols=["P_emaildomain"])
        params = enc.get_params()
        assert "cols" in params, (
            "get_params() doesn't include 'cols'. This breaks GridSearchCV "
            "and Pipeline cloning. Ensure __init__ stores self.cols = cols."
        )

    # ── fit() ─────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self, email_df):
        enc = FrequencyEncoder(cols=["P_emaildomain"])
        result = enc.fit(email_df)
        assert result is enc, "fit() must return self for Pipeline chaining."

    def test_fit_creates_freq_map(self, email_df):
        """freq_map_ must be set after fitting."""
        enc = FrequencyEncoder(cols=["P_emaildomain"]).fit(email_df)
        assert hasattr(enc, "freq_map_"), (
            "freq_map_ not created in fit(). Without it, transform() has no "
            "frequency table to look up values from."
        )
        assert "P_emaildomain" in enc.freq_map_

    def test_frequencies_sum_to_one(self, email_df):
        """All frequency values for a column must sum to 1.0 (proportions)."""
        enc = FrequencyEncoder(cols=["P_emaildomain"]).fit(email_df)
        freq_sum = enc.freq_map_["P_emaildomain"].sum()
        assert abs(freq_sum - 1.0) < 1e-5, (
            f"Frequencies sum to {freq_sum}, not 1.0. "
            "value_counts(normalize=True) must be used, not normalize=False. "
            "Raw counts instead of proportions would give the model wrong signal."
        )

    def test_frequencies_are_between_zero_and_one(self, email_df):
        """Every frequency value must be in (0, 1]."""
        enc = FrequencyEncoder(cols=["P_emaildomain"]).fit(email_df)
        freqs = enc.freq_map_["P_emaildomain"]
        assert (freqs > 0).all() and (freqs <= 1.0).all(), (
            "Frequency values outside (0,1] detected. "
            "This indicates raw counts were used instead of proportions."
        )

    def test_freq_map_frozen_after_fit(self, email_df):
        """freq_map_ must not change when transform() is called on new data."""
        enc = FrequencyEncoder(cols=["P_emaildomain"]).fit(email_df)
        gmail_freq_before = enc.freq_map_["P_emaildomain"]["gmail.com"]

        # Transform completely different data
        new_data = pd.DataFrame({"P_emaildomain": ["hotmail.com"] * 100})
        enc.transform(new_data)

        gmail_freq_after = enc.freq_map_["P_emaildomain"]["gmail.com"]
        assert gmail_freq_before == gmail_freq_after, (
            "freq_map_ changed after transform() — data leakage! "
            "fit() statistics must be frozen and never recomputed in transform()."
        )

    # ── transform() ───────────────────────────────────────────────────────────

    def test_new_freq_column_created(self, email_df):
        """transform() must create a new column named '{col}_freq'."""
        result = FrequencyEncoder(cols=["P_emaildomain"]).fit_transform(email_df)
        assert "P_emaildomain_freq" in result.columns, (
            "Expected column 'P_emaildomain_freq' not found. "
            "FrequencyEncoder should ADD a new column, not replace the original."
        )

    def test_original_column_preserved(self, email_df):
        """Original column must still exist after encoding."""
        result = FrequencyEncoder(cols=["P_emaildomain"]).fit_transform(email_df)
        assert "P_emaildomain" in result.columns, (
            "Original 'P_emaildomain' column was removed. "
            "FrequencyEncoder should add a new column, not replace the original, "
            "because other transformers downstream may need the original."
        )

    def test_freq_column_dtype_is_float32(self, email_df):
        """Frequency column must be float32, not float64."""
        result = FrequencyEncoder(cols=["P_emaildomain"]).fit_transform(email_df)
        assert result["P_emaildomain_freq"].dtype == np.float32, (
            "Frequency column is float64 instead of float32. "
            "Use .astype(np.float32) to maintain memory efficiency."
        )

    def test_correct_frequency_values(self, email_df):
        """Frequency values must exactly match the training proportions."""
        result = FrequencyEncoder(cols=["P_emaildomain"]).fit_transform(email_df)
        gmail_rows = result[result["P_emaildomain"] == "gmail.com"]
        expected = 3 / 6  # gmail appears 3 out of 6 times
        actual = gmail_rows["P_emaildomain_freq"].iloc[0]
        assert abs(actual - expected) < 1e-5, (
            f"Expected gmail.com frequency {expected}, got {actual}."
        )

    def test_unseen_category_produces_nan(self, email_df):
        """Categories in test set not seen in training must produce NaN."""
        enc = FrequencyEncoder(cols=["P_emaildomain"]).fit(email_df)
        test_df = pd.DataFrame({"P_emaildomain": ["completely_new_domain.com"]})
        result = enc.transform(test_df)
        assert pd.isna(result["P_emaildomain_freq"].iloc[0]), (
            "Unseen category should map to NaN, not crash or produce a wrong value. "
            "NaN is the correct signal: 'we have no frequency data for this value'. "
            "NullImputer downstream will handle this NaN."
        )

    def test_does_not_mutate_original(self, email_df):
        """transform() must not modify the caller's DataFrame."""
        original_cols = set(email_df.columns)
        FrequencyEncoder(cols=["P_emaildomain"]).fit_transform(email_df)
        assert set(email_df.columns) == original_cols, (
            "Original DataFrame was mutated — new columns added to it. "
            "transform() must work on X.copy()."
        )

    def test_multiple_columns_encoded(self):
        """Multiple columns must all be encoded in one call."""
        df = pd.DataFrame({
            "card1":  [1, 1, 2, 3],
            "domain": ["a", "a", "b", "c"],
        })
        enc = FrequencyEncoder(cols=["card1", "domain"]).fit_transform(df)
        assert "card1_freq"  in enc.columns
        assert "domain_freq" in enc.columns

    def test_works_inside_pipeline(self, email_df):
        """Must plug into sklearn Pipeline without errors."""
        pipe = Pipeline([
            ("enc", FrequencyEncoder(cols=["P_emaildomain"]))
        ])
        result = pipe.fit_transform(email_df)
        assert "P_emaildomain_freq" in result.columns


# ══════════════════════════════════════════════════════════════════════════════
# TransactionAggregator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransactionAggregator:

    # ── fit() ─────────────────────────────────────────────────────────────────

    def test_fit_returns_self(self, transaction_df):
        agg = TransactionAggregator()
        result = agg.fit(transaction_df)
        assert result is agg, "fit() must return self for Pipeline chaining."

    def test_fit_creates_stats(self, transaction_df):
        """stats_ must be created after fit()."""
        agg = TransactionAggregator().fit(transaction_df)
        assert hasattr(agg, "stats_"), (
            "stats_ not created in fit(). Without it, transform() has no "
            "per-card statistics to merge onto transactions."
        )

    def test_stats_has_correct_columns(self, transaction_df):
        """stats_ must have mean, std, and count columns."""
        agg = TransactionAggregator().fit(transaction_df)
        assert set(agg.stats_.columns) == {"mean", "std", "count"}, (
            f"stats_ columns are {set(agg.stats_.columns)}. "
            "Expected mean, std, count — these are the three behavioral signals "
            "that give the model context about each card's spending pattern."
        )

    def test_correct_mean_per_card(self, transaction_df):
        """Mean transaction amount must be computed correctly per card."""
        agg = TransactionAggregator().fit(transaction_df)
        # card1=1: (100+200+300)/3 = 200
        assert abs(agg.stats_.loc[1, "mean"] - 200.0) < 1e-5, (
            f"card1=1 mean should be 200.0, got {agg.stats_.loc[1, 'mean']}."
        )

    def test_correct_count_per_card(self, transaction_df):
        """Transaction count must be correct per card."""
        agg = TransactionAggregator().fit(transaction_df)
        assert agg.stats_.loc[1, "count"] == 3, (
            f"card1=1 should have count=3, got {agg.stats_.loc[1, 'count']}. "
            "Count indicates how much history we have — a card with 1 transaction "
            "is riskier than one with 500."
        )

    def test_stats_frozen_after_fit(self, transaction_df):
        """stats_ must not change when transform() is called on new data."""
        agg = TransactionAggregator().fit(transaction_df)
        mean_before = agg.stats_.loc[1, "mean"]

        new_data = pd.DataFrame({
            "card1": [1], "TransactionAmt": [9999.0], "isFraud": [1]
        })
        agg.transform(new_data)
        assert agg.stats_.loc[1, "mean"] == mean_before, (
            "stats_ changed after transform() — data leakage! "
            "Statistics must be frozen from fit() and never updated in transform()."
        )

    # ── transform() ───────────────────────────────────────────────────────────

    def test_stats_columns_added_to_output(self, transaction_df):
        """mean, std, count columns must appear in transform() output."""
        result = TransactionAggregator().fit_transform(transaction_df)
        for col in ["mean", "std", "count"]:
            assert col in result.columns, (
                f"Column '{col}' missing from transform() output. "
                f"The merge with stats_ must attach all three stat columns."
            )

    def test_row_count_preserved(self, transaction_df):
        """transform() must not add or drop rows."""
        result = TransactionAggregator().fit_transform(transaction_df)
        assert len(result) == len(transaction_df), (
            f"Row count changed: {len(transaction_df)} → {len(result)}. "
            "how='left' in merge must preserve all original transaction rows."
        )

    def test_unseen_card_gets_nan_stats(self, transaction_df):
        """Cards in test set not seen in training must produce NaN stats."""
        agg = TransactionAggregator().fit(transaction_df)
        test_df = pd.DataFrame({
            "card1": [9999], "TransactionAmt": [100.0], "isFraud": [0]
        })
        result = agg.transform(test_df)
        assert pd.isna(result["mean"].iloc[0]), (
            "Unseen card should get NaN stats, not crash. "
            "how='left' in merge handles this correctly. "
            "NullImputer downstream will fill these NaNs."
        )

    def test_does_not_mutate_original(self, transaction_df):
        """transform() must not modify the caller's DataFrame."""
        original_cols = set(transaction_df.columns)
        TransactionAggregator().fit_transform(transaction_df)
        assert set(transaction_df.columns) == original_cols, (
            "Original DataFrame columns were modified. "
            "transform() must work on X.copy()."
        )

    def test_transform_before_fit_raises_error(self, transaction_df):
        """Calling transform() without fit() must raise AttributeError."""
        agg = TransactionAggregator()
        with pytest.raises(AttributeError):
            agg.transform(transaction_df)

    def test_works_inside_pipeline(self, transaction_df):
        """Must plug into sklearn Pipeline without errors."""
        pipe = Pipeline([("agg", TransactionAggregator())])
        result = pipe.fit_transform(transaction_df)
        assert "mean" in result.columns
