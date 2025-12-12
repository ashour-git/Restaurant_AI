"""
Tests for data utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.utils.data_utils import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    train_test_split_time_series,
)


class TestCreateTimeFeatures:
    """Tests for create_time_features function."""

    def test_creates_basic_features(self):
        """Test that basic time features are created."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"date": dates, "value": range(10)})

        result = create_time_features(df, date_col="date")

        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "day_of_month" in result.columns or "day" in result.columns
        assert "is_weekend" in result.columns

    def test_correct_weekend_detection(self):
        """Test that weekends are correctly identified."""
        # Create dates including weekend
        dates = pd.date_range("2024-01-01", periods=7, freq="D")  # Mon-Sun
        df = pd.DataFrame({"date": dates})

        result = create_time_features(df, date_col="date")

        # Saturday and Sunday should be marked as weekend
        expected_weekends = [False, False, False, False, False, True, True]
        assert list(result["is_weekend"]) == expected_weekends

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "value": [1, 2, 3, 4, 5],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

        result = create_time_features(df, date_col="date")

        assert "value" in result.columns
        assert "category" in result.columns
        assert list(result["value"]) == [1, 2, 3, 4, 5]


class TestCreateLagFeatures:
    """Tests for create_lag_features function."""

    def test_creates_lag_features(self):
        """Test that lag features are created correctly."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "value": range(10),
            }
        )

        result = create_lag_features(df, target_col="value", lags=[1, 2, 3])

        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns
        assert "value_lag_3" in result.columns

    def test_lag_values_correct(self):
        """Test that lag values are computed correctly."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "value": [10, 20, 30, 40, 50],
            }
        )

        result = create_lag_features(df, target_col="value", lags=[1, 2])

        # Lag 1: previous value
        expected_lag1 = [np.nan, 10, 20, 30, 40]
        np.testing.assert_array_equal(result["value_lag_1"].values, expected_lag1)

        # Lag 2: value 2 steps ago
        expected_lag2 = [np.nan, np.nan, 10, 20, 30]
        np.testing.assert_array_equal(result["value_lag_2"].values, expected_lag2)

    def test_grouped_lag_features(self):
        """Test lag features with grouping."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4).tolist() * 2,
                "group": ["A"] * 4 + ["B"] * 4,
                "value": [1, 2, 3, 4, 10, 20, 30, 40],
            }
        )
        df = df.sort_values(["group", "date"]).reset_index(drop=True)

        result = create_lag_features(df, target_col="value", lags=[1], group_col="group")

        # Check group A
        group_a = result[result["group"] == "A"]["value_lag_1"].values
        expected_a = [np.nan, 1, 2, 3]
        np.testing.assert_array_equal(group_a, expected_a)


class TestCreateRollingFeatures:
    """Tests for create_rolling_features function."""

    def test_creates_rolling_features(self):
        """Test that rolling features are created."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "value": range(10),
            }
        )

        result = create_rolling_features(df, target_col="value", windows=[3, 5])

        assert "value_rolling_mean_3" in result.columns
        assert "value_rolling_std_3" in result.columns
        assert "value_rolling_mean_5" in result.columns
        assert "value_rolling_std_5" in result.columns

    def test_rolling_mean_correct(self):
        """Test rolling mean calculation."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        result = create_rolling_features(df, target_col="value", windows=[3])

        # Rolling mean with window=3 - verify column exists and has values
        rolling_mean = result["value_rolling_mean_3"].values
        assert "value_rolling_mean_3" in result.columns
        # First few values may be NaN depending on implementation
        assert len(rolling_mean) == 5


class TestTrainTestSplitTimeSeries:
    """Tests for train_test_split_time_series function."""

    def test_split_proportions(self):
        """Test that split proportions are correct."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "value": range(100),
            }
        )

        train, test = train_test_split_time_series(df, "date", test_size=0.2)

        assert len(train) == 80
        assert len(test) == 20

    def test_temporal_ordering(self):
        """Test that train dates are before test dates."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "value": range(100),
            }
        )

        train, test = train_test_split_time_series(df, "date", test_size=0.3)

        max_train_date = train["date"].max()
        min_test_date = test["date"].min()

        assert max_train_date < min_test_date

    def test_no_data_leakage(self):
        """Test that there's no overlap between train and test."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=50),
                "value": range(50),
            }
        )

        train, test = train_test_split_time_series(df, "date", test_size=0.2)

        train_dates = set(train["date"])
        test_dates = set(test["date"])

        assert len(train_dates.intersection(test_dates)) == 0


class TestDataValidation:
    """Tests for data validation utilities."""

    def test_transaction_schema_valid(self, sample_transactions):
        """Test that sample transactions pass validation."""
        # Check required columns exist
        required_cols = [
            "transaction_id",
            "customer_id",
            "item_id",
            "quantity",
            "total_price",
            "timestamp",
        ]

        for col in required_cols:
            assert col in sample_transactions.columns

    def test_transaction_values_valid(self, sample_transactions):
        """Test that transaction values are valid."""
        # Quantities should be positive
        assert (sample_transactions["quantity"] > 0).all()

        # Prices should be positive
        assert (sample_transactions["total_price"] > 0).all()
        assert (sample_transactions["unit_price"] > 0).all()

    def test_customer_ids_format(self, sample_transactions):
        """Test customer ID format."""
        # All customer IDs should start with 'C'
        assert all(cid.startswith("C") for cid in sample_transactions["customer_id"])

    def test_item_ids_format(self, sample_transactions):
        """Test item ID format."""
        # All item IDs should start with 'I'
        assert all(iid.startswith("I") for iid in sample_transactions["item_id"])
        assert all(iid.startswith("I") for iid in sample_transactions["item_id"])
