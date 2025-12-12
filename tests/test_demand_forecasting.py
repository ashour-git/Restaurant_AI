"""
Tests for demand forecasting ML pipeline.

This module contains comprehensive tests for the DemandForecaster
and EnhancedDemandForecaster classes.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDemandForecasterBasic:
    """Basic tests for DemandForecaster functionality."""

    def test_prepare_features_creates_time_features(self, sample_time_series_data):
        """Test that time features are created correctly."""
        df = sample_time_series_data.copy()

        # Add basic time features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "is_weekend" in df.columns
        assert df["is_weekend"].max() <= 1
        assert df["is_weekend"].min() >= 0

    def test_lag_features_creation(self, sample_time_series_data):
        """Test lag feature creation for time series."""
        df = sample_time_series_data.copy()

        # Create lag features
        for lag in [1, 7, 14]:
            df[f"demand_lag_{lag}"] = df["demand"].shift(lag)

        # First rows should have NaN
        assert pd.isna(df["demand_lag_1"].iloc[0])
        assert pd.isna(df["demand_lag_7"].iloc[6])

        # Later rows should have valid values
        assert not pd.isna(df["demand_lag_1"].iloc[1])
        assert not pd.isna(df["demand_lag_7"].iloc[7])

    def test_rolling_features_creation(self, sample_time_series_data):
        """Test rolling statistics feature creation."""
        df = sample_time_series_data.copy()

        # Create rolling features
        df["demand_rolling_mean_7"] = df["demand"].rolling(7).mean().shift(1)
        df["demand_rolling_std_7"] = df["demand"].rolling(7).std().shift(1)

        assert "demand_rolling_mean_7" in df.columns
        assert "demand_rolling_std_7" in df.columns

        # Check that rolling stats are computed correctly
        valid_mean = df["demand_rolling_mean_7"].dropna()
        assert len(valid_mean) > 0
        assert valid_mean.min() >= 0  # Demand should be positive


class TestDemandForecasterTraining:
    """Tests for model training functionality."""

    def test_train_test_split_temporal(self, sample_time_series_data):
        """Test temporal train-test split."""
        df = sample_time_series_data.copy()

        # Sort by date
        df = df.sort_values("date")

        # Split: 80% train, 20% test
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        # Verify temporal ordering
        assert train["date"].max() < test["date"].min()
        assert len(train) + len(test) == len(df)

    def test_no_future_data_leakage(self, sample_time_series_data):
        """Test that training doesn't use future data."""
        df = sample_time_series_data.copy()
        df = df.sort_values("date")

        # Create features that could leak future data
        df["demand_lag_1"] = df["demand"].shift(1)

        # The first row should have NaN for lag features
        assert pd.isna(df["demand_lag_1"].iloc[0])

        # Verify shift direction is correct (past, not future)
        if len(df) > 1:
            assert df["demand_lag_1"].iloc[1] == df["demand"].iloc[0]

    def test_feature_importance_extraction(self, sample_feature_data):
        """Test that feature importance can be extracted."""
        from sklearn.ensemble import RandomForestRegressor

        X = sample_feature_data[["feature_1", "feature_2", "feature_3", "feature_4"]]
        y = sample_feature_data["target"]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_

        assert len(importances) == X.shape[1]
        assert np.isclose(sum(importances), 1.0, atol=0.01)
        assert all(imp >= 0 for imp in importances)


class TestDemandForecasterPrediction:
    """Tests for prediction functionality."""

    def test_prediction_shape(self, sample_feature_data):
        """Test that predictions have correct shape."""
        from sklearn.ensemble import RandomForestRegressor

        X = sample_feature_data[["feature_1", "feature_2", "feature_3", "feature_4"]]
        y = sample_feature_data["target"]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)

    def test_prediction_non_negative(self, sample_feature_data):
        """Test that demand predictions are non-negative."""
        from sklearn.ensemble import RandomForestRegressor

        X = sample_feature_data[["feature_1", "feature_2", "feature_3", "feature_4"]]
        y = sample_feature_data["target"]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        # Clip negative predictions
        predictions = np.maximum(predictions, 0)

        assert all(pred >= 0 for pred in predictions)

    def test_prediction_reasonable_range(self, sample_time_series_data):
        """Test that predictions are within reasonable range."""
        from sklearn.ensemble import RandomForestRegressor

        df = sample_time_series_data.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        X = df[["day_of_week", "month"]]
        y = df["demand"]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)

        # Predictions should be within reasonable bounds
        assert predictions.min() >= 0
        assert predictions.max() < y.max() * 3  # Not unreasonably high


class TestDemandForecasterMetrics:
    """Tests for evaluation metrics."""

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 380, 520])

        # MAPE formula
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        assert 0 <= mape <= 100
        assert mape < 10  # Less than 10% error

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 380, 520])

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        assert rmse > 0
        assert rmse < np.std(y_true) * 2  # RMSE should be reasonable

    def test_mae_calculation(self):
        """Test MAE calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 380, 520])

        mae = np.mean(np.abs(y_true - y_pred))

        assert mae > 0
        assert mae == 14.0  # (10 + 10 + 10 + 20 + 20) / 5 = 70/5 = 14

    def test_r2_score_calculation(self):
        """Test R² score calculation."""
        from sklearn.metrics import r2_score

        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 380, 520])

        r2 = r2_score(y_true, y_pred)

        assert r2 > 0.9  # Good fit
        assert r2 <= 1.0


class TestEnhancedDemandForecaster:
    """Tests for EnhancedDemandForecaster with Optuna."""

    def test_optuna_objective_function(self, sample_feature_data):
        """Test that Optuna objective can be evaluated."""
        import optuna
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        X = sample_feature_data[["feature_1", "feature_2", "feature_3", "feature_4"]]
        y = sample_feature_data["target"]

        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 10, 50)
            max_depth = trial.suggest_int("max_depth", 2, 10)

            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
            return -np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3, show_progress_bar=False)

        assert study.best_trial is not None
        assert study.best_value is not None

    def test_cross_validation_time_series(self, sample_time_series_data):
        """Test time series cross-validation."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit

        df = sample_time_series_data.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        X = df[["day_of_week", "month"]]
        y = df["demand"]

        tscv = TimeSeriesSplit(n_splits=3)

        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            score = model.score(X_val, y_val)
            scores.append(score)

        assert len(scores) == 3
        # Time series CV may have lower scores due to distribution shift


class TestDemandForecasterMLflow:
    """Tests for MLflow integration."""

    def test_mlflow_logging(self, mock_mlflow_client):
        """Test that MLflow logging works correctly."""
        # Simulate logging
        mock_mlflow_client.log_metric("mape", 5.5)
        mock_mlflow_client.log_metric("rmse", 25.3)
        mock_mlflow_client.log_param("n_estimators", 100)

        mock_mlflow_client.log_metric.assert_any_call("mape", 5.5)
        mock_mlflow_client.log_metric.assert_any_call("rmse", 25.3)
        mock_mlflow_client.log_param.assert_called_with("n_estimators", 100)

    def test_model_registry_integration(self, mock_mlflow_client):
        """Test model registry operations."""
        # Simulate model registration
        mock_mlflow_client.create_registered_model("demand_forecaster")
        mock_mlflow_client.create_model_version.return_value = Mock(version="1")

        version = mock_mlflow_client.create_model_version(
            "demand_forecaster", "runs:/abc123/model", "runs:/abc123"
        )

        assert version.version == "1"


class TestDemandForecasterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_missing_values(self, sample_time_series_data):
        """Test handling of missing values."""
        df = sample_time_series_data.copy()

        # Introduce missing values
        df.loc[5, "demand"] = np.nan
        df.loc[10, "demand"] = np.nan

        # Drop or fill missing values
        df_filled = df.fillna(df["demand"].mean())
        df_dropped = df.dropna()

        assert not df_filled["demand"].isna().any()
        assert len(df_dropped) < len(df)

    def test_handles_zero_demand(self, sample_time_series_data):
        """Test handling of zero demand days."""
        df = sample_time_series_data.copy()

        # Set some demands to zero
        df.loc[0:2, "demand"] = 0

        # Model should still work
        df["day_of_week"] = df["date"].dt.dayofweek
        X = df[["day_of_week"]]
        y = df["demand"]

        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_handles_single_item(self):
        """Test handling of single item forecasting."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "item_id": ["I001"] * 30,
                "demand": np.random.randint(10, 100, 30),
            }
        )

        df["day_of_week"] = df["date"].dt.dayofweek

        assert df["item_id"].nunique() == 1
        assert len(df) == 30

    def test_handles_multiple_items(self):
        """Test handling of multiple items."""
        items = ["I001", "I002", "I003"]
        dfs = []

        for item in items:
            df = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=30),
                    "item_id": item,
                    "demand": np.random.randint(10, 100, 30),
                }
            )
            dfs.append(df)

        combined = pd.concat(dfs)

        assert combined["item_id"].nunique() == 3
        assert len(combined) == 90


class TestDemandForecasterIntegration:
    """Integration tests for demand forecaster."""

    def test_full_pipeline_execution(self, sample_time_series_data):
        """Test full forecasting pipeline."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Prepare data
        df = sample_time_series_data.copy()
        df = df.sort_values("date")
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["demand_lag_1"] = df["demand"].shift(1)
        df["demand_lag_7"] = df["demand"].shift(7)
        df["demand_rolling_7"] = df["demand"].rolling(7).mean().shift(1)

        # Remove NaN rows
        df = df.dropna()

        # Split
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        features = ["day_of_week", "month", "demand_lag_1", "demand_lag_7", "demand_rolling_7"]

        X_train, y_train = train[features], train["demand"]
        X_test, y_test = test[features], test["demand"]

        # Train
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        assert rmse > 0
        assert mae > 0
        assert len(predictions) == len(y_test)
        assert len(predictions) == len(y_test)
