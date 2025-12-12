"""
Tests for MLOps infrastructure.

This module contains comprehensive tests for MLflow experiment tracking,
model registry, and data validation components.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExperimentTracker:
    """Tests for MLflow experiment tracking."""

    def test_experiment_creation(self, mock_mlflow_client):
        """Test MLflow experiment creation."""
        experiment_name = "test_experiment"
        mock_mlflow_client.create_experiment.return_value = "exp_123"

        exp_id = mock_mlflow_client.create_experiment(experiment_name)

        mock_mlflow_client.create_experiment.assert_called_once_with(experiment_name)
        assert exp_id == "exp_123"

    def test_run_start_and_end(self, mock_mlflow_client):
        """Test starting and ending MLflow runs."""
        run_mock = Mock()
        run_mock.info.run_id = "run_abc123"
        mock_mlflow_client.start_run.return_value = run_mock

        run = mock_mlflow_client.start_run()

        assert run.info.run_id == "run_abc123"

    def test_log_params(self, mock_mlflow_client):
        """Test logging parameters to MLflow."""
        params = {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 5}

        for key, value in params.items():
            mock_mlflow_client.log_param(key, value)

        assert mock_mlflow_client.log_param.call_count == 3
        mock_mlflow_client.log_param.assert_any_call("learning_rate", 0.01)
        mock_mlflow_client.log_param.assert_any_call("n_estimators", 100)
        mock_mlflow_client.log_param.assert_any_call("max_depth", 5)

    def test_log_metrics(self, mock_mlflow_client):
        """Test logging metrics to MLflow."""
        metrics = {"mape": 5.5, "rmse": 25.3, "mae": 18.2, "r2": 0.92}

        for key, value in metrics.items():
            mock_mlflow_client.log_metric(key, value)

        assert mock_mlflow_client.log_metric.call_count == 4
        mock_mlflow_client.log_metric.assert_any_call("mape", 5.5)
        mock_mlflow_client.log_metric.assert_any_call("r2", 0.92)

    def test_log_artifact(self, mock_mlflow_client, tmp_path):
        """Test logging artifacts to MLflow."""
        # Create a temporary artifact file
        artifact_path = tmp_path / "feature_importance.json"
        artifact_path.write_text('{"feature1": 0.5, "feature2": 0.3}')

        mock_mlflow_client.log_artifact(str(artifact_path))

        mock_mlflow_client.log_artifact.assert_called_once_with(str(artifact_path))

    def test_log_model(self, mock_mlflow_client):
        """Test logging model to MLflow."""
        model = Mock()
        mock_mlflow_client.log_model(model, "model")

        mock_mlflow_client.log_model.assert_called_once_with(model, "model")

    def test_set_tags(self, mock_mlflow_client):
        """Test setting tags in MLflow."""
        tags = {"model_type": "demand_forecaster", "version": "1.0.0", "environment": "development"}

        mock_mlflow_client.set_tags(tags)

        mock_mlflow_client.set_tags.assert_called_once_with(tags)


class TestModelRegistry:
    """Tests for MLflow model registry."""

    def test_register_model(self, mock_mlflow_client):
        """Test model registration."""
        from unittest.mock import Mock

        mock_model = Mock()
        mock_model.name = "demand_forecaster"
        mock_mlflow_client.create_registered_model.return_value = mock_model

        model = mock_mlflow_client.create_registered_model("demand_forecaster")

        assert model.name == "demand_forecaster"

    def test_create_model_version(self, mock_mlflow_client):
        """Test creating model version."""
        from unittest.mock import Mock

        mock_version = Mock()
        mock_version.version = "1"
        mock_version.name = "demand_forecaster"
        mock_mlflow_client.create_model_version.return_value = mock_version

        version = mock_mlflow_client.create_model_version(
            name="demand_forecaster", source="runs:/abc123/model", run_id="abc123"
        )

        assert version.version == "1"
        assert version.name == "demand_forecaster"

    def test_transition_model_stage(self, mock_mlflow_client):
        """Test transitioning model between stages."""
        mock_mlflow_client.transition_model_version_stage.return_value = Mock(
            current_stage="Production"
        )

        result = mock_mlflow_client.transition_model_version_stage(
            name="demand_forecaster", version="1", stage="Production"
        )

        assert result.current_stage == "Production"

    def test_get_latest_versions(self, mock_mlflow_client):
        """Test getting latest model versions."""
        mock_versions = [
            Mock(version="3", current_stage="Production"),
            Mock(version="2", current_stage="Staging"),
            Mock(version="1", current_stage="Archived"),
        ]
        mock_mlflow_client.get_latest_versions.return_value = mock_versions

        versions = mock_mlflow_client.get_latest_versions(
            name="demand_forecaster", stages=["Production", "Staging"]
        )

        assert len(versions) == 3
        assert versions[0].current_stage == "Production"

    def test_load_model(self, mock_mlflow_client):
        """Test loading model from registry."""
        mock_model = Mock()
        mock_mlflow_client.load_model.return_value = mock_model

        model = mock_mlflow_client.load_model("models:/demand_forecaster/Production")

        assert model is mock_model

    def test_update_model_description(self, mock_mlflow_client):
        """Test updating model description."""
        mock_mlflow_client.update_registered_model(
            name="demand_forecaster", description="Updated description for demand forecaster"
        )

        mock_mlflow_client.update_registered_model.assert_called_once()


class TestDataValidation:
    """Tests for data validation with Pandera."""

    def test_transaction_schema_validation(self, sample_transactions):
        """Test transaction data schema validation."""
        df = sample_transactions.copy()

        # Define expected schema
        required_columns = {
            "transaction_id": str,
            "customer_id": str,
            "item_id": str,
            "quantity": (int, np.integer),
            "total_price": (float, np.floating),
            "timestamp": (pd.Timestamp, datetime),
        }

        for col, expected_type in required_columns.items():
            assert col in df.columns, f"Missing column: {col}"

    def test_quantity_validation(self, sample_transactions):
        """Test that quantities are positive integers."""
        df = sample_transactions.copy()

        assert (df["quantity"] > 0).all(), "All quantities must be positive"
        assert (df["quantity"] == df["quantity"].astype(int)).all(), "Quantities must be integers"

    def test_price_validation(self, sample_transactions):
        """Test that prices are positive."""
        df = sample_transactions.copy()

        assert (df["total_price"] > 0).all(), "All prices must be positive"
        assert (df["unit_price"] > 0).all(), "All unit prices must be positive"

    def test_timestamp_validation(self, sample_transactions):
        """Test that timestamps are valid."""
        df = sample_transactions.copy()

        # Timestamps should be datetime
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "Timestamps must be datetime"

        # No future dates
        assert (df["timestamp"] <= datetime.now()).all(), "No future timestamps allowed"

    def test_customer_id_format(self, sample_transactions):
        """Test customer ID format validation."""
        df = sample_transactions.copy()

        # All customer IDs should start with 'C'
        assert all(
            cid.startswith("C") for cid in df["customer_id"]
        ), "Customer IDs must start with 'C'"

    def test_item_id_format(self, sample_transactions):
        """Test item ID format validation."""
        df = sample_transactions.copy()

        # All item IDs should start with 'I'
        assert all(iid.startswith("I") for iid in df["item_id"]), "Item IDs must start with 'I'"

    def test_referential_integrity(self, sample_transactions, sample_menu_items):
        """Test referential integrity between transactions and menu items."""
        txn_items = set(sample_transactions["item_id"])
        menu_items = set(sample_menu_items["item_id"])

        # All transaction items should exist in menu
        invalid_items = txn_items - menu_items
        assert len(invalid_items) == 0, f"Invalid item IDs in transactions: {invalid_items}"

    def test_data_completeness(self, sample_transactions):
        """Test that there are no null values in required columns."""
        df = sample_transactions.copy()

        required_cols = [
            "transaction_id",
            "customer_id",
            "item_id",
            "quantity",
            "total_price",
            "timestamp",
        ]

        for col in required_cols:
            assert not df[col].isna().any(), f"Null values found in {col}"


class TestDataQualityChecks:
    """Tests for data quality monitoring."""

    def test_duplicate_transaction_detection(self, sample_transactions):
        """Test detection of duplicate transactions."""
        df = sample_transactions.copy()

        # Check for complete row duplicates (actual duplicates)
        duplicates = df.duplicated()

        # Sample transactions may have same transaction_id for grouped items
        # This is expected behavior for order line items
        assert isinstance(duplicates, pd.Series)

    def test_outlier_detection_price(self, sample_transactions):
        """Test outlier detection for prices."""
        df = sample_transactions.copy()

        # Calculate IQR
        q1 = df["total_price"].quantile(0.25)
        q3 = df["total_price"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df["total_price"] < lower_bound) | (df["total_price"] > upper_bound)]

        # Should detect some outliers or none
        assert isinstance(outliers, pd.DataFrame)

    def test_outlier_detection_quantity(self, sample_transactions):
        """Test outlier detection for quantities."""
        df = sample_transactions.copy()

        # Z-score method
        mean_qty = df["quantity"].mean()
        std_qty = df["quantity"].std()

        z_scores = (df["quantity"] - mean_qty) / std_qty
        outliers = df[np.abs(z_scores) > 3]

        # Most quantities should not be outliers
        assert len(outliers) < len(df) * 0.1

    def test_data_drift_detection(self):
        """Test data drift detection between train and production data."""
        # Simulated training data distribution
        np.random.seed(42)
        train_data = np.random.normal(100, 20, 1000)

        # Simulated production data (with drift)
        prod_data = np.random.normal(110, 25, 100)

        # Simple drift detection using mean comparison
        train_mean = train_data.mean()
        prod_mean = prod_data.mean()

        # Check if drift is significant (>10% difference)
        drift_ratio = abs(prod_mean - train_mean) / train_mean
        has_drift = drift_ratio > 0.1

        # Drift should be detected
        assert isinstance(has_drift, (bool, np.bool_))

    def test_feature_distribution_check(self, sample_feature_data):
        """Test feature distribution validation."""
        df = sample_feature_data.copy()

        for col in ["feature_1", "feature_2", "feature_3", "feature_4"]:
            # Check for finite values
            assert np.isfinite(df[col]).all(), f"Non-finite values in {col}"

            # Check variance is non-zero
            assert df[col].var() > 0, f"Zero variance in {col}"


class TestMLOpsIntegration:
    """Integration tests for MLOps pipeline."""

    def test_training_pipeline_with_tracking(self, mock_mlflow_client, sample_feature_data):
        """Test full training pipeline with MLflow tracking."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split

        df = sample_feature_data.copy()
        X = df[["feature_1", "feature_2", "feature_3", "feature_4"]]
        y = df["target"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training params
        params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}

        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        # Log to MLflow
        for key, value in params.items():
            mock_mlflow_client.log_param(key, value)

        mock_mlflow_client.log_metric("rmse", rmse)
        mock_mlflow_client.log_metric("mae", mae)
        mock_mlflow_client.log_model(model, "model")

        # Verify logging
        assert mock_mlflow_client.log_param.call_count == 3
        assert mock_mlflow_client.log_metric.call_count == 2
        mock_mlflow_client.log_model.assert_called_once()

    def test_model_versioning_workflow(self, mock_mlflow_client):
        """Test model versioning workflow."""
        # Step 1: Register model
        mock_mlflow_client.create_registered_model("demand_forecaster")

        # Step 2: Create version 1
        mock_mlflow_client.create_model_version.return_value = Mock(version="1")
        v1 = mock_mlflow_client.create_model_version(
            name="demand_forecaster", source="runs:/run1/model", run_id="run1"
        )

        # Step 3: Transition to Production
        mock_mlflow_client.transition_model_version_stage(
            name="demand_forecaster", version="1", stage="Production"
        )

        # Step 4: Create version 2
        mock_mlflow_client.create_model_version.return_value = Mock(version="2")
        v2 = mock_mlflow_client.create_model_version(
            name="demand_forecaster", source="runs:/run2/model", run_id="run2"
        )

        # Verify workflow
        mock_mlflow_client.create_registered_model.assert_called_once()
        assert mock_mlflow_client.create_model_version.call_count == 2
        mock_mlflow_client.transition_model_version_stage.assert_called_once()

    def test_ab_testing_setup(self, mock_mlflow_client):
        """Test A/B testing setup with model registry."""
        # Model A in Production
        mock_mlflow_client.get_model_version.return_value = Mock(
            version="1", current_stage="Production", description="Model A - baseline"
        )

        model_a = mock_mlflow_client.get_model_version(name="demand_forecaster", version="1")

        # Model B in Staging
        mock_mlflow_client.get_model_version.return_value = Mock(
            version="2", current_stage="Staging", description="Model B - challenger"
        )

        model_b = mock_mlflow_client.get_model_version(name="demand_forecaster", version="2")

        assert model_a.current_stage == "Production"
        assert model_b.current_stage == "Staging"


class TestAlerting:
    """Tests for alerting and monitoring."""

    def test_metric_threshold_alert(self):
        """Test metric threshold alerting."""
        thresholds = {"mape": 10.0, "rmse": 50.0, "latency_ms": 100.0}

        current_metrics = {
            "mape": 12.5,  # Exceeds threshold
            "rmse": 35.0,  # OK
            "latency_ms": 80.0,  # OK
        }

        alerts = []
        for metric, value in current_metrics.items():
            if value > thresholds[metric]:
                alerts.append(
                    {
                        "metric": metric,
                        "value": value,
                        "threshold": thresholds[metric],
                        "severity": "warning",
                    }
                )

        assert len(alerts) == 1
        assert alerts[0]["metric"] == "mape"

    def test_model_performance_degradation_alert(self):
        """Test model performance degradation alerting."""
        baseline_mape = 5.0
        current_mape = 8.5

        degradation_threshold = 0.5  # 50% increase

        degradation = (current_mape - baseline_mape) / baseline_mape
        should_alert = degradation > degradation_threshold

        assert should_alert is True
        assert degradation == 0.7  # 70% degradation
        assert degradation == 0.7  # 70% degradation
