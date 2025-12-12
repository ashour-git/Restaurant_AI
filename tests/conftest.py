"""
Pytest configuration and fixtures for Smart Restaurant tests.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_records = 1000
    n_customers = 100
    n_items = 20

    # Generate dates over 6 months
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 180)) for _ in range(n_records)]

    data = {
        "transaction_id": [f"T{i//3:04d}" for i in range(n_records)],
        "customer_id": [f"C{np.random.randint(1, n_customers+1):03d}" for _ in range(n_records)],
        "item_id": [f"I{np.random.randint(1, n_items+1):03d}" for _ in range(n_records)],
        "quantity": np.random.randint(1, 5, n_records),
        "unit_price": np.random.uniform(5, 50, n_records).round(2),
        "timestamp": dates,
    }

    df = pd.DataFrame(data)
    df["total_price"] = df["quantity"] * df["unit_price"]
    df["total_price"] = df["total_price"].round(2)

    return df


@pytest.fixture
def sample_customers():
    """Create sample customer data for testing."""
    n_customers = 100

    tiers = ["Bronze", "Silver", "Gold", "Platinum"]

    data = {
        "customer_id": [f"C{i:03d}" for i in range(1, n_customers + 1)],
        "name": [f"Customer {i}" for i in range(1, n_customers + 1)],
        "email": [f"customer{i}@example.com" for i in range(1, n_customers + 1)],
        "loyalty_tier": np.random.choice(tiers, n_customers),
        "created_at": [
            datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            for _ in range(n_customers)
        ],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_menu_items():
    """Create sample menu item data for testing."""
    categories = ["Appetizers", "Main Courses", "Desserts", "Beverages"]

    items = []
    for i in range(1, 21):
        category = categories[i % len(categories)]
        items.append(
            {
                "item_id": f"I{i:03d}",
                "name": f"Item {i}",
                "description": f"Delicious {category.lower()[:-1]} dish number {i}",
                "category": category,
                "price": round(np.random.uniform(5, 50), 2),
                "cost": round(np.random.uniform(2, 20), 2),
            }
        )

    return pd.DataFrame(items)


@pytest.fixture
def sample_daily_sales(sample_transactions):
    """Create sample daily aggregated sales data."""
    df = sample_transactions.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = (
        df.groupby("date")
        .agg(
            {
                "transaction_id": "nunique",
                "total_price": "sum",
                "quantity": "sum",
            }
        )
        .reset_index()
    )

    daily.columns = ["date", "total_orders", "total_revenue", "total_items"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


@pytest.fixture
def sample_item_daily_sales(sample_transactions):
    """Create sample item-level daily sales data."""
    df = sample_transactions.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = (
        df.groupby(["date", "item_id"])
        .agg(
            {
                "quantity": "sum",
                "total_price": "sum",
            }
        )
        .reset_index()
    )

    daily.columns = ["date", "item_id", "quantity_sold", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])

    return daily.sort_values(["date", "item_id"]).reset_index(drop=True)


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def rfm_features():
    """Create sample RFM features for testing."""
    n_customers = 50

    data = {
        "customer_id": [f"C{i:03d}" for i in range(1, n_customers + 1)],
        "recency_days": np.random.randint(1, 100, n_customers),
        "total_transactions": np.random.randint(1, 50, n_customers),
        "total_spent": np.random.uniform(50, 5000, n_customers).round(2),
        "avg_order_value": np.random.uniform(10, 100, n_customers).round(2),
        "tenure_days": np.random.randint(30, 365, n_customers),
    }

    return pd.DataFrame(data)


@pytest.fixture
def customer_features(sample_transactions, sample_customers):
    """Create customer features from transactions."""
    df = sample_transactions.copy()

    features = df.groupby("customer_id").agg(
        {
            "timestamp": ["min", "max"],
            "transaction_id": "nunique",
            "total_price": ["sum", "mean"],
            "quantity": "sum",
            "item_id": "nunique",
        }
    )

    features.columns = [
        "first_purchase",
        "last_purchase",
        "total_transactions",
        "total_spent",
        "avg_order_value",
        "total_items",
        "unique_items",
    ]

    features = features.reset_index()

    # Add derived features
    now = datetime.now()
    features["recency_days"] = (now - features["last_purchase"]).dt.days
    features["tenure_days"] = (now - features["first_purchase"]).dt.days
    features["orders_per_month"] = features["total_transactions"] / (features["tenure_days"] / 30)

    return features


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for data storage."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# ============================================================================
# Time Series Fixtures
# ============================================================================


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for demand forecasting tests."""
    np.random.seed(42)
    n_days = 90

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # Create demand with weekly seasonality
    base_demand = 100
    weekly_pattern = [1.2, 1.0, 0.9, 1.0, 1.3, 1.5, 1.4]  # Mon-Sun

    demand = []
    for i, date in enumerate(dates):
        day_factor = weekly_pattern[date.dayofweek]
        noise = np.random.normal(0, 10)
        demand.append(max(0, base_demand * day_factor + noise))

    return pd.DataFrame(
        {
            "date": dates,
            "demand": np.array(demand).round(0).astype(int),
            "item_id": "I001",
        }
    )


@pytest.fixture
def sample_rfm_data():
    """Create sample RFM data for customer analytics tests."""
    np.random.seed(42)
    n_customers = 50

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(1, n_customers + 1)],
            "recency": np.random.randint(1, 120, n_customers),
            "frequency": np.random.randint(1, 30, n_customers),
            "monetary": np.random.uniform(50, 2000, n_customers).round(2),
        }
    )


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for ML model tests."""
    np.random.seed(42)
    n_samples = 100

    # Create correlated features
    feature_1 = np.random.randn(n_samples)
    feature_2 = feature_1 * 0.5 + np.random.randn(n_samples) * 0.5
    feature_3 = np.random.randn(n_samples)
    feature_4 = np.random.randn(n_samples)

    # Target is a function of features
    target = 10 + 2 * feature_1 + 1.5 * feature_2 + np.random.randn(n_samples) * 0.5

    return pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_3": feature_3,
            "feature_4": feature_4,
            "target": target,
        }
    )


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client for testing."""
    from unittest.mock import Mock

    client = Mock()
    client.create_experiment = Mock(return_value="exp_123")
    client.start_run = Mock()
    client.log_param = Mock()
    client.log_metric = Mock()
    client.log_artifact = Mock()
    client.log_model = Mock()
    client.set_tags = Mock()
    client.create_registered_model = Mock()
    client.create_model_version = Mock()
    client.transition_model_version_stage = Mock()
    client.get_latest_versions = Mock()
    client.load_model = Mock()
    client.update_registered_model = Mock()
    client.get_model_version = Mock()

    return client


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "ml: marks tests that require ML models")
