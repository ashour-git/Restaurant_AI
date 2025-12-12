"""
ML utilities for the Restaurant SaaS platform.

Provides common functions for data loading, preprocessing, and model management.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
EMBEDDINGS_DIR = PROJECT_ROOT / "ml" / "embeddings"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_transactions() -> pd.DataFrame:
    """
    Load transactions dataset.

    Returns:
        pd.DataFrame: Transactions with parsed timestamps.

    Example:
        >>> df = load_transactions()
        >>> df.columns.tolist()
        ['transaction_id', 'order_id', 'customer_id', ...]
    """
    df = pd.read_csv(RAW_DIR / "transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_menu_items() -> pd.DataFrame:
    """
    Load menu items dataset.

    Returns:
        pd.DataFrame: Menu items.

    Example:
        >>> df = load_menu_items()
        >>> df.columns.tolist()
        ['item_id', 'item_name', 'description', ...]
    """
    df = pd.read_csv(RAW_DIR / "menu_items.csv")
    # Rename 'name' to 'item_name' for consistency
    if "name" in df.columns and "item_name" not in df.columns:
        df = df.rename(columns={"name": "item_name"})
    df["allergens"] = df["allergens"].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    return df


def load_customers() -> pd.DataFrame:
    """
    Load customers dataset.

    Returns:
        pd.DataFrame: Customer profiles.

    Example:
        >>> df = load_customers()
        >>> df.columns.tolist()
        ['customer_id', 'first_name', 'last_name', ...]
    """
    df = pd.read_csv(RAW_DIR / "customers.csv")
    df["dietary_preferences"] = df["dietary_preferences"].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )
    return df


def load_inventory() -> pd.DataFrame:
    """
    Load inventory dataset.

    Returns:
        pd.DataFrame: Inventory items.

    Example:
        >>> df = load_inventory()
        >>> df.columns.tolist()
        ['inventory_id', 'ingredient_name', 'category', ...]
    """
    return pd.read_csv(RAW_DIR / "inventory.csv")


def load_cooccurrence() -> pd.DataFrame:
    """
    Load item co-occurrence dataset.

    Returns:
        pd.DataFrame: Item co-occurrence pairs with metrics.

    Example:
        >>> df = load_cooccurrence()
        >>> df.columns.tolist()
        ['item_id_1', 'item_id_2', 'cooccurrence_count', ...]
    """
    return pd.read_csv(RAW_DIR / "cooccurrence.csv")


def load_daily_aggregates() -> pd.DataFrame:
    """
    Load daily sales aggregates.

    Returns:
        pd.DataFrame: Daily aggregated sales data.

    Example:
        >>> df = load_daily_aggregates()
        >>> df.columns.tolist()
        ['date', 'total_revenue', 'total_orders', ...]
    """
    df = pd.read_csv(PROCESSED_DIR / "daily_aggregates.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_item_daily_sales() -> pd.DataFrame:
    """
    Load item-level daily sales.

    Returns:
        pd.DataFrame: Item-daily sales data.

    Example:
        >>> df = load_item_daily_sales()
        >>> df.columns.tolist()
        ['date', 'item_id', 'item_name', ...]
    """
    df = pd.read_csv(PROCESSED_DIR / "item_daily_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Create time-based features from a date column.

    Args:
        df: DataFrame with date column.
        date_col: Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with additional time features.

    Example:
        >>> df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=5)})
        >>> df = create_time_features(df)
        >>> 'day_of_week' in df.columns
        True
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] = [1, 7, 14, 28],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.

    Args:
        df: DataFrame with target column.
        target_col: Name of the target column.
        lags: List of lag periods.
        group_col: Column to group by (for item-level lags).

    Returns:
        pd.DataFrame: DataFrame with lag features.

    Example:
        >>> df = pd.DataFrame({'value': range(10)})
        >>> df = create_lag_features(df, 'value', lags=[1, 2])
        >>> 'value_lag_1' in df.columns
        True
    """
    df = df.copy()

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        if group_col:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int] = [7, 14, 28],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Create rolling window features for time series forecasting.

    Args:
        df: DataFrame with target column.
        target_col: Name of the target column.
        windows: List of rolling window sizes.
        group_col: Column to group by.

    Returns:
        pd.DataFrame: DataFrame with rolling features.

    Example:
        >>> df = pd.DataFrame({'value': range(30)})
        >>> df = create_rolling_features(df, 'value', windows=[7])
        >>> 'value_rolling_mean_7' in df.columns
        True
    """
    df = df.copy()

    for window in windows:
        if group_col:
            grouped = df.groupby(group_col)[target_col]
            df[f"{target_col}_rolling_mean_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        else:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].shift(1).rolling(window, min_periods=1).std()
            )

    return df


def save_model(model: Any, name: str, metadata: dict | None = None) -> Path:
    """
    Save a trained model to disk.

    Args:
        model: Trained model object.
        name: Model name (without extension).
        metadata: Optional metadata dictionary.

    Returns:
        Path: Path to saved model.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> path = save_model(model, "test_model")
    """
    model_path = MODELS_DIR / f"{name}.pkl"

    model_data = {
        "model": model,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat(),
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")
    return model_path


def load_model(name: str) -> tuple[Any, dict]:
    """
    Load a trained model from disk.

    Args:
        name: Model name (without extension).

    Returns:
        Tuple[Any, Dict]: Model object and metadata.

    Raises:
        FileNotFoundError: If model file doesn't exist.

    Example:
        >>> model, metadata = load_model("demand_forecast")
    """
    model_path = MODELS_DIR / f"{name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["model"], model_data.get("metadata", {})


def save_embeddings(
    embeddings: np.ndarray,
    ids: list[str],
    name: str,
    metadata: dict | None = None,
) -> Path:
    """
    Save embeddings to disk.

    Args:
        embeddings: Embedding matrix (n_items x embedding_dim).
        ids: List of item IDs corresponding to rows.
        name: Embedding name.
        metadata: Optional metadata dictionary.

    Returns:
        Path: Path to saved embeddings.

    Example:
        >>> embeddings = np.random.randn(10, 64)
        >>> ids = [f"item_{i}" for i in range(10)]
        >>> path = save_embeddings(embeddings, ids, "test_embeddings")
    """
    embedding_path = EMBEDDINGS_DIR / f"{name}.npz"

    np.savez(
        embedding_path,
        embeddings=embeddings,
        ids=np.array(ids),
        metadata=json.dumps(metadata or {}),
    )

    print(f"Embeddings saved to {embedding_path}")
    return embedding_path


def load_embeddings(name: str) -> tuple[np.ndarray, list[str], dict]:
    """
    Load embeddings from disk.

    Args:
        name: Embedding name.

    Returns:
        Tuple[np.ndarray, List[str], Dict]: Embeddings, IDs, and metadata.

    Example:
        >>> embeddings, ids, metadata = load_embeddings("item_embeddings")
    """
    embedding_path = EMBEDDINGS_DIR / f"{name}.npz"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embedding_path}")

    data = np.load(embedding_path, allow_pickle=True)

    embeddings = data["embeddings"]
    ids = data["ids"].tolist()
    metadata = json.loads(str(data["metadata"]))

    return embeddings, ids, metadata


def train_test_split_time_series(
    df: pd.DataFrame,
    date_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data maintaining temporal order.

    Args:
        df: DataFrame with date column.
        date_col: Name of date column.
        test_size: Proportion of data for testing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.

    Example:
        >>> df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=100)})
        >>> train, test = train_test_split_time_series(df, 'date', 0.2)
        >>> len(test) / len(df)
        0.2
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test
