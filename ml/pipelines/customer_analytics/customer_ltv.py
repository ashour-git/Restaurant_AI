"""
Customer Lifetime Value (CLV) Prediction Model.

Predicts the future value of customers using probabilistic models
and machine learning approaches.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ml.utils.data_utils import (
    MODELS_DIR,
    load_customers,
    load_model,
    load_transactions,
    save_model,
)

logger = logging.getLogger(__name__)


class CustomerLTV:
    """
    Customer Lifetime Value Prediction Model.

    Uses a combination of RFM features and machine learning to
    predict customer lifetime value. Supports both probabilistic
    (BG/NBD + Gamma-Gamma) and ML-based approaches.

    Attributes:
        model: Trained regression model.
        scaler: StandardScaler for features.
        prediction_horizon_days: Days to predict forward.
        feature_cols: List of feature column names.

    Example:
        >>> ltv_model = CustomerLTV()
        >>> ltv_model.train(transactions_df)
        >>> predictions = ltv_model.predict(customer_features)
        >>> high_value = ltv_model.get_high_value_customers(transactions_df)
    """

    def __init__(
        self,
        prediction_horizon_days: int = 365,
        model_params: dict | None = None,
    ) -> None:
        """
        Initialize the LTV model.

        Args:
            prediction_horizon_days: Days to predict future value.
            model_params: Optional GradientBoosting parameters.
        """
        self.prediction_horizon_days = prediction_horizon_days
        self.model_params = model_params or {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        }

        self.model: GradientBoostingRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.trained_at: datetime | None = None
        self.feature_importance_: pd.DataFrame | None = None

    def _create_features(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        analysis_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Create customer features for LTV prediction.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.
            analysis_date: Reference date for calculations.

        Returns:
            DataFrame with customer features.
        """
        analysis_date = analysis_date or datetime.now()
        df = transactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to valid customers
        df = df[df["customer_id"].notna()]

        # Core RFM features
        features = df.groupby("customer_id").agg(
            {
                "timestamp": ["min", "max", "count"],
                "transaction_id": "nunique",
                "total_price": ["sum", "mean", "std", "max", "min"],
                "quantity": ["sum", "mean"],
                "item_id": "nunique",
            }
        )

        # Flatten columns
        features.columns = [
            "first_purchase",
            "last_purchase",
            "total_records",
            "total_transactions",
            "total_revenue",
            "avg_order_value",
            "std_order_value",
            "max_order_value",
            "min_order_value",
            "total_items",
            "avg_items_per_order",
            "unique_items",
        ]
        features = features.reset_index()

        # Calculate time-based features
        features["tenure_days"] = (analysis_date - features["first_purchase"]).dt.days
        features["recency_days"] = (analysis_date - features["last_purchase"]).dt.days
        features["purchase_span"] = (features["last_purchase"] - features["first_purchase"]).dt.days

        # Frequency features
        features["orders_per_month"] = np.where(
            features["tenure_days"] > 0,
            features["total_transactions"] / (features["tenure_days"] / 30),
            features["total_transactions"],
        )

        features["avg_days_between_orders"] = np.where(
            features["total_transactions"] > 1,
            features["purchase_span"] / (features["total_transactions"] - 1),
            0,
        )

        # Monetary features
        features["revenue_per_month"] = np.where(
            features["tenure_days"] > 0,
            features["total_revenue"] / (features["tenure_days"] / 30),
            features["total_revenue"],
        )

        features["order_value_range"] = features["max_order_value"] - features["min_order_value"]

        features["order_value_cv"] = np.where(
            features["avg_order_value"] > 0,
            features["std_order_value"].fillna(0) / features["avg_order_value"],
            0,
        )

        # Item diversity
        features["item_diversity_ratio"] = np.where(
            features["total_items"] > 0, features["unique_items"] / features["total_items"], 0
        )

        # Recency ratio
        features["recency_ratio"] = np.where(
            features["tenure_days"] > 0, features["recency_days"] / features["tenure_days"], 1
        )

        # Expected transactions in next period (simple heuristic)
        features["expected_future_orders"] = np.where(
            features["avg_days_between_orders"] > 0,
            self.prediction_horizon_days / features["avg_days_between_orders"],
            features["orders_per_month"] * (self.prediction_horizon_days / 30),
        )

        # Drop date columns
        features = features.drop(columns=["first_purchase", "last_purchase"])

        return features

    def _create_target(
        self,
        transactions_df: pd.DataFrame,
        analysis_date: datetime,
    ) -> pd.DataFrame:
        """
        Create LTV target variable using historical data.

        For training, we use a portion of data as history and
        the remaining as the actual future value.

        Args:
            transactions_df: Full transaction DataFrame.
            analysis_date: Date to split history/future.

        Returns:
            DataFrame with customer_id and future_value.
        """
        df = transactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter to customers who made purchases before analysis date
        historical = df[df["timestamp"] <= analysis_date]
        future = df[df["timestamp"] > analysis_date]

        # Calculate future value
        future_value = future.groupby("customer_id")["total_price"].sum().reset_index()
        future_value.columns = ["customer_id", "future_value"]

        # Include customers with 0 future value
        historical_customers = historical["customer_id"].unique()
        all_customers = pd.DataFrame({"customer_id": historical_customers})

        target = all_customers.merge(future_value, on="customer_id", how="left")
        target["future_value"] = target["future_value"].fillna(0)

        return target

    def prepare_training_data(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        holdout_days: int = 90,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Uses temporal split to simulate prediction scenario.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.
            holdout_days: Days to use for calculating actual LTV.

        Returns:
            Tuple of (features, target).
        """
        df = transactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Split data temporally
        max_date = df["timestamp"].max()
        analysis_date = max_date - pd.Timedelta(days=holdout_days)

        # Use only historical data for features
        historical = df[df["timestamp"] <= analysis_date]

        # Create features from historical data
        features = self._create_features(historical, customers_df, analysis_date)

        # Create target from future data
        target = self._create_target(df, analysis_date)

        # Merge
        data = features.merge(target, on="customer_id")

        X = data.drop(columns=["customer_id", "future_value"])
        y = data["future_value"]

        # Store feature columns
        self.feature_cols = X.columns.tolist()

        return X, y

    def train(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """
        Train the LTV prediction model.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.
            test_size: Proportion of data for testing.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Preparing training data...")
        X, y = self.prepare_training_data(transactions_df, customers_df)

        logger.info(f"Dataset: {len(X)} customers")
        logger.info(f"Target stats: mean=${y.mean():.2f}, median=${y.median():.2f}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Cross-validation
        cv_model = GradientBoostingRegressor(**self.model_params)
        cv_scores = cross_val_score(
            cv_model, X_train_scaled, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        logger.info(f"CV MAE: ${-cv_scores.mean():.2f} (+/- ${cv_scores.std():.2f})")

        # Train final model
        logger.info("Training final model...")
        self.model = GradientBoostingRegressor(**self.model_params)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100,
        }

        # Store feature importance
        self.feature_importance_ = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.trained_at = datetime.now()

        logger.info(f"Test Metrics: MAE=${metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

        return metrics

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict LTV for customers.

        Args:
            features: Customer features DataFrame.

        Returns:
            DataFrame with LTV predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features.copy()
        customer_ids = X.get("customer_id")

        if "customer_id" in X.columns:
            X = X.drop(columns=["customer_id"])

        # Ensure correct features
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols].fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Ensure non-negative
        predictions = np.maximum(predictions, 0)

        # Segment by LTV
        results = pd.DataFrame(
            {
                "predicted_ltv": predictions,
                "ltv_segment": pd.qcut(
                    predictions,
                    q=5,
                    labels=["Low", "Below Average", "Average", "Above Average", "High"],
                    duplicates="drop",
                ),
            }
        )

        if customer_ids is not None:
            results.insert(0, "customer_id", customer_ids.values)

        return results

    def predict_for_transactions(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Predict LTV from transaction data.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.

        Returns:
            DataFrame with customer LTV predictions.
        """
        features = self._create_features(transactions_df, customers_df)
        return self.predict(features)

    def get_high_value_customers(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """
        Get highest LTV customers.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.
            top_n: Number of top customers to return.

        Returns:
            DataFrame of high-value customers sorted by LTV.
        """
        predictions = self.predict_for_transactions(transactions_df, customers_df)
        return predictions.nlargest(top_n, "predicted_ltv")

    def get_ltv_distribution(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Get LTV distribution statistics.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.

        Returns:
            Dictionary with distribution statistics.
        """
        predictions = self.predict_for_transactions(transactions_df, customers_df)
        ltv = predictions["predicted_ltv"]

        segment_counts = predictions["ltv_segment"].value_counts().to_dict()

        return {
            "count": len(ltv),
            "mean": float(ltv.mean()),
            "median": float(ltv.median()),
            "std": float(ltv.std()),
            "min": float(ltv.min()),
            "max": float(ltv.max()),
            "percentiles": {
                "25": float(ltv.quantile(0.25)),
                "50": float(ltv.quantile(0.50)),
                "75": float(ltv.quantile(0.75)),
                "90": float(ltv.quantile(0.90)),
                "95": float(ltv.quantile(0.95)),
            },
            "segment_counts": segment_counts,
            "total_predicted_ltv": float(ltv.sum()),
        }

    def calculate_simple_ltv(
        self,
        transactions_df: pd.DataFrame,
        avg_lifespan_months: int = 24,
    ) -> pd.DataFrame:
        """
        Calculate simple historical LTV without ML.

        Uses: LTV = (Avg Order Value × Purchase Frequency × Lifespan)

        Args:
            transactions_df: Transaction DataFrame.
            avg_lifespan_months: Expected customer lifespan in months.

        Returns:
            DataFrame with simple LTV calculations.
        """
        df = transactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["customer_id"].notna()]

        # Calculate per customer
        customer_metrics = df.groupby("customer_id").agg(
            {
                "total_price": ["mean", "sum", "count"],
                "timestamp": ["min", "max"],
            }
        )

        customer_metrics.columns = [
            "avg_order_value",
            "total_spent",
            "total_orders",
            "first_purchase",
            "last_purchase",
        ]
        customer_metrics = customer_metrics.reset_index()

        # Tenure in months
        customer_metrics["tenure_months"] = (
            (customer_metrics["last_purchase"] - customer_metrics["first_purchase"]).dt.days / 30
        ).clip(lower=1)

        # Purchase frequency (orders per month)
        customer_metrics["purchase_frequency"] = (
            customer_metrics["total_orders"] / customer_metrics["tenure_months"]
        )

        # Simple LTV calculation
        customer_metrics["simple_ltv"] = (
            customer_metrics["avg_order_value"]
            * customer_metrics["purchase_frequency"]
            * avg_lifespan_months
        )

        # Historical LTV (actual spend)
        customer_metrics["historical_ltv"] = customer_metrics["total_spent"]

        # LTV segment
        customer_metrics["ltv_segment"] = pd.qcut(
            customer_metrics["simple_ltv"],
            q=5,
            labels=["Low", "Below Average", "Average", "Above Average", "High"],
            duplicates="drop",
        )

        return customer_metrics[
            [
                "customer_id",
                "avg_order_value",
                "purchase_frequency",
                "tenure_months",
                "total_spent",
                "simple_ltv",
                "historical_ltv",
                "ltv_segment",
            ]
        ]

    def save(self, path: Path | None = None) -> Path:
        """Save the model to disk."""
        path = path or MODELS_DIR / "customer_ltv.joblib"
        save_model(self, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "CustomerLTV":
        """Load a saved model."""
        path = path or MODELS_DIR / "customer_ltv.joblib"
        return load_model(path)


def train_ltv_model(
    transactions_df: pd.DataFrame | None = None,
    customers_df: pd.DataFrame | None = None,
    prediction_horizon_days: int = 365,
    save_path: Path | None = None,
) -> tuple[CustomerLTV, dict[str, float]]:
    """
    Train a Customer LTV prediction model.

    Args:
        transactions_df: Transaction data.
        customers_df: Customer data.
        prediction_horizon_days: Days to predict forward.
        save_path: Path to save the model.

    Returns:
        Tuple of (trained model, metrics dictionary).
    """
    if transactions_df is None:
        transactions_df = load_transactions()

    if customers_df is None:
        try:
            customers_df = load_customers()
        except Exception:
            customers_df = None

    model = CustomerLTV(prediction_horizon_days=prediction_horizon_days)
    metrics = model.train(transactions_df, customers_df)

    if save_path:
        model.save(save_path)

    return model, metrics


def predict_ltv(
    transactions_df: pd.DataFrame,
    model: CustomerLTV | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Predict LTV for customers.

    Args:
        transactions_df: Transaction DataFrame.
        model: Trained CustomerLTV model.
        model_path: Path to load model from.

    Returns:
        DataFrame with LTV predictions.
    """
    if model is None:
        model = CustomerLTV.load(model_path)

    return model.predict_for_transactions(transactions_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Train model
    ltv_model, metrics = train_ltv_model()

    print("\nLTV Prediction Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTop Feature Importance:")
    print(ltv_model.feature_importance_.head(10))

    # Simple LTV calculation
    transactions = load_transactions()
    simple_ltv = ltv_model.calculate_simple_ltv(transactions)

    print("\nSimple LTV Statistics:")
    print(f"  Mean: ${simple_ltv['simple_ltv'].mean():.2f}")
    print(f"  Median: ${simple_ltv['simple_ltv'].median():.2f}")
    print(f"  Max: ${simple_ltv['simple_ltv'].max():.2f}")
