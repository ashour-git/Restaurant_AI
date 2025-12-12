"""
Customer Segmentation using K-Means Clustering.

Groups customers into distinct segments based on behavioral and
demographic features for targeted marketing strategies.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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


# Segment descriptions for interpretation
SEGMENT_PROFILES: dict[str, dict[str, str]] = {
    "high_value_frequent": {
        "name": "VIP Customers",
        "description": "High spenders who visit frequently",
        "strategy": "Priority service, exclusive offers, loyalty rewards",
    },
    "high_value_infrequent": {
        "name": "Big Spenders",
        "description": "High order values but occasional visits",
        "strategy": "Re-engagement campaigns, subscription offers",
    },
    "low_value_frequent": {
        "name": "Regular Customers",
        "description": "Frequent visitors with smaller orders",
        "strategy": "Upselling, combo deals, volume discounts",
    },
    "low_value_infrequent": {
        "name": "Casual Visitors",
        "description": "Occasional visitors with small orders",
        "strategy": "Welcome back offers, trial promotions",
    },
    "new_customers": {
        "name": "New Customers",
        "description": "Recently acquired customers",
        "strategy": "Onboarding journey, first purchase incentives",
    },
    "churning": {
        "name": "At Risk",
        "description": "Previously active but declining engagement",
        "strategy": "Win-back campaigns, personalized outreach",
    },
}


class CustomerSegmenter:
    """
    Customer Segmentation using K-Means Clustering.

    Automatically determines optimal number of clusters and provides
    interpretable segment profiles with actionable recommendations.

    Attributes:
        model: Fitted KMeans model.
        scaler: StandardScaler for feature normalization.
        n_clusters: Number of customer segments.
        segment_profiles: Dictionary of segment characteristics.

    Example:
        >>> segmenter = CustomerSegmenter()
        >>> segmenter.fit(transactions_df, customers_df)
        >>> segments = segmenter.predict(new_customer_features)
        >>> profiles = segmenter.get_segment_profiles()
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        min_clusters: int = 3,
        max_clusters: int = 8,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the Customer Segmenter.

        Args:
            n_clusters: Number of clusters. If None, auto-determine.
            min_clusters: Minimum clusters to try.
            max_clusters: Maximum clusters to try.
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.segment_profiles_: pd.DataFrame | None = None
        self.cluster_centers_: pd.DataFrame | None = None
        self.fitted_at: datetime | None = None

    def _create_features(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        analysis_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Create customer features for segmentation.

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

        # Aggregate customer-level features
        features = df.groupby("customer_id").agg(
            {
                # Recency
                "timestamp": lambda x: (analysis_date - x.max()).days,
                # Frequency
                "transaction_id": "nunique",
                # Monetary
                "total_price": ["sum", "mean", "std"],
                # Items
                "quantity": ["sum", "mean"],
                # Item variety
                "item_id": "nunique",
            }
        )

        # Flatten column names
        features.columns = [
            "recency_days",
            "total_transactions",
            "total_spent",
            "avg_order_value",
            "std_order_value",
            "total_items",
            "avg_items_per_order",
            "unique_items_ordered",
        ]
        features = features.reset_index()

        # Fill NaN for std
        features["std_order_value"] = features["std_order_value"].fillna(0)

        # Calculate tenure
        tenure = df.groupby("customer_id")["timestamp"].agg(["min", "max"])
        tenure["tenure_days"] = (analysis_date - tenure["min"]).dt.days
        tenure = tenure.reset_index()[["customer_id", "tenure_days"]]

        features = features.merge(tenure, on="customer_id")

        # Derived features
        features["orders_per_month"] = np.where(
            features["tenure_days"] > 0,
            features["total_transactions"] / (features["tenure_days"] / 30),
            features["total_transactions"],
        )

        features["spend_per_month"] = np.where(
            features["tenure_days"] > 0,
            features["total_spent"] / (features["tenure_days"] / 30),
            features["total_spent"],
        )

        # Coefficient of variation for order value
        features["order_value_cv"] = np.where(
            features["avg_order_value"] > 0,
            features["std_order_value"] / features["avg_order_value"],
            0,
        )

        # Item diversity ratio
        features["item_diversity"] = np.where(
            features["total_items"] > 0,
            features["unique_items_ordered"] / features["total_items"],
            0,
        )

        return features

    def _find_optimal_clusters(
        self,
        X: np.ndarray,
    ) -> tuple[int, dict[int, dict[str, float]]]:
        """
        Find optimal number of clusters using multiple metrics.

        Args:
            X: Scaled feature matrix.

        Returns:
            Tuple of (optimal k, metrics dictionary).
        """
        metrics: dict[int, dict[str, float]] = {}

        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(X)

            metrics[k] = {
                "inertia": kmeans.inertia_,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }

            logger.info(
                f"k={k}: silhouette={metrics[k]['silhouette']:.3f}, "
                f"CH={metrics[k]['calinski_harabasz']:.1f}"
            )

        # Find optimal k using silhouette score
        optimal_k = max(metrics.keys(), key=lambda k: metrics[k]["silhouette"])

        logger.info(f"Optimal clusters: {optimal_k}")
        return optimal_k, metrics

    def fit(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        analysis_date: datetime | None = None,
    ) -> "CustomerSegmenter":
        """
        Fit the segmentation model.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics.
            analysis_date: Reference date for calculations.

        Returns:
            Self for method chaining.
        """
        logger.info("Creating customer features...")
        features = self._create_features(transactions_df, customers_df, analysis_date)

        # Store customer IDs
        customer_ids = features["customer_id"].values

        # Select features for clustering
        self.feature_cols = [
            "recency_days",
            "total_transactions",
            "total_spent",
            "avg_order_value",
            "tenure_days",
            "orders_per_month",
            "spend_per_month",
            "item_diversity",
        ]

        X = features[self.feature_cols].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal clusters if not specified
        if self.n_clusters is None:
            self.n_clusters, cluster_metrics = self._find_optimal_clusters(X_scaled)

        # Fit final model
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.model.fit_predict(X_scaled)

        # Store cluster centers in original scale
        self.cluster_centers_ = pd.DataFrame(
            self.scaler.inverse_transform(self.model.cluster_centers_),
            columns=self.feature_cols,
        )
        self.cluster_centers_["cluster"] = range(self.n_clusters)

        # Create segment profiles
        features["cluster"] = labels
        self._create_segment_profiles(features)

        self.fitted_at = datetime.now()

        return self

    def _create_segment_profiles(self, features: pd.DataFrame) -> None:
        """
        Create interpretable segment profiles.

        Args:
            features: Features DataFrame with cluster labels.
        """
        # Aggregate by cluster
        profiles = (
            features.groupby("cluster")
            .agg(
                {
                    "customer_id": "count",
                    "recency_days": "mean",
                    "total_transactions": "mean",
                    "total_spent": "mean",
                    "avg_order_value": "mean",
                    "tenure_days": "mean",
                    "orders_per_month": "mean",
                    "spend_per_month": "mean",
                    "item_diversity": "mean",
                }
            )
            .reset_index()
        )

        profiles.columns = [
            "cluster",
            "customer_count",
            "avg_recency",
            "avg_transactions",
            "avg_total_spent",
            "avg_order_value",
            "avg_tenure",
            "avg_orders_per_month",
            "avg_spend_per_month",
            "avg_item_diversity",
        ]

        # Calculate percentage of total
        profiles["pct_of_customers"] = (
            profiles["customer_count"] / profiles["customer_count"].sum() * 100
        )

        # Assign segment names based on characteristics
        segment_names = []
        segment_descriptions = []
        strategies = []

        for _, row in profiles.iterrows():
            # Classify based on spend and frequency
            high_value = row["avg_total_spent"] > profiles["avg_total_spent"].median()
            high_freq = row["avg_transactions"] > profiles["avg_transactions"].median()
            is_new = row["avg_tenure"] < profiles["avg_tenure"].quantile(0.25)
            is_churning = row["avg_recency"] > profiles["avg_recency"].quantile(0.75)

            if is_new:
                profile = SEGMENT_PROFILES["new_customers"]
            elif is_churning:
                profile = SEGMENT_PROFILES["churning"]
            elif high_value and high_freq:
                profile = SEGMENT_PROFILES["high_value_frequent"]
            elif high_value and not high_freq:
                profile = SEGMENT_PROFILES["high_value_infrequent"]
            elif not high_value and high_freq:
                profile = SEGMENT_PROFILES["low_value_frequent"]
            else:
                profile = SEGMENT_PROFILES["low_value_infrequent"]

            segment_names.append(profile["name"])
            segment_descriptions.append(profile["description"])
            strategies.append(profile["strategy"])

        profiles["segment_name"] = segment_names
        profiles["description"] = segment_descriptions
        profiles["recommended_strategy"] = strategies

        self.segment_profiles_ = profiles

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Assign customers to segments.

        Args:
            features: Customer features DataFrame.

        Returns:
            DataFrame with segment assignments.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = features.copy()
        customer_ids = X.get("customer_id")

        # Ensure correct features
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols].fillna(0)

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)

        # Map to segment names
        segment_names = []
        for label in labels:
            if self.segment_profiles_ is not None:
                row = self.segment_profiles_[self.segment_profiles_["cluster"] == label]
                if len(row) > 0:
                    segment_names.append(row["segment_name"].values[0])
                else:
                    segment_names.append(f"Segment_{label}")
            else:
                segment_names.append(f"Segment_{label}")

        results = pd.DataFrame(
            {
                "cluster": labels,
                "segment_name": segment_names,
            }
        )

        if customer_ids is not None:
            results.insert(0, "customer_id", customer_ids.values)

        return results

    def get_segment_profiles(self) -> pd.DataFrame:
        """
        Get segment profiles with statistics and recommendations.

        Returns:
            DataFrame with segment profiles.
        """
        if self.segment_profiles_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.segment_profiles_.copy()

    def get_cluster_centers(self) -> pd.DataFrame:
        """
        Get cluster centers in original feature scale.

        Returns:
            DataFrame with cluster centers.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.cluster_centers_.copy()

    def get_segment_summary(self) -> dict[str, Any]:
        """
        Get a summary of all segments.

        Returns:
            Dictionary with segment summary.
        """
        if self.segment_profiles_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        summary = {
            "total_customers": int(self.segment_profiles_["customer_count"].sum()),
            "num_segments": self.n_clusters,
            "segments": [],
        }

        for _, row in self.segment_profiles_.iterrows():
            summary["segments"].append(
                {
                    "cluster": int(row["cluster"]),
                    "name": row["segment_name"],
                    "customer_count": int(row["customer_count"]),
                    "pct_of_customers": round(row["pct_of_customers"], 1),
                    "avg_order_value": round(row["avg_order_value"], 2),
                    "avg_orders_per_month": round(row["avg_orders_per_month"], 2),
                    "description": row["description"],
                    "strategy": row["recommended_strategy"],
                }
            )

        return summary

    def save(self, path: Path | None = None) -> Path:
        """Save the model to disk."""
        path = path or MODELS_DIR / "customer_segmenter.joblib"
        save_model(self, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "CustomerSegmenter":
        """Load a saved model."""
        path = path or MODELS_DIR / "customer_segmenter.joblib"
        return load_model(path)


def segment_customers(
    transactions_df: pd.DataFrame | None = None,
    customers_df: pd.DataFrame | None = None,
    n_clusters: int | None = None,
    save_path: Path | None = None,
) -> tuple[CustomerSegmenter, pd.DataFrame]:
    """
    Segment customers using K-Means clustering.

    Args:
        transactions_df: Transaction data.
        customers_df: Customer data.
        n_clusters: Number of clusters.
        save_path: Path to save the model.

    Returns:
        Tuple of (fitted segmenter, segment profiles).
    """
    if transactions_df is None:
        transactions_df = load_transactions()

    if customers_df is None:
        try:
            customers_df = load_customers()
        except Exception:
            customers_df = None

    segmenter = CustomerSegmenter(n_clusters=n_clusters)
    segmenter.fit(transactions_df, customers_df)

    if save_path:
        segmenter.save(save_path)

    return segmenter, segmenter.get_segment_profiles()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Segment customers
    segmenter, profiles = segment_customers()

    print("\nCustomer Segment Profiles:")
    print("=" * 60)

    summary = segmenter.get_segment_summary()
    for segment in summary["segments"]:
        print(f"\n{segment['name']} (Cluster {segment['cluster']})")
        print(f"  Customers: {segment['customer_count']} ({segment['pct_of_customers']}%)")
        print(f"  Avg Order Value: ${segment['avg_order_value']:.2f}")
        print(f"  Orders/Month: {segment['avg_orders_per_month']:.2f}")
        print(f"  Description: {segment['description']}")
        print(f"  Strategy: {segment['strategy']}")
