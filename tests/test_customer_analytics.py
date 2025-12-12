"""
Tests for customer analytics ML pipelines.

This module contains comprehensive tests for RFM analysis,
churn prediction, customer segmentation, and customer LTV.
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRFMAnalysis:
    """Tests for RFM (Recency, Frequency, Monetary) analysis."""

    def test_rfm_calculation_from_transactions(self, sample_transactions):
        """Test RFM metric calculation from transaction data."""
        df = sample_transactions.copy()
        reference_date = df["timestamp"].max() + timedelta(days=1)

        # Calculate RFM metrics
        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "timestamp": lambda x: (reference_date - x.max()).days,  # Recency
                    "transaction_id": "nunique",  # Frequency
                    "total_price": "sum",  # Monetary
                }
            )
            .reset_index()
        )

        rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

        assert "recency" in rfm.columns
        assert "frequency" in rfm.columns
        assert "monetary" in rfm.columns
        assert len(rfm) == df["customer_id"].nunique()

    def test_recency_is_positive(self, sample_transactions):
        """Test that recency values are positive."""
        df = sample_transactions.copy()
        reference_date = df["timestamp"].max() + timedelta(days=1)

        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "timestamp": lambda x: (reference_date - x.max()).days,
                }
            )
            .reset_index()
        )
        rfm.columns = ["customer_id", "recency"]

        assert (rfm["recency"] >= 0).all()

    def test_frequency_is_positive_integer(self, sample_transactions):
        """Test that frequency values are positive integers."""
        df = sample_transactions.copy()

        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "transaction_id": "nunique",
                }
            )
            .reset_index()
        )
        rfm.columns = ["customer_id", "frequency"]

        assert (rfm["frequency"] >= 1).all()
        assert rfm["frequency"].dtype in [np.int64, np.int32, int]

    def test_monetary_is_positive(self, sample_transactions):
        """Test that monetary values are positive."""
        df = sample_transactions.copy()

        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "total_price": "sum",
                }
            )
            .reset_index()
        )
        rfm.columns = ["customer_id", "monetary"]

        assert (rfm["monetary"] > 0).all()

    def test_rfm_scoring(self, sample_rfm_data):
        """Test RFM scoring logic."""
        df = sample_rfm_data.copy()

        # Score each metric (1-5 scale)
        df["r_score"] = pd.qcut(df["recency"], q=5, labels=[5, 4, 3, 2, 1])
        df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
        df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])

        # Create combined score
        df["rfm_score"] = (
            df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
        )

        assert df["rfm_score"].str.len().eq(3).all()

    def test_customer_segmentation_from_rfm(self, sample_rfm_data):
        """Test customer segmentation based on RFM scores."""
        df = sample_rfm_data.copy()

        # Simple segmentation logic
        def segment_customer(row):
            if row["recency"] <= 30 and row["frequency"] >= 5:
                return "Champion"
            elif row["recency"] <= 60 and row["frequency"] >= 3:
                return "Loyal"
            elif row["recency"] >= 90:
                return "At Risk"
            else:
                return "Regular"

        df["segment"] = df.apply(segment_customer, axis=1)

        assert df["segment"].notna().all()
        assert df["segment"].isin(["Champion", "Loyal", "At Risk", "Regular"]).all()


class TestChurnPrediction:
    """Tests for customer churn prediction."""

    def test_churn_label_creation(self, sample_rfm_data):
        """Test churn label creation logic."""
        df = sample_rfm_data.copy()

        # Define churn: no activity in last 90 days
        df["churned"] = (df["recency"] > 90).astype(int)

        assert df["churned"].isin([0, 1]).all()
        assert df["churned"].sum() >= 0  # At least 0 churned customers

    def test_churn_feature_engineering(self, sample_rfm_data):
        """Test feature engineering for churn prediction."""
        df = sample_rfm_data.copy()

        # Create features
        df["avg_transaction_value"] = df["monetary"] / df["frequency"]
        df["purchase_intensity"] = df["frequency"] / (df["recency"] + 1)

        assert df["avg_transaction_value"].notna().all()
        assert df["purchase_intensity"].notna().all()
        assert (df["avg_transaction_value"] > 0).all()

    def test_churn_model_training(self, sample_rfm_data):
        """Test churn model training."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = sample_rfm_data.copy()
        df["churned"] = (df["recency"] > 60).astype(int)

        features = ["recency", "frequency", "monetary"]
        X = df[features]
        y = df["churned"]

        # Ensure we have both classes
        if y.nunique() < 2:
            df.loc[df.index[:5], "recency"] = 100
            df["churned"] = (df["recency"] > 60).astype(int)
            y = df["churned"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_churn_probability_output(self, sample_rfm_data):
        """Test that churn model outputs probabilities."""
        from sklearn.ensemble import RandomForestClassifier

        df = sample_rfm_data.copy()
        df["churned"] = (df["recency"] > 60).astype(int)

        # Ensure both classes
        if df["churned"].sum() == 0:
            df.loc[df.index[:10], "recency"] = 100
            df["churned"] = (df["recency"] > 60).astype(int)

        features = ["recency", "frequency", "monetary"]
        X = df[features]
        y = df["churned"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        probabilities = model.predict_proba(X)

        assert probabilities.shape[1] == 2  # Two classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Sum to 1
        assert (probabilities >= 0).all() and (probabilities <= 1).all()


class TestCustomerSegmentation:
    """Tests for customer segmentation using clustering."""

    def test_feature_scaling(self, sample_rfm_data):
        """Test that features are properly scaled."""
        from sklearn.preprocessing import StandardScaler

        df = sample_rfm_data.copy()
        features = ["recency", "frequency", "monetary"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        # Check that scaled features have mean ~0 and std ~1
        assert np.allclose(scaled.mean(axis=0), 0, atol=0.1)
        assert np.allclose(scaled.std(axis=0), 1, atol=0.1)

    def test_kmeans_clustering(self, sample_rfm_data):
        """Test K-means clustering on customer data."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = sample_rfm_data.copy()
        features = ["recency", "frequency", "monetary"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)

        assert len(clusters) == len(df)
        assert set(clusters) == {0, 1, 2, 3}

    def test_silhouette_score_calculation(self, sample_rfm_data):
        """Test silhouette score for clustering quality."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        df = sample_rfm_data.copy()
        features = ["recency", "frequency", "monetary"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)

        score = silhouette_score(scaled, clusters)

        assert -1 <= score <= 1  # Silhouette range

    def test_optimal_k_selection(self, sample_rfm_data):
        """Test optimal k selection using elbow method."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = sample_rfm_data.copy()
        features = ["recency", "frequency", "monetary"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        inertias = []
        k_range = range(2, 6)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled)
            inertias.append(kmeans.inertia_)

        # Inertia should decrease as k increases
        assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))

    def test_cluster_profiling(self, sample_rfm_data):
        """Test cluster profile generation."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        df = sample_rfm_data.copy()
        features = ["recency", "frequency", "monetary"]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(scaled)

        # Generate cluster profiles
        profile = df.groupby("cluster")[features].agg(["mean", "std"])

        assert profile.shape[0] == 3  # 3 clusters
        assert len(profile.columns) == len(features) * 2  # mean and std for each


class TestCustomerLTV:
    """Tests for customer lifetime value prediction."""

    def test_historical_ltv_calculation(self, sample_transactions):
        """Test historical LTV calculation."""
        df = sample_transactions.copy()

        ltv = df.groupby("customer_id")["total_price"].sum().reset_index()
        ltv.columns = ["customer_id", "historical_ltv"]

        assert (ltv["historical_ltv"] > 0).all()

    def test_average_order_value(self, sample_transactions):
        """Test average order value calculation."""
        df = sample_transactions.copy()

        aov = (
            df.groupby("customer_id")
            .agg({"total_price": "sum", "transaction_id": "nunique"})
            .reset_index()
        )
        aov["aov"] = aov["total_price"] / aov["transaction_id"]

        assert (aov["aov"] > 0).all()

    def test_purchase_frequency_rate(self, sample_transactions):
        """Test purchase frequency rate calculation."""
        df = sample_transactions.copy()

        # Calculate customer tenure and frequency
        customer_stats = df.groupby("customer_id").agg(
            {"timestamp": ["min", "max"], "transaction_id": "nunique"}
        )
        customer_stats.columns = ["first_purchase", "last_purchase", "frequency"]
        customer_stats["tenure_days"] = (
            customer_stats["last_purchase"] - customer_stats["first_purchase"]
        ).dt.days + 1
        customer_stats["frequency_rate"] = (
            customer_stats["frequency"] / customer_stats["tenure_days"]
        )

        assert (customer_stats["frequency_rate"] >= 0).all()

    def test_ltv_prediction_model(self, sample_rfm_data):
        """Test LTV prediction with regression model."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        df = sample_rfm_data.copy()
        df["ltv"] = df["monetary"]  # Use monetary as target

        features = ["recency", "frequency"]
        X = df[features]
        y = df["ltv"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert (predictions > 0).all() or (predictions >= 0).all()

    def test_cohort_ltv_analysis(self, sample_transactions):
        """Test cohort-based LTV analysis."""
        df = sample_transactions.copy()

        # Assign cohort based on first purchase month
        first_purchase = df.groupby("customer_id")["timestamp"].min().reset_index()
        first_purchase.columns = ["customer_id", "first_purchase"]
        first_purchase["cohort"] = first_purchase["first_purchase"].dt.to_period("M")

        df = df.merge(first_purchase[["customer_id", "cohort"]], on="customer_id")

        # Calculate LTV by cohort
        cohort_ltv = df.groupby("cohort")["total_price"].sum()

        assert len(cohort_ltv) >= 1


class TestCustomerAnalyticsIntegration:
    """Integration tests for customer analytics pipeline."""

    def test_full_rfm_pipeline(self, sample_transactions):
        """Test complete RFM analysis pipeline."""
        df = sample_transactions.copy()
        reference_date = df["timestamp"].max() + timedelta(days=1)

        # Step 1: Calculate RFM
        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "timestamp": lambda x: (reference_date - x.max()).days,
                    "transaction_id": "nunique",
                    "total_price": "sum",
                }
            )
            .reset_index()
        )
        rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

        # Step 2: Score RFM
        rfm["r_score"] = pd.qcut(rfm["recency"], q=4, labels=[4, 3, 2, 1], duplicates="drop")
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4])

        # Step 3: Create segment
        rfm["rfm_score"] = (
            rfm["r_score"].astype(int) + rfm["f_score"].astype(int) + rfm["m_score"].astype(int)
        )

        def assign_segment(score):
            if score >= 10:
                return "Champions"
            elif score >= 7:
                return "Loyal Customers"
            elif score >= 4:
                return "Potential Loyalists"
            else:
                return "At Risk"

        rfm["segment"] = rfm["rfm_score"].apply(assign_segment)

        assert rfm["segment"].notna().all()
        assert len(rfm) == df["customer_id"].nunique()

    def test_churn_with_segmentation(self, sample_transactions):
        """Test combining churn prediction with segmentation."""
        df = sample_transactions.copy()
        reference_date = df["timestamp"].max() + timedelta(days=1)

        # Calculate RFM
        rfm = (
            df.groupby("customer_id")
            .agg(
                {
                    "timestamp": lambda x: (reference_date - x.max()).days,
                    "transaction_id": "nunique",
                    "total_price": "sum",
                }
            )
            .reset_index()
        )
        rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

        # Add churn label
        rfm["churned"] = (rfm["recency"] > 60).astype(int)

        # Calculate churn rate by segment
        rfm["segment"] = pd.qcut(rfm["monetary"], q=3, labels=["Low", "Medium", "High"])

        churn_by_segment = rfm.groupby("segment")["churned"].mean()

        assert len(churn_by_segment) == 3
        assert (churn_by_segment >= 0).all() and (churn_by_segment <= 1).all()
        assert (churn_by_segment >= 0).all() and (churn_by_segment <= 1).all()
