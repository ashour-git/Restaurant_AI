"""
Customer Churn Prediction Model.

Predicts which customers are likely to churn (stop visiting) using
machine learning with SHAP explainability.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ml.utils.data_utils import (
    MODELS_DIR,
    load_customers,
    load_model,
    load_transactions,
    save_model,
)

logger = logging.getLogger(__name__)

# Churn definition: customer hasn't ordered in X days
DEFAULT_CHURN_DAYS = 60


class ChurnPredictor:
    """
    Customer Churn Prediction Model.

    Uses LightGBM to predict customer churn probability based on
    behavioral features derived from transaction history.

    Attributes:
        model: Trained LightGBM classifier.
        feature_cols: List of feature column names.
        scaler: StandardScaler for feature normalization.
        churn_threshold_days: Days of inactivity to define churn.

    Example:
        >>> predictor = ChurnPredictor()
        >>> predictor.train(transactions_df, customers_df)
        >>> predictions = predictor.predict(new_customer_features)
        >>> explanations = predictor.explain(new_customer_features)
    """

    def __init__(
        self,
        churn_threshold_days: int = DEFAULT_CHURN_DAYS,
        model_params: dict | None = None,
    ) -> None:
        """
        Initialize the Churn Predictor.

        Args:
            churn_threshold_days: Days without purchase to consider churned.
            model_params: Optional LightGBM parameters.
        """
        self.churn_threshold_days = churn_threshold_days
        self.model_params = model_params or {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
            "class_weight": "balanced",
        }

        self.model: lgb.LGBMClassifier | None = None
        self.feature_cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.trained_at: datetime | None = None
        self.feature_importance_: pd.DataFrame | None = None

    def _create_churn_label(
        self,
        transactions_df: pd.DataFrame,
        analysis_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Create churn labels based on recency.

        Args:
            transactions_df: Transaction DataFrame.
            analysis_date: Reference date for analysis.

        Returns:
            DataFrame with customer_id and is_churned label.
        """
        analysis_date = analysis_date or datetime.now()

        # Get last purchase date per customer
        last_purchase = transactions_df.groupby("customer_id")["timestamp"].max().reset_index()
        last_purchase.columns = ["customer_id", "last_purchase_date"]

        # Calculate days since last purchase
        last_purchase["last_purchase_date"] = pd.to_datetime(last_purchase["last_purchase_date"])
        last_purchase["days_since_last_purchase"] = (
            analysis_date - last_purchase["last_purchase_date"]
        ).dt.days

        # Define churn
        last_purchase["is_churned"] = (
            last_purchase["days_since_last_purchase"] > self.churn_threshold_days
        ).astype(int)

        return last_purchase

    def _create_features(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        analysis_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Create customer features for churn prediction.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer demographics DataFrame.
            analysis_date: Reference date for feature calculation.

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

        # Fill NaN for std (single transaction customers)
        features["std_order_value"] = features["std_order_value"].fillna(0)

        # Time-based features
        time_features = df.groupby("customer_id").agg({"timestamp": ["min", "max"]})
        time_features.columns = ["first_purchase_date", "last_purchase_date"]
        time_features = time_features.reset_index()

        # Customer tenure
        time_features["tenure_days"] = (
            analysis_date - time_features["first_purchase_date"]
        ).dt.days

        # Days between first and last purchase
        time_features["purchase_span_days"] = (
            time_features["last_purchase_date"] - time_features["first_purchase_date"]
        ).dt.days

        features = features.merge(
            time_features[["customer_id", "tenure_days", "purchase_span_days"]], on="customer_id"
        )

        # Order frequency (orders per month of tenure)
        features["orders_per_month"] = np.where(
            features["tenure_days"] > 0,
            features["total_transactions"] / (features["tenure_days"] / 30),
            features["total_transactions"],
        )

        # Recency ratio (recency / tenure)
        features["recency_ratio"] = np.where(
            features["tenure_days"] > 0, features["recency_days"] / features["tenure_days"], 1
        )

        # Day of week preferences
        dow_features = (
            df.groupby(["customer_id", df["timestamp"].dt.dayofweek]).size().unstack(fill_value=0)
        )
        dow_features.columns = [f"orders_dow_{i}" for i in dow_features.columns]
        dow_features = dow_features.reset_index()

        features = features.merge(dow_features, on="customer_id", how="left")

        # Hour preferences (simplified)
        df["hour"] = df["timestamp"].dt.hour
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[0, 11, 14, 17, 21, 24],
            labels=["morning", "lunch", "afternoon", "dinner", "late"],
            include_lowest=True,
        )

        time_of_day_features = (
            df.groupby(["customer_id", "time_of_day"]).size().unstack(fill_value=0)
        )
        time_of_day_features.columns = [f"orders_{c}" for c in time_of_day_features.columns]
        time_of_day_features = time_of_day_features.reset_index()

        features = features.merge(time_of_day_features, on="customer_id", how="left")

        # Add customer demographics if available
        if customers_df is not None and len(customers_df) > 0:
            features = features.merge(
                customers_df[["customer_id", "loyalty_tier"]], on="customer_id", how="left"
            )

        return features

    def prepare_training_data(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        analysis_date: datetime | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels for training.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer DataFrame.
            analysis_date: Reference date.

        Returns:
            Tuple of (features DataFrame, labels Series).
        """
        # Create features
        features = self._create_features(transactions_df, customers_df, analysis_date)

        # Create labels
        labels = self._create_churn_label(transactions_df, analysis_date)

        # Merge
        data = features.merge(labels[["customer_id", "is_churned"]], on="customer_id")

        # Separate features and labels
        X = data.drop(columns=["customer_id", "is_churned"])
        y = data["is_churned"]

        # Store feature columns
        self.feature_cols = X.columns.tolist()

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].fillna("unknown"))

        # Fill remaining NaN
        X = X.fillna(0)

        return X, y

    def train(
        self,
        transactions_df: pd.DataFrame,
        customers_df: pd.DataFrame | None = None,
        test_size: float = 0.2,
        use_cv: bool = True,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """
        Train the churn prediction model.

        Args:
            transactions_df: Transaction DataFrame.
            customers_df: Optional customer DataFrame.
            test_size: Proportion of data for testing.
            use_cv: Whether to use cross-validation.
            n_folds: Number of CV folds.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("Preparing training data...")
        X, y = self.prepare_training_data(transactions_df, customers_df)

        logger.info(f"Dataset: {len(X)} customers, {y.mean()*100:.1f}% churned")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        if use_cv:
            # Cross-validation
            cv_scores = []
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                model = lgb.LGBMClassifier(**self.model_params)
                model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                )

                val_pred = model.predict_proba(X_fold_val)[:, 1]
                fold_auc = roc_auc_score(y_fold_val, val_pred)
                cv_scores.append(fold_auc)
                logger.info(f"Fold {fold+1} AUC: {fold_auc:.4f}")

            logger.info(f"CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final model on full training set
        self.model = lgb.LGBMClassifier(**self.model_params)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Store feature importance
        self.feature_importance_ = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        self.trained_at = datetime.now()

        logger.info(f"Test Metrics: {metrics}")

        return metrics

    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Predict churn for customers.

        Args:
            features: Customer features DataFrame.
            threshold: Probability threshold for classification.

        Returns:
            DataFrame with predictions and probabilities.
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
        X = X[self.feature_cols]

        # Encode categorical
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna("unknown").astype(str)
                # Handle unknown categories
                known = set(encoder.classes_)
                X[col] = X[col].apply(lambda x: x if x in known else "unknown")
                X[col] = encoder.transform(X[col])

        X = X.fillna(0)

        # Predict
        churn_proba = self.model.predict_proba(X)[:, 1]
        churn_pred = (churn_proba >= threshold).astype(int)

        results = pd.DataFrame(
            {
                "churn_probability": churn_proba,
                "is_churned_pred": churn_pred,
                "churn_risk": pd.cut(
                    churn_proba,
                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=["Low", "Medium", "High", "Critical"],
                ),
            }
        )

        if customer_ids is not None:
            results.insert(0, "customer_id", customer_ids.values)

        return results

    def explain(
        self,
        features: pd.DataFrame,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """
        Explain predictions using SHAP values.

        Args:
            features: Customer features DataFrame.
            num_features: Number of top features to show.

        Returns:
            Dictionary with SHAP explanations.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Returning feature importance only.")
            return {"feature_importance": self.feature_importance_.head(num_features).to_dict()}

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features.copy()
        if "customer_id" in X.columns:
            X = X.drop(columns=["customer_id"])
        X = X[self.feature_cols].fillna(0)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        # If binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "mean_shap_value": mean_shap,
            }
        ).sort_values("mean_shap_value", ascending=False)

        return {
            "feature_importance_shap": feature_importance_shap.head(num_features).to_dict(),
            "shap_values": shap_values,
        }

    def get_at_risk_customers(
        self,
        features: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Get customers at high risk of churning.

        Args:
            features: Customer features DataFrame.
            threshold: Probability threshold for high risk.

        Returns:
            DataFrame of at-risk customers sorted by churn probability.
        """
        predictions = self.predict(features, threshold)
        at_risk = predictions[predictions["churn_probability"] >= threshold]
        return at_risk.sort_values("churn_probability", ascending=False)

    def save(self, path: Path | None = None) -> Path:
        """Save the model to disk."""
        path = path or MODELS_DIR / "churn_predictor.joblib"
        save_model(self, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ChurnPredictor":
        """Load a saved model."""
        path = path or MODELS_DIR / "churn_predictor.joblib"
        return load_model(path)


def train_churn_model(
    transactions_df: pd.DataFrame | None = None,
    customers_df: pd.DataFrame | None = None,
    churn_threshold_days: int = DEFAULT_CHURN_DAYS,
    save_path: Path | None = None,
) -> tuple[ChurnPredictor, dict[str, float]]:
    """
    Train a churn prediction model.

    Args:
        transactions_df: Transaction data.
        customers_df: Customer data.
        churn_threshold_days: Days threshold for churn definition.
        save_path: Optional path to save the model.

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

    predictor = ChurnPredictor(churn_threshold_days=churn_threshold_days)
    metrics = predictor.train(transactions_df, customers_df)

    if save_path:
        predictor.save(save_path)

    return predictor, metrics


def predict_churn(
    customer_features: pd.DataFrame,
    model: ChurnPredictor | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    """
    Predict churn for customers.

    Args:
        customer_features: DataFrame with customer features.
        model: Trained ChurnPredictor.
        model_path: Path to load model from.

    Returns:
        DataFrame with churn predictions.
    """
    if model is None:
        model = ChurnPredictor.load(model_path)

    return model.predict(customer_features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Train model
    predictor, metrics = train_churn_model()

    print("\nChurn Prediction Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTop Feature Importance:")
    print(predictor.feature_importance_.head(10))
