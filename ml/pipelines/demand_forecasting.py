"""
Demand Forecasting Pipeline.

Time-series forecasting for predicting daily/hourly demand for menu items.
Uses LightGBM with engineered time features, lags, and rolling statistics.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.utils.data_utils import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    load_daily_aggregates,
    load_item_daily_sales,
    load_model,
    save_model,
    train_test_split_time_series,
)


class DemandForecaster:
    """
    Time-series demand forecasting model.

    Uses LightGBM with engineered features to predict future demand
    at both aggregate and item levels.

    Attributes:
        model: Trained LightGBM model.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        label_encoders: Dictionary of label encoders for categorical features.

    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.train(train_df, target_col='quantity_sold')
        >>> predictions = forecaster.predict(test_df)
    """

    def __init__(
        self,
        target_col: str = "quantity_sold",
        lags: list[int] = [1, 7, 14, 28],
        rolling_windows: list[int] = [7, 14, 28],
    ):
        """
        Initialize the demand forecaster.

        Args:
            target_col: Name of the target column to predict.
            lags: List of lag periods for feature engineering.
            rolling_windows: List of rolling window sizes for features.
        """
        self.target_col = target_col
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.model: lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.trained_at: datetime | None = None

    def _prepare_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Prepare features for the model.

        Args:
            df: Input DataFrame.
            is_training: Whether this is training data (fit encoders).
            group_col: Column to group by for lag/rolling features.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.copy()

        # Time features
        df = create_time_features(df, date_col="date")

        # Lag features
        df = create_lag_features(
            df,
            target_col=self.target_col,
            lags=self.lags,
            group_col=group_col,
        )

        # Rolling features
        df = create_rolling_features(
            df,
            target_col=self.target_col,
            windows=self.rolling_windows,
            group_col=group_col,
        )

        # Encode categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # Remove date and ID columns from encoding
        categorical_cols = [
            c for c in categorical_cols if c not in ["date", "item_id", "item_name"]
        ]

        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(
                    df[col].fillna("unknown").astype(str)
                )
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].fillna("unknown").astype(str)
                    df[col] = df[col].apply(lambda x: x if x in known_labels else "unknown")
                    df[col] = self.label_encoders[col].transform(df[col])

        return df

    def train(
        self,
        df: pd.DataFrame,
        group_col: str | None = None,
        validation_size: float = 0.2,
        lgb_params: dict | None = None,
    ) -> dict[str, float]:
        """
        Train the demand forecasting model.

        Args:
            df: Training DataFrame with date, features, and target.
            group_col: Column to group by for item-level forecasting.
            validation_size: Proportion of data for validation.
            lgb_params: Optional LightGBM parameters.

        Returns:
            Dict[str, float]: Training and validation metrics.

        Example:
            >>> df = load_item_daily_sales()
            >>> forecaster = DemandForecaster()
            >>> metrics = forecaster.train(df, group_col='item_id')
            >>> print(f"RMSE: {metrics['val_rmse']:.2f}")
        """
        print("Preparing features...")
        df = self._prepare_features(df, is_training=True, group_col=group_col)

        # Drop rows with NaN from lag features
        df = df.dropna()

        # Split data
        train_df, val_df = train_test_split_time_series(df, "date", validation_size)

        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ["date", self.target_col, "item_id", "item_name"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        X_val = val_df[self.feature_cols]
        y_val = val_df[self.target_col]

        # Default LightGBM parameters
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }

        if lgb_params:
            default_params.update(lgb_params)

        print("Training LightGBM model...")
        self.model = lgb.LGBMRegressor(**default_params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_r2": r2_score(y_train, train_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
            "val_r2": r2_score(y_val, val_pred),
        }

        self.trained_at = datetime.now()

        print("\nTraining complete!")
        print(f"Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"Val RMSE: {metrics['val_rmse']:.4f}")
        print(f"Val R²: {metrics['val_r2']:.4f}")

        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        group_col: str | None = None,
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: DataFrame with features.
            group_col: Column to group by for item-level forecasting.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If model is not trained.

        Example:
            >>> predictions = forecaster.predict(test_df)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df = self._prepare_features(df, is_training=False, group_col=group_col)
        df = df.dropna(subset=self.feature_cols)

        X = df[self.feature_cols]
        return self.model.predict(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            top_n: Number of top features to return.

        Returns:
            pd.DataFrame: Feature importance DataFrame.

        Example:
            >>> importance_df = forecaster.get_feature_importance()
            >>> print(importance_df.head())
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return importance.head(top_n).reset_index(drop=True)

    def forecast_future(
        self,
        historical_df: pd.DataFrame,
        days_ahead: int = 7,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts for future dates.

        Args:
            historical_df: Historical data for feature generation.
            days_ahead: Number of days to forecast.
            group_col: Column to group by.

        Returns:
            pd.DataFrame: Forecasted values with dates.

        Example:
            >>> forecast = forecaster.forecast_future(df, days_ahead=14)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get the last date in historical data
        last_date = historical_df["date"].max()

        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead,
            freq="D",
        )

        forecasts = []
        current_df = historical_df.copy()

        for future_date in future_dates:
            # Create row for prediction
            if group_col:
                # For item-level forecasting
                unique_items = current_df[[group_col]].drop_duplicates()
                future_rows = unique_items.copy()
                future_rows["date"] = future_date
            else:
                future_rows = pd.DataFrame({"date": [future_date]})

            # Add target column with NaN (will use lag features)
            future_rows[self.target_col] = np.nan

            # Prepare features using historical + future row
            temp_df = pd.concat([current_df, future_rows], ignore_index=True)
            temp_df = temp_df.sort_values("date")

            # Prepare and predict
            pred_df = self._prepare_features(temp_df, is_training=False, group_col=group_col)
            pred_df = pred_df[pred_df["date"] == future_date]

            if len(pred_df) > 0:
                pred_df = pred_df.dropna(
                    subset=[c for c in self.feature_cols if c in pred_df.columns]
                )
                available_features = [c for c in self.feature_cols if c in pred_df.columns]

                if len(available_features) == len(self.feature_cols):
                    predictions = self.model.predict(pred_df[self.feature_cols])

                    # Store forecasts
                    result = future_rows.copy()
                    result[self.target_col] = predictions
                    forecasts.append(result)

                    # Add to historical for next iteration
                    update_df = future_rows.copy()
                    update_df[self.target_col] = predictions
                    current_df = pd.concat([current_df, update_df], ignore_index=True)

        if forecasts:
            return pd.concat(forecasts, ignore_index=True)
        return pd.DataFrame()

    def save(self, name: str = "demand_forecaster") -> Path:
        """
        Save the trained model.

        Args:
            name: Model name.

        Returns:
            Path: Path to saved model.
        """
        metadata = {
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "label_encoders": self.label_encoders,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }

        return save_model(self.model, name, metadata)

    @classmethod
    def load(cls, name: str = "demand_forecaster") -> "DemandForecaster":
        """
        Load a trained model.

        Args:
            name: Model name.

        Returns:
            DemandForecaster: Loaded model instance.
        """
        model, metadata = load_model(name)

        forecaster = cls(
            target_col=metadata.get("target_col", "quantity_sold"),
            lags=metadata.get("lags", [1, 7, 14, 28]),
            rolling_windows=metadata.get("rolling_windows", [7, 14, 28]),
        )

        forecaster.model = model
        forecaster.feature_cols = metadata.get("feature_cols", [])
        forecaster.label_encoders = metadata.get("label_encoders", {})
        forecaster.trained_at = (
            datetime.fromisoformat(metadata["trained_at"]) if metadata.get("trained_at") else None
        )

        return forecaster


def train_aggregate_forecaster() -> tuple[DemandForecaster, dict[str, float]]:
    """
    Train a forecaster for aggregate daily sales.

    Returns:
        Tuple[DemandForecaster, Dict]: Trained forecaster and metrics.

    Example:
        >>> forecaster, metrics = train_aggregate_forecaster()
        >>> print(f"RMSE: {metrics['val_rmse']:.2f}")
    """
    print("=" * 60)
    print("TRAINING AGGREGATE DEMAND FORECASTER")
    print("=" * 60)

    df = load_daily_aggregates()

    forecaster = DemandForecaster(
        target_col="total_orders",
        lags=[1, 7, 14, 21, 28],
        rolling_windows=[7, 14, 28],
    )

    metrics = forecaster.train(df)
    forecaster.save("demand_forecaster_aggregate")

    return forecaster, metrics


def train_item_level_forecaster() -> tuple[DemandForecaster, dict[str, float]]:
    """
    Train a forecaster for item-level daily sales.

    Returns:
        Tuple[DemandForecaster, Dict]: Trained forecaster and metrics.

    Example:
        >>> forecaster, metrics = train_item_level_forecaster()
        >>> print(f"RMSE: {metrics['val_rmse']:.2f}")
    """
    print("=" * 60)
    print("TRAINING ITEM-LEVEL DEMAND FORECASTER")
    print("=" * 60)

    df = load_item_daily_sales()

    forecaster = DemandForecaster(
        target_col="quantity_sold",
        lags=[1, 7, 14, 28],
        rolling_windows=[7, 14, 28],
    )

    metrics = forecaster.train(df, group_col="item_id")
    forecaster.save("demand_forecaster_item_level")

    return forecaster, metrics


if __name__ == "__main__":
    # Train both forecasters
    agg_forecaster, agg_metrics = train_aggregate_forecaster()
    print("\nAggregate Forecaster Feature Importance:")
    print(agg_forecaster.get_feature_importance(10))

    print("\n")

    item_forecaster, item_metrics = train_item_level_forecaster()
    print("\nItem-Level Forecaster Feature Importance:")
    print(item_forecaster.get_feature_importance(10))
