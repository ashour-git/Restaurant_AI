"""
Enhanced Demand Forecasting with Hyperparameter Optimization.

Extends the base demand forecaster with Optuna hyperparameter tuning,
cross-validation, and MLflow experiment tracking.
"""

import logging
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.pipelines.demand_forecasting import DemandForecaster
from ml.utils.data_utils import (
    load_daily_aggregates,
    load_item_daily_sales,
    train_test_split_time_series,
)

logger = logging.getLogger(__name__)


# Default hyperparameter search space
DEFAULT_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "max_depth": {"type": "int", "low": 3, "high": 12},
    "num_leaves": {"type": "int", "low": 15, "high": 127},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "min_child_samples": {"type": "int", "low": 5, "high": 100},
    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}


class EnhancedDemandForecaster(DemandForecaster):
    """
    Enhanced Demand Forecaster with Optuna hyperparameter optimization.

    Extends the base DemandForecaster with:
    - Optuna hyperparameter tuning
    - Time series cross-validation
    - MLflow experiment tracking
    - SHAP-based feature importance

    Attributes:
        best_params: Best hyperparameters found during optimization.
        cv_results: Cross-validation results.
        optimization_history: Optuna study history.

    Example:
        >>> forecaster = EnhancedDemandForecaster()
        >>> forecaster.train_with_optimization(
        ...     df,
        ...     n_trials=50,
        ...     experiment_name="demand_forecast_v1"
        ... )
        >>> predictions = forecaster.predict(test_df)
    """

    def __init__(
        self,
        target_col: str = "quantity_sold",
        lags: list[int] = [1, 7, 14, 28],
        rolling_windows: list[int] = [7, 14, 28],
        param_space: dict[str, dict] | None = None,
    ) -> None:
        """
        Initialize the Enhanced Demand Forecaster.

        Args:
            target_col: Name of the target column.
            lags: List of lag periods for features.
            rolling_windows: List of rolling window sizes.
            param_space: Custom hyperparameter search space.
        """
        super().__init__(target_col, lags, rolling_windows)

        self.param_space = param_space or DEFAULT_PARAM_SPACE
        self.best_params: dict[str, Any] = {}
        self.cv_results: list[dict[str, float]] = []
        self.optimization_history: list[dict] = []

    def _sample_params(self, trial) -> dict[str, Any]:
        """
        Sample hyperparameters from the search space.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of sampled parameters.
        """
        params = {}

        for name, config in self.param_space.items():
            if config["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                )
            elif config["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif config["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    config["choices"],
                )

        # Add fixed parameters
        params["verbose"] = -1
        params["random_state"] = 42

        return params

    def _create_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        metric: str = "rmse",
    ) -> Callable:
        """
        Create an Optuna objective function.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            n_splits: Number of time series splits.
            metric: Metric to optimize.

        Returns:
            Objective function for Optuna.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        def objective(trial) -> float:
            params = self._sample_params(trial)

            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                y_pred = model.predict(X_val)

                if metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                elif metric == "mae":
                    score = mean_absolute_error(y_val, y_pred)
                elif metric == "r2":
                    score = -r2_score(y_val, y_pred)  # Negative for minimization
                else:
                    score = np.sqrt(mean_squared_error(y_val, y_pred))

                scores.append(score)

            return np.mean(scores)

        return objective

    def train_with_optimization(
        self,
        df: pd.DataFrame,
        group_col: str | None = None,
        validation_size: float = 0.2,
        n_trials: int = 50,
        n_cv_splits: int = 5,
        metric: str = "rmse",
        timeout: int | None = None,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Train with Optuna hyperparameter optimization.

        Args:
            df: Training DataFrame.
            group_col: Column to group by for item-level forecasting.
            validation_size: Proportion of data for validation.
            n_trials: Number of Optuna trials.
            n_cv_splits: Number of cross-validation splits.
            metric: Metric to optimize.
            timeout: Optional timeout in seconds.
            experiment_name: Name for MLflow experiment.

        Returns:
            Dictionary with best parameters and metrics.
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return self.train(df, group_col, validation_size)

        logger.info("Preparing features for optimization...")
        df = self._prepare_features(df, is_training=True, group_col=group_col)
        df = df.dropna()

        # Split data
        train_df, test_df = train_test_split_time_series(df, "date", validation_size)

        # Define feature columns
        exclude_cols = ["date", self.target_col, "item_id", "item_name"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target_col]

        # Create Optuna study
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=experiment_name or "demand_forecast_optimization",
        )

        objective = self._create_objective(X_train, y_train, n_cv_splits, metric)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Store optimization results
        self.best_params = study.best_params
        self.optimization_history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ]

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best {metric}: {study.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        final_params = {**self.best_params, "verbose": -1, "random_state": 42}
        self.model = lgb.LGBMRegressor(**final_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Calculate final metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_r2": r2_score(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "test_r2": r2_score(y_test, test_pred),
            "best_cv_score": study.best_value,
            "n_trials": n_trials,
        }

        self.trained_at = datetime.now()

        # Log to MLflow if available
        self._log_to_mlflow(experiment_name, metrics)

        logger.info(f"Final Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Final Test R²: {metrics['test_r2']:.4f}")

        return {
            "best_params": self.best_params,
            "metrics": metrics,
            "optimization_history": self.optimization_history,
        }

    def _log_to_mlflow(
        self,
        experiment_name: str | None,
        metrics: dict[str, float],
    ) -> None:
        """
        Log results to MLflow.

        Args:
            experiment_name: Name for the experiment.
            metrics: Dictionary of metrics.
        """
        try:
            from ml.mlops.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker(experiment_name=experiment_name or "demand_forecasting")

            with tracker.start_run(run_name="optimized_run"):
                tracker.log_params(self.best_params)
                tracker.log_params(
                    {
                        "target_col": self.target_col,
                        "lags": str(self.lags),
                        "rolling_windows": str(self.rolling_windows),
                    }
                )
                tracker.log_metrics(metrics)
                tracker.log_feature_importance(
                    self.feature_cols,
                    self.model.feature_importances_,
                )

                if self.model is not None:
                    tracker.log_model(
                        self.model,
                        "demand_forecaster",
                        model_type="lightgbm",
                    )

            logger.info("Results logged to MLflow")

        except ImportError:
            logger.warning("MLflow not configured. Skipping logging.")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def cross_validate(
        self,
        df: pd.DataFrame,
        group_col: str | None = None,
        n_splits: int = 5,
        lgb_params: dict | None = None,
    ) -> dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            df: DataFrame with features and target.
            group_col: Column to group by.
            n_splits: Number of time series splits.
            lgb_params: LightGBM parameters.

        Returns:
            Dictionary with mean and std of CV metrics.
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")

        df = self._prepare_features(df, is_training=True, group_col=group_col)
        df = df.dropna()

        exclude_cols = ["date", self.target_col, "item_id", "item_name"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df[self.target_col]

        params = (
            lgb_params
            or self.best_params
            or {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 8,
                "verbose": -1,
                "random_state": 42,
            }
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = {"mae": [], "rmse": [], "r2": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            y_pred = model.predict(X_val)

            cv_scores["mae"].append(mean_absolute_error(y_val, y_pred))
            cv_scores["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            cv_scores["r2"].append(r2_score(y_val, y_pred))

            logger.info(f"Fold {fold+1}: RMSE={cv_scores['rmse'][-1]:.4f}")

        self.cv_results = cv_scores

        results = {
            "cv_mae_mean": np.mean(cv_scores["mae"]),
            "cv_mae_std": np.std(cv_scores["mae"]),
            "cv_rmse_mean": np.mean(cv_scores["rmse"]),
            "cv_rmse_std": np.std(cv_scores["rmse"]),
            "cv_r2_mean": np.mean(cv_scores["r2"]),
            "cv_r2_std": np.std(cv_scores["r2"]),
        }

        logger.info(f"CV RMSE: {results['cv_rmse_mean']:.4f} (+/- {results['cv_rmse_std']:.4f})")

        return results

    def explain_predictions(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """
        Explain predictions using SHAP.

        Args:
            X: Features DataFrame.
            num_features: Number of top features to show.

        Returns:
            Dictionary with SHAP explanations.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Returning feature importance only.")
            return {"feature_importance": self.get_feature_importance(num_features).to_dict()}

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure features are in correct order
        X = X[self.feature_cols].fillna(0)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

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
            "expected_value": explainer.expected_value,
        }

    def get_optimization_summary(self) -> dict[str, Any]:
        """
        Get a summary of the hyperparameter optimization.

        Returns:
            Dictionary with optimization summary.
        """
        if not self.optimization_history:
            return {"status": "No optimization performed"}

        values = [t["value"] for t in self.optimization_history]

        return {
            "n_trials": len(self.optimization_history),
            "best_score": min(values),
            "worst_score": max(values),
            "mean_score": np.mean(values),
            "std_score": np.std(values),
            "best_params": self.best_params,
            "improvement": (values[0] - min(values)) / values[0] * 100,
        }


def train_optimized_aggregate_forecaster(
    n_trials: int = 50,
    experiment_name: str = "aggregate_demand_forecast",
) -> tuple[EnhancedDemandForecaster, dict]:
    """
    Train an optimized aggregate demand forecaster.

    Args:
        n_trials: Number of optimization trials.
        experiment_name: Name for the experiment.

    Returns:
        Tuple of (forecaster, results dictionary).
    """
    logger.info("=" * 60)
    logger.info("TRAINING OPTIMIZED AGGREGATE DEMAND FORECASTER")
    logger.info("=" * 60)

    df = load_daily_aggregates()

    forecaster = EnhancedDemandForecaster(
        target_col="total_orders",
        lags=[1, 7, 14, 21, 28],
        rolling_windows=[7, 14, 28],
    )

    results = forecaster.train_with_optimization(
        df,
        n_trials=n_trials,
        experiment_name=experiment_name,
    )

    forecaster.save("demand_forecaster_optimized_aggregate")

    return forecaster, results


def train_optimized_item_forecaster(
    n_trials: int = 50,
    experiment_name: str = "item_demand_forecast",
) -> tuple[EnhancedDemandForecaster, dict]:
    """
    Train an optimized item-level demand forecaster.

    Args:
        n_trials: Number of optimization trials.
        experiment_name: Name for the experiment.

    Returns:
        Tuple of (forecaster, results dictionary).
    """
    logger.info("=" * 60)
    logger.info("TRAINING OPTIMIZED ITEM-LEVEL DEMAND FORECASTER")
    logger.info("=" * 60)

    df = load_item_daily_sales()

    forecaster = EnhancedDemandForecaster(
        target_col="quantity_sold",
        lags=[1, 7, 14, 28],
        rolling_windows=[7, 14, 28],
    )

    results = forecaster.train_with_optimization(
        df,
        group_col="item_id",
        n_trials=n_trials,
        experiment_name=experiment_name,
    )

    forecaster.save("demand_forecaster_optimized_item_level")

    return forecaster, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Train optimized forecaster
    forecaster, results = train_optimized_aggregate_forecaster(n_trials=20)

    print("\nOptimization Summary:")
    summary = forecaster.get_optimization_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nTop Feature Importance:")
    print(forecaster.get_feature_importance(10))
