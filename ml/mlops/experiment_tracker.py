"""
MLflow Experiment Tracking Module.

Provides utilities for tracking ML experiments, logging metrics,
parameters, and artifacts using MLflow.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Default MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:///D:/Smart Restaurant/mlruns")

# Experiment names
EXPERIMENTS = {
    "demand_forecasting": "Restaurant-Demand-Forecasting",
    "recommender": "Restaurant-Recommender-System",
    "churn_prediction": "Restaurant-Churn-Prediction",
    "customer_segmentation": "Restaurant-Customer-Segmentation",
    "customer_ltv": "Restaurant-Customer-LTV",
}


class ExperimentTracker:
    """
    MLflow experiment tracker for restaurant ML models.

    Provides a unified interface for tracking experiments, logging metrics,
    and managing model artifacts.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: URI for MLflow tracking server.
        client: MLflow tracking client.

    Example:
        >>> tracker = ExperimentTracker("demand_forecasting")
        >>> with tracker.start_run(run_name="lgbm_v1") as run:
        ...     tracker.log_params({"n_estimators": 100})
        ...     tracker.log_metrics({"rmse": 0.15, "mae": 0.12})
        ...     tracker.log_model(model, "demand_model")
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Key from EXPERIMENTS or custom experiment name.
            tracking_uri: MLflow tracking server URI.
            tags: Additional tags for the experiment.
        """
        self.experiment_name = EXPERIMENTS.get(experiment_name, experiment_name)
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self.tags = tags or {}

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        self.experiment_id = self._get_or_create_experiment()

        # Initialize client
        self.client = MlflowClient(self.tracking_uri)

        self._active_run = None

        logger.info(
            f"Initialized ExperimentTracker for '{self.experiment_name}' "
            f"(ID: {self.experiment_id})"
        )

    def _get_or_create_experiment(self) -> str:
        """Get or create the MLflow experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags=self.tags,
            )
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ):
        """
        Start an MLflow run as a context manager.

        Args:
            run_name: Name for this run.
            tags: Additional tags for this run.
            description: Run description.

        Yields:
            mlflow.ActiveRun: The active MLflow run.
        """
        run_tags = {**self.tags, **(tags or {})}

        if description:
            run_tags["mlflow.note.content"] = description

        try:
            self._active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=run_tags,
            )
            yield self._active_run
        finally:
            mlflow.end_run()
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values.
        """
        # Convert non-string values
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                clean_params[key] = json.dumps(value)
            else:
                clean_params[key] = value

        mlflow.log_params(clean_params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values.
            step: Step number for the metrics.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Step number.
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file.
            artifact_path: Destination path in artifact storage.
        """
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_figure(
        self,
        figure,
        artifact_file: str,
    ) -> None:
        """
        Log a matplotlib or plotly figure.

        Args:
            figure: Matplotlib or plotly figure object.
            artifact_file: Filename for the figure.
        """
        mlflow.log_figure(figure, artifact_file)

    def log_dict(
        self,
        dictionary: dict[str, Any],
        artifact_file: str,
    ) -> None:
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the JSON file.
        """
        mlflow.log_dict(dictionary, artifact_file)

    def log_dataframe(
        self,
        df: pd.DataFrame,
        artifact_file: str,
    ) -> None:
        """
        Log a DataFrame as a CSV artifact.

        Args:
            df: DataFrame to log.
            artifact_file: Filename for the CSV.
        """
        # Save to temp file and log
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, artifact_file)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Any | None = None,
        input_example: Any | None = None,
        registered_model_name: str | None = None,
        **kwargs,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: The model object to log.
            artifact_path: Path name for the model artifact.
            signature: Model signature for input/output.
            input_example: Example input for the model.
            registered_model_name: If provided, register the model.
            **kwargs: Additional arguments for the model logger.
        """
        # Determine model flavor
        model_type = type(model).__module__

        if "lightgbm" in model_type:
            mlflow.lightgbm.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif "xgboost" in model_type:
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        elif "sklearn" in model_type:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            # Fallback to pickle
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                **kwargs,
            )

    def log_feature_importance(
        self,
        feature_names: list[str],
        importance_values: np.ndarray,
        importance_type: str = "gain",
    ) -> None:
        """
        Log feature importance as metrics and artifact.

        Args:
            feature_names: List of feature names.
            importance_values: Array of importance values.
            importance_type: Type of importance (e.g., "gain", "split").
        """
        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        ).sort_values("importance", ascending=False)

        # Log top features as metrics
        for i, row in importance_df.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row["importance"])

        # Log full importance as artifact
        self.log_dict(
            {
                "type": importance_type,
                "features": importance_df.to_dict(orient="records"),
            },
            "feature_importance.json",
        )

    def get_best_run(
        self,
        metric: str,
        ascending: bool = True,
    ) -> dict[str, Any] | None:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize.
            ascending: If True, lower is better.

        Returns:
            Best run info or None.
        """
        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        return None

    def compare_runs(
        self,
        metric: str,
        top_k: int = 5,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Compare top runs based on a metric.

        Args:
            metric: Metric to compare.
            top_k: Number of top runs.
            ascending: If True, lower is better.

        Returns:
            DataFrame with run comparisons.
        """
        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=top_k,
        )

        results = []
        for run in runs:
            results.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    **run.data.params,
                    **run.data.metrics,
                }
            )

        return pd.DataFrame(results)


# Convenience functions


def get_or_create_experiment(name: str) -> str:
    """Get or create an MLflow experiment."""
    tracker = ExperimentTracker(name)
    return tracker.experiment_id


def log_model_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the active run."""
    mlflow.log_metrics(metrics, step=step)


def log_model_params(params: dict[str, Any]) -> None:
    """Log parameters to the active run."""
    mlflow.log_params(params)


def log_model_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log an artifact to the active run."""
    mlflow.log_artifact(local_path, artifact_path)


def start_run(
    experiment_name: str,
    run_name: str | None = None,
    **kwargs,
) -> mlflow.ActiveRun:
    """Start a new MLflow run."""
    tracker = ExperimentTracker(experiment_name)
    experiment_id = tracker.experiment_id
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        **kwargs,
    )


def end_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()
