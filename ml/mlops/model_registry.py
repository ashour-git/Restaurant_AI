"""
MLflow Model Registry Module.

Provides utilities for managing models in MLflow Model Registry,
including versioning, staging, and production deployment.
"""

import logging
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry manager using MLflow.

    Handles model versioning, stage transitions, and loading
    production models for inference.

    Attributes:
        client: MLflow tracking client.
        tracking_uri: MLflow tracking server URI.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model("runs:/abc123/model", "DemandForecaster")
        >>> model = registry.load_production_model("DemandForecaster")
    """

    STAGES = ["None", "Staging", "Production", "Archived"]

    def __init__(self, tracking_uri: str | None = None) -> None:
        """
        Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        import os

        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "file:///D:/Smart Restaurant/mlruns"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> ModelVersion:
        """
        Register a model from a run.

        Args:
            model_uri: URI of the model artifact (e.g., "runs:/run_id/model").
            name: Name for the registered model.
            tags: Optional tags for the model version.
            description: Description of this model version.

        Returns:
            ModelVersion: The registered model version.
        """
        # Register the model
        model_version = mlflow.register_model(model_uri, name)

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, model_version.version, key, value)

        # Add description if provided
        if description:
            self.client.update_model_version(name, model_version.version, description=description)

        logger.info(f"Registered model '{name}' version {model_version.version}")
        return model_version

    def promote_model(
        self,
        name: str,
        version: int | str,
        stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """
        Promote a model version to a new stage.

        Args:
            name: Registered model name.
            version: Model version number.
            stage: Target stage ("Staging" or "Production").
            archive_existing: Archive existing models in target stage.

        Returns:
            ModelVersion: Updated model version.
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.STAGES}")

        # Transition the model
        model_version = self.client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )

        logger.info(f"Promoted model '{name}' v{version} to {stage}")
        return model_version

    def load_model(
        self,
        name: str,
        version: int | str | None = None,
        stage: str | None = None,
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            name: Registered model name.
            version: Specific version to load.
            stage: Stage to load from (e.g., "Production").

        Returns:
            The loaded model object.
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            # Load latest version
            model_uri = f"models:/{name}/latest"

        # Try different model flavors
        try:
            return mlflow.lightgbm.load_model(model_uri)
        except Exception:
            pass

        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            pass

        try:
            return mlflow.xgboost.load_model(model_uri)
        except Exception:
            pass

        # Fallback to pyfunc
        return mlflow.pyfunc.load_model(model_uri)

    def load_production_model(self, name: str) -> Any:
        """
        Load the production version of a model.

        Args:
            name: Registered model name.

        Returns:
            The production model.
        """
        return self.load_model(name, stage="Production")

    def load_staging_model(self, name: str) -> Any:
        """
        Load the staging version of a model.

        Args:
            name: Registered model name.

        Returns:
            The staging model.
        """
        return self.load_model(name, stage="Staging")

    def get_model_versions(
        self,
        name: str,
        stages: list[str] | None = None,
    ) -> list[ModelVersion]:
        """
        Get all versions of a registered model.

        Args:
            name: Registered model name.
            stages: Filter by stages.

        Returns:
            List of model versions.
        """
        filter_string = f"name='{name}'"
        if stages:
            stage_filter = " OR ".join([f"stage='{s}'" for s in stages])
            filter_string += f" AND ({stage_filter})"

        return self.client.search_model_versions(filter_string)

    def get_model_version(
        self,
        name: str,
        version: int | str,
    ) -> ModelVersion:
        """
        Get a specific model version.

        Args:
            name: Registered model name.
            version: Version number.

        Returns:
            ModelVersion object.
        """
        return self.client.get_model_version(name, str(version))

    def get_latest_versions(
        self,
        name: str,
        stages: list[str] | None = None,
    ) -> list[ModelVersion]:
        """
        Get the latest versions for each stage.

        Args:
            name: Registered model name.
            stages: Filter by stages.

        Returns:
            List of latest model versions per stage.
        """
        return self.client.get_latest_versions(name, stages)

    def delete_model_version(
        self,
        name: str,
        version: int | str,
    ) -> None:
        """
        Delete a model version.

        Args:
            name: Registered model name.
            version: Version to delete.
        """
        self.client.delete_model_version(name, str(version))
        logger.info(f"Deleted model '{name}' version {version}")

    def list_registered_models(self) -> list[str]:
        """
        List all registered models.

        Returns:
            List of registered model names.
        """
        models = self.client.search_registered_models()
        return [m.name for m in models]

    def get_model_info(self, name: str) -> dict[str, Any]:
        """
        Get comprehensive info about a registered model.

        Args:
            name: Registered model name.

        Returns:
            Dictionary with model information.
        """
        try:
            model = self.client.get_registered_model(name)
            versions = self.get_model_versions(name)

            return {
                "name": model.name,
                "description": model.description,
                "creation_time": model.creation_timestamp,
                "last_updated": model.last_updated_timestamp,
                "tags": dict(model.tags) if model.tags else {},
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                    }
                    for v in versions
                ],
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}


# Convenience functions


def register_model(
    model_uri: str,
    name: str,
    **kwargs,
) -> ModelVersion:
    """Register a model to the registry."""
    registry = ModelRegistry()
    return registry.register_model(model_uri, name, **kwargs)


def load_production_model(name: str) -> Any:
    """Load the production model."""
    registry = ModelRegistry()
    return registry.load_production_model(name)


def promote_model(
    name: str,
    version: int | str,
    stage: str,
) -> ModelVersion:
    """Promote a model to a new stage."""
    registry = ModelRegistry()
    return registry.promote_model(name, version, stage)


def get_model_version(
    name: str,
    version: int | str,
) -> ModelVersion:
    """Get a model version."""
    registry = ModelRegistry()
    return registry.get_model_version(name, version)
