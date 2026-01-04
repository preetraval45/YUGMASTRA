"""
MLOps Experiment Tracking with MLflow
Track ML experiments, hyperparameters, metrics, and model versioning
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracking for YUGMASTRA ML models
    Track training runs, hyperparameters, metrics, and artifacts
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "yugmastra-security-ai"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.current_run = None

        logger.info(f"Initialized MLflow tracker: {experiment_name}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_run = mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

        # Default tags
        mlflow.set_tag("platform", "yugmastra")
        mlflow.set_tag("model_type", "security_ai")

        logger.info(f"Started MLflow run: {run_name}")
        return self.current_run

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {list(params.keys())}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics (loss, accuracy, etc.)"""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics at step {step}: {metrics}")

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None
    ):
        """Log single metric"""
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = "pytorch",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None
    ):
        """
        Log trained model

        Args:
            model: Trained model
            artifact_path: Path in MLflow artifacts
            model_type: pytorch, sklearn, tensorflow
            signature: Model signature (input/output schema)
            input_example: Example input for inference
        """
        if model_type == "pytorch":
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example
            )
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                signature=signature,
                input_example=input_example
            )

        logger.info(f"Logged {model_type} model to {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log file artifact (plots, configs, etc.)"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, filename)

    def log_figure(self, figure, filename: str):
        """Log matplotlib/plotly figure"""
        mlflow.log_figure(figure, filename)

    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        self.current_run = None
        logger.info("Ended MLflow run")

    def get_best_run(
        self,
        metric: str = "val_accuracy",
        ascending: bool = False
    ) -> Optional[Dict]:
        """
        Get best run based on metric

        Args:
            metric: Metric to optimize
            ascending: True for minimizing (loss), False for maximizing (accuracy)

        Returns:
            Best run info
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )

        if runs.empty:
            return None

        best_run = runs.iloc[0].to_dict()
        return best_run

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple runs

        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare

        Returns:
            Dictionary mapping run_id -> {metric: value}
        """
        comparison = {}

        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            comparison[run_id] = {
                metric: run.data.metrics.get(metric)
                for metric in metrics
            }

        return comparison


class ModelRegistry:
    """
    Model versioning and registry management
    Track production models, staging, archived
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model to registry

        Args:
            model_uri: URI of logged model (runs:/<run_id>/model)
            model_name: Name in registry
            tags: Optional tags

        Returns:
            Model version
        """
        result = mlflow.register_model(model_uri, model_name)

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name,
                    result.version,
                    key,
                    value
                )

        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version

    def promote_to_production(
        self,
        model_name: str,
        version: str
    ):
        """Promote model version to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Promoted {model_name} v{version} to Production")

    def promote_to_staging(
        self,
        model_name: str,
        version: str
    ):
        """Promote model version to staging"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        logger.info(f"Promoted {model_name} v{version} to Staging")

    def archive_model(
        self,
        model_name: str,
        version: str
    ):
        """Archive old model version"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Archived"
        )
        logger.info(f"Archived {model_name} v{version}")

    def get_latest_production_model(self, model_name: str) -> Optional[Any]:
        """Load latest production model"""
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded production model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

    def list_models(self) -> List[str]:
        """List all registered models"""
        models = self.client.search_registered_models()
        return [model.name for model in models]


# Example usage for Red Team AI training
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Initialize tracker
    tracker = ExperimentTracker(
        tracking_uri="http://localhost:5000",
        experiment_name="yugmastra-red-team-ai"
    )

    # Start training run
    tracker.start_run(
        run_name="red_team_ppo_v1",
        tags={
            "model": "PPO",
            "agent": "red_team",
            "environment": "cyber_range"
        }
    )

    # Log hyperparameters
    tracker.log_params({
        "learning_rate": 3e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5
    })

    # Simulate training loop
    for epoch in range(100):
        # Training metrics
        tracker.log_metrics({
            "train/loss": np.random.rand(),
            "train/policy_loss": np.random.rand(),
            "train/value_loss": np.random.rand(),
            "train/entropy": np.random.rand(),
        }, step=epoch)

        # Validation metrics
        tracker.log_metrics({
            "val/reward": 100 + np.random.randn() * 10,
            "val/success_rate": 0.7 + np.random.rand() * 0.2,
            "val/episode_length": 50 + np.random.randn() * 5,
        }, step=epoch)

        # Log attack metrics
        tracker.log_metrics({
            "attacks/sql_injection_success": np.random.rand(),
            "attacks/xss_success": np.random.rand(),
            "attacks/rce_success": np.random.rand(),
            "defense/detection_rate": np.random.rand(),
        }, step=epoch)

    # Save model
    dummy_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    tracker.log_model(
        dummy_model,
        artifact_path="model",
        model_type="pytorch"
    )

    # Log training config
    tracker.log_dict({
        "architecture": "PPO Actor-Critic",
        "layers": [128, 256, 256, 10],
        "activation": "ReLU",
        "optimizer": "Adam"
    }, "model_config.json")

    tracker.end_run()

    # Model Registry
    registry = ModelRegistry()

    # Register best model
    run_id = tracker.current_run.info.run_id if tracker.current_run else "latest"
    version = registry.register_model(
        model_uri=f"runs:/{run_id}/model",
        model_name="red-team-ppo",
        tags={"validated": "true", "dataset": "cyber_range_v2"}
    )

    # Promote to production
    registry.promote_to_production("red-team-ppo", version)

    # Load production model for inference
    prod_model = registry.get_latest_production_model("red-team-ppo")
    print(f"Loaded production model: {prod_model}")
