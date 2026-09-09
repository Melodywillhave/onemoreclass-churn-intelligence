from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn-model-development"


def configure_mlflow() -> None:
    """Configure MLflow tracking for local model development."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_evaluation_results(
    metrics_df: pd.DataFrame,
    artifact_path: Path,
) -> None:
    """Log walk-forward evaluation results to MLflow."""
    configure_mlflow()

    metric_columns = [
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
    ]

    for _, row in metrics_df.iterrows():
        fold = int(row["fold"])
        model_name = str(row["model"])

        with mlflow.start_run(
            run_name=f"{model_name}-fold-{fold}"
        ):
            mlflow.log_param(
                "model",
                model_name,
            )
            mlflow.log_param(
                "fold",
                fold,
            )
            mlflow.log_param(
                "train_snapshots",
                str(row["train_snapshots"]),
            )
            mlflow.log_param(
                "validation_snapshot",
                str(row["val_snapshot"]),
            )

            for metric_name in metric_columns:
                mlflow.log_metric(
                    metric_name,
                    float(row[metric_name]),
                )

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="evaluation",
            )