from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from src.modeling.mlflow_tracking import (
    EXPERIMENT_NAME,
    configure_mlflow,
)


REGISTERED_MODEL_NAME = "churn-risk-model"
CHAMPION_ALIAS = "champion"


def register_champion_model(
    model: Pipeline,
    X_example: pd.DataFrame,
    training_rows: int,
    scoring_snapshot: str,
    label_availability_cutoff: str,
) -> str:
    """Log the final scoring model and register it as the champion model."""
    configure_mlflow()

    with mlflow.start_run(
        run_name="final-hist-gradient-boosting"
    ):
        mlflow.log_param(
            "model",
            "hist_gradient_boosting",
        )
        mlflow.log_param(
            "training_rows",
            training_rows,
        )
        mlflow.log_param(
            "scoring_snapshot",
            scoring_snapshot,
        )
        mlflow.log_param(
            "label_availability_cutoff",
            label_availability_cutoff,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_example.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    if model_info.registered_model_version is None:
        raise RuntimeError(
            "MLflow did not return a registered model version."
        )

    model_version = str(
        model_info.registered_model_version
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version,
    )

    return model_version