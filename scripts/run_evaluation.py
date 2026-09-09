from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling.evaluate import evaluate_walk_forward
from src.modeling.mlflow_tracking import log_evaluation_results
from src.modeling.walk_forward import generate_purged_walk_forward_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Evaluate walk-forward folds, save results, and log them to MLflow."""
    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_features.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Input data not found: {data_path}"
        )

    df = pd.read_parquet(data_path)

    folds = generate_purged_walk_forward_splits(
        df=df,
        date_col="snapshot_date",
        prediction_horizon_days=30,
        min_train_snapshots=1,
    )

    metrics_df = evaluate_walk_forward(folds)
    print(metrics_df)

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "walk_forward_metrics_model_comparison.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    log_evaluation_results(
        metrics_df=metrics_df,
        artifact_path=output_path,
    )

    print(
        f"Saved evaluation results to: {output_path}"
    )
    print(
        "Logged evaluation results to MLflow experiment: "
        "churn-model-development"
    )


if __name__ == "__main__":
    main()