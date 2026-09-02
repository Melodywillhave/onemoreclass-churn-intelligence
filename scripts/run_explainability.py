from pathlib import Path

import joblib
import pandas as pd

from src.modeling.explain import calculate_permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_HORIZON_DAYS = 30


def main() -> None:
    """Calculate global feature importance for the production churn model."""

    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_features.parquet"
    )

    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "churn_model_hgb.joblib"
    )

    modeling_df = pd.read_parquet(
        data_path
    )

    modeling_df["snapshot_date"] = pd.to_datetime(
        modeling_df["snapshot_date"]
    )

    latest_snapshot = modeling_df[
        "snapshot_date"
    ].max()

    evaluation_df = modeling_df.loc[
        modeling_df["snapshot_date"]
        == latest_snapshot
    ].copy()

    if evaluation_df.empty:
        raise ValueError(
            "Evaluation dataset must not be empty."
        )

    model = joblib.load(
        model_path
    )

    importance = calculate_permutation_importance(
        model=model,
        evaluation_df=evaluation_df,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
    )

    print("\nGlobal Feature Importance\n")

    print(
        importance.head(20).to_string(
            index=False
        )
    )

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
        / "feature_importance_hgb.csv"
    )

    importance.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nEvaluation snapshot: {latest_snapshot.date()}"
    )

    print(
        f"Saved feature importance to: {output_path}"
    )


if __name__ == "__main__":
    main()