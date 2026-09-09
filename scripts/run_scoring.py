from pathlib import Path

import joblib
import pandas as pd

from src.modeling.drivers import build_customer_driver_scores
from src.modeling.model_registry import register_champion_model
from src.modeling.risk import assign_risk_tiers
from src.modeling.train import fit_hist_gradient_boosting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_HORIZON_DAYS = 30

def main() -> None:
    """Train the final HGB model and score the latest customer snapshot."""

    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_features.parquet"
    )

    modeling_df = pd.read_parquet(data_path)

    modeling_df["snapshot_date"] = pd.to_datetime(
        modeling_df["snapshot_date"]
    )

    latest_snapshot = modeling_df["snapshot_date"].max()

    label_availability_cutoff = (
    latest_snapshot
        - pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    )

    train_df = modeling_df.loc[
        modeling_df["snapshot_date"] < label_availability_cutoff
    ].copy()

    scoring_df = modeling_df.loc[
        modeling_df["snapshot_date"] == latest_snapshot
    ].copy()

    if train_df.empty:
        raise ValueError("Training dataset must not be empty.")

    if scoring_df.empty:
        raise ValueError("Scoring dataset must not be empty.")

    model = fit_hist_gradient_boosting(
        train_df
    )

    feature_columns = [
        column
        for column in scoring_df.columns
        if column not in {
            "customer_id",
            "snapshot_date",
            "churn",
        }
    ]

    X_score = scoring_df[
        feature_columns
    ].copy()

    churn_probabilities = model.predict_proba(
        X_score
    )[:, 1]

    risk_results = assign_risk_tiers(
        customer_ids=scoring_df["customer_id"].to_numpy(),
        y_prob=churn_probabilities,
        high_risk_rate=0.20,
        medium_risk_rate=0.30,
    )

    driver_results = build_customer_driver_scores(
        scoring_df
    )

    predictions = (
        scoring_df[
            [
                "customer_id",
                "snapshot_date",
            ]
        ]
        .merge(
            risk_results,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            driver_results,
            on=[
                "customer_id",
                "snapshot_date",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    predictions = predictions.sort_values(
        "risk_rank"
    ).reset_index(drop=True)

    predictions_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "predictions"
    )

    models_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        predictions_dir
        / "churn_predictions.parquet"
    )

    model_path = (
        models_dir
        / "churn_model_hgb.joblib"
    )

    predictions.to_parquet(
        predictions_path,
        index=False,
    )

    joblib.dump(
        model,
        model_path,
    )

    model_version = register_champion_model(
    model=model,
    X_example=X_score,
    training_rows=len(train_df),
    scoring_snapshot=str(latest_snapshot.date()),
    label_availability_cutoff=str(
        label_availability_cutoff.date()
    ),
)

    print(
        f"Training rows: {len(train_df):,}"
    )
    print(
        f"Scoring rows: {len(scoring_df):,}"
    )
    print(
        f"Latest snapshot: {latest_snapshot.date()}"
    )

    print("\nRisk tier distribution:")
    print(
        predictions["risk_tier"].value_counts()
    )

    print(
        f"\nSaved predictions to: {predictions_path}"
    )
    print(
        f"Saved model to: {model_path}"
    )
    print(
        "Label availability cutoff: "
        f"{label_availability_cutoff.date()}"
    )
    print(
    "Registered MLflow model: "
    f"churn-risk-model version {model_version}"
    )
    print(
        "Assigned MLflow alias: "
        "churn-risk-model@champion"
    )


if __name__ == "__main__":
    main()