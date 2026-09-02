from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.activity import build_activity_features
from src.features.interactions import build_interaction_features
from src.features.transactions import build_transaction_features
from src.features.validation import validate_feature_matrix


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

SNAPSHOT_KEY = ["customer_id", "snapshot_date"]
TARGET_COLUMN = "churn"
ACTIVITY_WINDOWS = [7, 14, 30, 60]

FORBIDDEN_FEATURES = {
    "future_active_days",
    "future_logins",
    "future_minutes",
    "future_assignments",
    "future_active_day_rate",
    "future_minutes_per_day",
    "future_assignments_per_day",
    "future_login_decline",
    "future_minutes_decline",
    "future_assignment_decline",
    "login_decline_flag",
    "minutes_decline_flag",
    "assignment_decline_flag",
    "decline_signal_count",
}

EXPECTED_NULLABLE_FEATURES = {
    "transaction_count",
    "paid_transactions",
    "failed_transactions",
    "pending_transactions",
    "avg_discount_rate",
    "discount_purchase_share",
    "live_purchase_share",
    "days_since_last_purchase",
    "payment_success_rate",
    "failed_payment_rate",
    "pending_payment_rate",
    "interaction_count",
    "complaint_count",
    "refund_request_count",
    "resolved_issue_rate",
    "avg_resolution_hours",
    "avg_satisfaction_score",
    "days_since_last_interaction",
    "complaint_rate",
    "refund_request_rate",
}


def main() -> None:
    """Build, validate, and persist the modeling feature matrix."""
    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv",
        parse_dates=["signup_date"],
    )

    activities = pd.read_csv(
        RAW_DATA_DIR / "activities.csv",
        parse_dates=["activity_date"],
    )

    transactions = pd.read_csv(
        RAW_DATA_DIR / "transactions.csv",
        parse_dates=["transaction_date"],
    )

    interactions = pd.read_csv(
        RAW_DATA_DIR / "interactions.csv",
        parse_dates=["interaction_date"],
    )

    modeling_labels = pd.read_csv(
        PROCESSED_DATA_DIR / "modeling_labels.csv",
        parse_dates=["snapshot_date"],
    )

    feature_base = modeling_labels.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    feature_base["tenure_days"] = (
        feature_base["snapshot_date"]
        - feature_base["signup_date"]
    ).dt.days

    activity_features = build_activity_features(
        labels=modeling_labels,
        activities=activities,
        windows=ACTIVITY_WINDOWS,
    )

    transaction_features = build_transaction_features(
        labels=modeling_labels,
        transactions=transactions,
    )

    interaction_features = build_interaction_features(
        labels=modeling_labels,
        interactions=interactions,
    )

    modeling_dataset = (
        feature_base
        .merge(
            activity_features,
            on=SNAPSHOT_KEY,
            how="left",
            validate="one_to_one",
        )
        .merge(
            transaction_features,
            on=SNAPSHOT_KEY,
            how="left",
            validate="one_to_one",
        )
        .merge(
            interaction_features,
            on=SNAPSHOT_KEY,
            how="left",
            validate="one_to_one",
        )
        .drop(
            columns=["signup_date"],
            errors="ignore",
        )
    )

    validate_feature_matrix(
        modeling_dataset,
        snapshot_key=SNAPSHOT_KEY,
        target_col=TARGET_COLUMN,
        forbidden_features=FORBIDDEN_FEATURES,
        expected_nullable=EXPECTED_NULLABLE_FEATURES,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "modeling_features.parquet"
    )

    modeling_dataset.to_parquet(
        output_path,
        index=False,
    )

    print("Feature matrix validation passed.")
    print(f"Shape: {modeling_dataset.shape}")
    print(
        "Snapshots: "
        f"{modeling_dataset['snapshot_date'].nunique()}"
    )
    print(
        "Churn rate: "
        f"{modeling_dataset['churn'].mean():.2%}"
    )
    print(f"Saved feature matrix to: {output_path}")


if __name__ == "__main__":
    main()