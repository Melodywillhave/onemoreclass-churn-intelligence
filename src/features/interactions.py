import numpy as np
import pandas as pd


def _build_interaction_history(
    labels: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build historical interaction aggregates as of each snapshot date.
    """

    rows = []

    for snapshot_date in sorted(labels["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)

        customers_at_snapshot = labels.loc[
            labels["snapshot_date"] == snapshot_date,
            ["customer_id"],
        ]

        history = interactions.loc[
            interactions["interaction_date"] <= snapshot_date
        ].copy()

        aggregated = (
            history
            .groupby("customer_id")
            .agg(
                interaction_count=("interaction_id", "count"),
                complaint_count=("complaint_flag", "sum"),
                refund_request_count=("refund_request_flag", "sum"),
                resolved_issue_rate=("issue_resolved", "mean"),
                avg_resolution_hours=("resolution_time_hours", "mean"),
                avg_satisfaction_score=("satisfaction_score", "mean"),
                last_interaction_date=("interaction_date", "max"),
            )
            .reset_index()
        )

        features = customers_at_snapshot.merge(
            aggregated,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        features["snapshot_date"] = snapshot_date

        features["days_since_last_interaction"] = (
            snapshot_date
            - features["last_interaction_date"]
        ).dt.days

        rows.append(features)

    return pd.concat(rows, ignore_index=True)


def build_interaction_features(
    labels: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe service interaction features at the
    customer-snapshot level.
    """

    labels = labels.copy()
    interactions = interactions.copy()

    labels["snapshot_date"] = pd.to_datetime(
        labels["snapshot_date"]
    )

    interactions["interaction_date"] = pd.to_datetime(
        interactions["interaction_date"]
    )

    features = _build_interaction_history(
        labels=labels,
        interactions=interactions,
    )

    features["has_interaction_history"] = (
        features["interaction_count"]
        .notna()
        .astype(int)
    )

    # Preserve NaN for customers with no interaction history.
    features["complaint_rate"] = np.where(
        features["interaction_count"].notna(),
        features["complaint_count"]
        / features["interaction_count"],
        np.nan,
    )

    features["refund_request_rate"] = np.where(
        features["interaction_count"].notna(),
        features["refund_request_count"]
        / features["interaction_count"],
        np.nan,
    )

    return features.drop(
        columns=["last_interaction_date"],
        errors="ignore",
    )