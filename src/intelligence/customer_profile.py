from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROFILE_FEATURES = {
    "customer_context": [
        "university",
        "country",
        "class_year",
        "academic_group",
        "plan_type",
        "acquisition_channel",
        "tenure_days",
    ],
    "engagement": [
        "active_days_30d",
        "logins_30d",
        "minutes_30d",
        "assignments_30d",
        "courses_started_30d",
        "courses_completed_30d",
        "days_since_last_active",
    ],
    "engagement_trends": [
        "active_days_trend_30d",
        "logins_trend_30d",
        "minutes_trend_30d",
        "assignments_trend_30d",
    ],
    "transactions": [
        "transaction_count",
        "payment_success_rate",
        "failed_payment_rate",
        "pending_payment_rate",
        "avg_discount_rate",
        "discount_purchase_share",
        "days_since_last_transaction",
    ],
    "service": [
        "interaction_count",
        "complaint_rate",
        "refund_request_rate",
        "resolved_issue_rate",
        "avg_resolution_hours",
        "avg_satisfaction_score",
        "days_since_last_interaction",
    ],
}


def _serialize_value(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-compatible Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def _extract_feature_group(
    row: pd.Series,
    features: list[str],
) -> dict[str, Any]:
    """Extract available features from one customer row."""

    return {
        feature: _serialize_value(row[feature])
        for feature in features
        if feature in row.index
    }


def build_customer_profile(
    customer_id: int,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Build a structured risk profile for one customer."""

    if predictions is None or predictions.empty:
        raise ValueError("predictions must not be empty.")

    if features is None or features.empty:
        raise ValueError("features must not be empty.")

    prediction_rows = predictions.loc[
        predictions["customer_id"] == customer_id
    ].copy()

    if prediction_rows.empty:
        raise ValueError(
            f"Customer {customer_id} was not found in predictions."
        )

    if len(prediction_rows) != 1:
        raise ValueError(
            f"Customer {customer_id} has multiple prediction rows."
        )

    prediction_row = prediction_rows.iloc[0]

    snapshot_date = pd.to_datetime(
        prediction_row["snapshot_date"]
    )

    feature_rows = features.loc[
        (features["customer_id"] == customer_id)
        & (
            pd.to_datetime(features["snapshot_date"])
            == snapshot_date
        )
    ].copy()

    if feature_rows.empty:
        raise ValueError(
            "No matching feature row found for "
            f"customer {customer_id} at {snapshot_date.date()}."
        )

    if len(feature_rows) != 1:
        raise ValueError(
            "Multiple feature rows found for "
            f"customer {customer_id} at {snapshot_date.date()}."
        )

    feature_row = feature_rows.iloc[0]

    profile = {
        "customer_id": _serialize_value(customer_id),
        "snapshot_date": snapshot_date.date().isoformat(),
        "risk": {
            "churn_probability": _serialize_value(
                prediction_row["churn_probability"]
            ),
            "risk_rank": _serialize_value(
                prediction_row["risk_rank"]
            ),
            "risk_percentile": _serialize_value(
                prediction_row["risk_percentile"]
            ),
            "risk_tier": _serialize_value(
                prediction_row["risk_tier"]
            ),
        },
        "drivers": {
            "primary": _serialize_value(
                prediction_row["primary_driver"]
            ),
            "secondary": _serialize_value(
                prediction_row["secondary_driver"]
            ),
            "scores": {
                "engagement_decline": _serialize_value(
                    prediction_row["Engagement Decline"]
                ),
                "low_recent_engagement": _serialize_value(
                    prediction_row["Low Recent Engagement"]
                ),
                "inactivity_recency": _serialize_value(
                    prediction_row["Inactivity / Recency"]
                ),
                "service_friction": _serialize_value(
                    prediction_row["Service Friction"]
                ),
            },
        },
    }

    for group_name, group_features in PROFILE_FEATURES.items():
        profile[group_name] = _extract_feature_group(
            feature_row,
            group_features,
        )

    return profile