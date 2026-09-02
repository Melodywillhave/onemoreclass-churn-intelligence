from __future__ import annotations

import numpy as np
import pandas as pd


DRIVER_RULES = {
    "Engagement Decline": [
        "active_days_trend_30d",
        "logins_trend_30d",
        "minutes_trend_30d",
        "assignments_trend_30d",
    ],
    "Low Recent Engagement": [
        "active_days_30d",
        "logins_30d",
        "minutes_30d",
        "assignments_30d",
    ],
    "Inactivity / Recency": [
        "days_since_last_active",
    ],
    "Service Friction": [
        "complaint_rate",
        "refund_request_rate",
        "avg_resolution_hours",
    ],
}


def build_customer_driver_scores(
    scoring_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create interpretable customer-level behavioral risk-driver scores."""

    if scoring_df is None or scoring_df.empty:
        raise ValueError("scoring_df must not be empty.")

    result = scoring_df[
        ["customer_id", "snapshot_date"]
    ].copy()

    driver_scores: dict[str, pd.Series] = {}

    for driver_name, features in DRIVER_RULES.items():
        available_features = [
            feature
            for feature in features
            if feature in scoring_df.columns
        ]

        if not available_features:
            continue

        standardized_features = []

        for feature in available_features:
            values = pd.to_numeric(
                scoring_df[feature],
                errors="coerce",
            )

            median = values.median()
            std = values.std()

            if pd.isna(std) or std == 0:
                standardized = pd.Series(
                    0.0,
                    index=values.index,
                )
            else:
                standardized = (
                    values.fillna(median) - median
                ) / std

            if driver_name == "Inactivity / Recency":
                standardized = pd.cut(
                    values.fillna(0),
                    bins=[
                        -np.inf,
                        2,
                        6,
                        13,
                        29,
                        np.inf,
                    ],
                    labels=[
                        0.0,
                        0.5,
                        1.0,
                        2.0,
                        3.0,
                    ],
                ).astype(float)

            elif driver_name == "Low Recent Engagement":
                standardized = -standardized

            standardized_features.append(
                standardized
            )

        driver_scores[driver_name] = pd.concat(
            standardized_features,
            axis=1,
        ).mean(axis=1)

    if not driver_scores:
        raise ValueError(
            "No driver features were available in scoring_df."
        )

    driver_frame = pd.DataFrame(
        driver_scores,
        index=scoring_df.index,
    )

    result = pd.concat(
        [
            result.reset_index(drop=True),
            driver_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    driver_columns = list(driver_scores)

    driver_values = result[
        driver_columns
    ].to_numpy()

    driver_order = np.argsort(
        -driver_values,
        axis=1,
    )

    driver_names = np.asarray(
        driver_columns
    )

    primary_indices = driver_order[:, 0]
    primary_scores = driver_values[
        np.arange(len(result)),
        primary_indices,
    ]

    result["primary_driver"] = np.where(
        primary_scores > 0,
        driver_names[primary_indices],
        "No Strong Risk Signal",
    )

    if len(driver_columns) > 1:
        secondary_indices = driver_order[:, 1]
        secondary_scores = driver_values[
            np.arange(len(result)),
            secondary_indices,
        ]

        result["secondary_driver"] = np.where(
            secondary_scores > 0,
            driver_names[secondary_indices],
            "No Secondary Risk Signal",
        )
    else:
        result["secondary_driver"] = (
            "No Secondary Risk Signal"
        )

    return result