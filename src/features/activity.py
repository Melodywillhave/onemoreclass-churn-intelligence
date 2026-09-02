import numpy as np
import pandas as pd

ACTIVITY_COLUMNS = [
    "login_count",
    "videos_watched",
    "minutes_watched",
    "courses_started",
    "courses_completed",
    "assignments_completed",
]


def _build_window_features(
    labels: pd.DataFrame,
    activities: pd.DataFrame,
    window_days: int,
) -> pd.DataFrame:
    """
    Build customer-snapshot activity aggregates over a fixed lookback window.
    """

    rows = []

    for snapshot_date in sorted(labels["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)

        customers_at_snapshot = labels.loc[
            labels["snapshot_date"] == snapshot_date,
            ["customer_id"],
        ]

        window_start = (
            snapshot_date
            - pd.Timedelta(days=window_days - 1)
        )

        window = activities.loc[
            (activities["activity_date"] >= window_start)
            & (activities["activity_date"] <= snapshot_date)
        ].copy()

        window["is_active"] = (
            window[ACTIVITY_COLUMNS]
            .sum(axis=1)
            .gt(0)
            .astype(int)
        )

        aggregated = (
            window
            .groupby("customer_id")
            .agg(
                active_days=("is_active", "sum"),
                logins=("login_count", "sum"),
                videos=("videos_watched", "sum"),
                minutes=("minutes_watched", "sum"),
                courses_started=("courses_started", "sum"),
                courses_completed=("courses_completed", "sum"),
                assignments=("assignments_completed", "sum"),
            )
            .reset_index()
        )

        aggregated = customers_at_snapshot.merge(
            aggregated,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        feature_cols = [
            "active_days",
            "logins",
            "videos",
            "minutes",
            "courses_started",
            "courses_completed",
            "assignments",
        ]

        aggregated[feature_cols] = (
            aggregated[feature_cols]
            .fillna(0)
        )

        aggregated["snapshot_date"] = snapshot_date

        aggregated = aggregated.rename(
            columns={
                col: f"{col}_{window_days}d"
                for col in feature_cols
            }
        )

        rows.append(aggregated)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def _safe_change_rate(
    previous: pd.Series,
    recent: pd.Series,
) -> np.ndarray:
    """
    Measure relative decline from a previous period to a recent period.

    Positive values indicate decline.
    Negative values indicate improvement.
    """

    return np.where(
        previous > 0,
        (previous - recent) / previous,
        np.where(
            recent == 0,
            0.0,
            -1.0,
        ),
    )


def _build_activity_recency(
    labels: pd.DataFrame,
    activities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate days since each customer's most recent active day
    as of each snapshot date.
    """

    activity_data = activities.copy()

    activity_data["is_active"] = (
        activity_data[ACTIVITY_COLUMNS]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    active_activity = activity_data.loc[
        activity_data["is_active"] == 1
    ]

    rows = []

    for snapshot_date in sorted(labels["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)

        customers_at_snapshot = labels.loc[
            labels["snapshot_date"] == snapshot_date,
            ["customer_id"],
        ]

        last_activity = (
            active_activity.loc[
                active_activity["activity_date"] <= snapshot_date
            ]
            .groupby("customer_id")["activity_date"]
            .max()
            .reset_index()
        )

        recency = customers_at_snapshot.merge(
            last_activity,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        recency["snapshot_date"] = snapshot_date

        recency["days_since_last_active"] = (
            snapshot_date
            - recency["activity_date"]
        ).dt.days

        rows.append(
            recency[
                [
                    "customer_id",
                    "snapshot_date",
                    "days_since_last_active",
                ]
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_activity_features(
    labels: pd.DataFrame,
    activities: pd.DataFrame,
    windows=(7, 14, 30, 60),
) -> pd.DataFrame:
    """
    Build leakage-safe activity features at the customer-snapshot level.

    Features include:
    - multi-window engagement aggregates
    - recent vs previous 30-day behavioral trends
    - activity recency
    - engagement intensity and completion ratios
    """

    activities = activities.copy()

    activities["activity_date"] = pd.to_datetime(
        activities["activity_date"]
    )

    labels = labels.copy()

    labels["snapshot_date"] = pd.to_datetime(
        labels["snapshot_date"]
    )

    # Multi-window activity aggregates
    feature_tables = [
        _build_window_features(
            labels=labels,
            activities=activities,
            window_days=window_days,
        )
        for window_days in windows
    ]

    features = feature_tables[0]

    for table in feature_tables[1:]:
        features = features.merge(
            table,
            on=[
                "customer_id",
                "snapshot_date",
            ],
            how="left",
            validate="one_to_one",
        )

    # Previous vs recent 30-day behavior
    trend_metrics = [
        "active_days",
        "logins",
        "minutes",
        "assignments",
    ]

    for metric in trend_metrics:
        features[f"{metric}_prev30d"] = (
            features[f"{metric}_60d"]
            - features[f"{metric}_30d"]
        )

        features[f"{metric}_trend_30d"] = (
            _safe_change_rate(
                previous=features[f"{metric}_prev30d"],
                recent=features[f"{metric}_30d"],
            )
        )

    # Activity recency
    recency_features = _build_activity_recency(
        labels=labels,
        activities=activities,
    )

    features = features.merge(
        recency_features,
        on=[
            "customer_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # Engagement intensity
    features["active_day_ratio_30d"] = (
        features["active_days_30d"] / 30
    )

    features["active_day_ratio_60d"] = (
        features["active_days_60d"] / 60
    )

    for metric in [
        "minutes",
        "logins",
        "assignments",
    ]:
        features[f"{metric}_per_active_day_30d"] = np.where(
            features["active_days_30d"] > 0,
            features[f"{metric}_30d"]
            / features["active_days_30d"],
            0,
        )

    for window in [30, 60]:
        features[f"course_completion_rate_{window}d"] = np.where(
            features[f"courses_started_{window}d"] > 0,
            features[f"courses_completed_{window}d"]
            / features[f"courses_started_{window}d"],
            0,
        )

    return features