from __future__ import annotations

import numpy as np
import pandas as pd


OBSERVATION_DAYS = 60
PREDICTION_DAYS = 30
SNAPSHOT_STEP_DAYS = 14

MIN_ACTIVE_DAYS = 15
MIN_WATCH_MINUTES = 500

LOGIN_DECLINE_THRESHOLD = 0.10
MINUTES_DECLINE_THRESHOLD = 0.20
ASSIGNMENT_DECLINE_THRESHOLD = 0.30


def safe_decline_rate(
    past: pd.Series,
    future: pd.Series,
) -> np.ndarray:
    """Calculate behavioral decline while safely handling zero baselines."""
    return np.where(
        past > 0,
        (past - future) / past,
        np.where(future == 0, 0.0, -1.0),
    )


def build_snapshot_labels(
    customers: pd.DataFrame,
    activities: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    observation_days: int = OBSERVATION_DAYS,
    prediction_days: int = PREDICTION_DAYS,
) -> pd.DataFrame:
    """Build past and future behavioral summaries for one snapshot."""
    observation_start = snapshot_date - pd.Timedelta(
        days=observation_days - 1
    )
    future_start = snapshot_date + pd.Timedelta(days=1)
    future_end = snapshot_date + pd.Timedelta(days=prediction_days)

    past = activities[
        activities["activity_date"].between(
            observation_start,
            snapshot_date,
        )
    ].copy()

    past["active_login_day"] = (
        past["login_count"] > 0
    ).astype(int)

    past_summary = (
        past.groupby("customer_id")
        .agg(
            past_active_days=("active_login_day", "sum"),
            past_logins=("login_count", "sum"),
            past_minutes=("minutes_watched", "sum"),
            past_assignments=("assignments_completed", "sum"),
        )
        .reset_index()
    )

    future = activities[
        activities["activity_date"].between(
            future_start,
            future_end,
        )
    ].copy()

    future["active_login_day"] = (
        future["login_count"] > 0
    ).astype(int)

    future_summary = (
        future.groupby("customer_id")
        .agg(
            future_active_days=("active_login_day", "sum"),
            future_logins=("login_count", "sum"),
            future_minutes=("minutes_watched", "sum"),
            future_assignments=("assignments_completed", "sum"),
        )
        .reset_index()
    )

    snapshot = (
        customers[["customer_id"]]
        .merge(past_summary, on="customer_id", how="left")
        .merge(future_summary, on="customer_id", how="left")
    )

    activity_cols = [
        "past_active_days",
        "past_logins",
        "past_minutes",
        "past_assignments",
        "future_active_days",
        "future_logins",
        "future_minutes",
        "future_assignments",
    ]

    snapshot[activity_cols] = snapshot[activity_cols].fillna(0)
    snapshot["snapshot_date"] = snapshot_date

    return snapshot

def build_churn_labels(
    customers: pd.DataFrame,
    activities: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final eligible customer-snapshot churn label dataset."""
    activity_start = activities["activity_date"].min()
    activity_end = activities["activity_date"].max()

    first_snapshot = (
        activity_start
        + pd.Timedelta(days=OBSERVATION_DAYS - 1)
    )

    last_snapshot = (
        activity_end
        - pd.Timedelta(days=PREDICTION_DAYS)
    )

    snapshot_dates = pd.date_range(
        start=first_snapshot,
        end=last_snapshot,
        freq=f"{SNAPSHOT_STEP_DAYS}D",
    )

    snapshot_list = []

    for snapshot_date in snapshot_dates:
        snapshot = build_snapshot_labels(
            customers=customers,
            activities=activities,
            snapshot_date=snapshot_date,
            observation_days=OBSERVATION_DAYS,
            prediction_days=PREDICTION_DAYS,
        )
        snapshot_list.append(snapshot)

    snapshot_panel = pd.concat(
        snapshot_list,
        ignore_index=True,
    )

    snapshot_panel["eligible"] = (
        (
            snapshot_panel["past_active_days"]
            >= MIN_ACTIVE_DAYS
        )
        & (
            snapshot_panel["past_minutes"]
            >= MIN_WATCH_MINUTES
        )
    ).astype(int)

    snapshot_panel["past_active_day_rate"] = (
        snapshot_panel["past_active_days"]
        / OBSERVATION_DAYS
    )

    snapshot_panel["future_active_day_rate"] = (
        snapshot_panel["future_active_days"]
        / PREDICTION_DAYS
    )

    snapshot_panel["past_minutes_per_day"] = (
        snapshot_panel["past_minutes"]
        / OBSERVATION_DAYS
    )

    snapshot_panel["future_minutes_per_day"] = (
        snapshot_panel["future_minutes"]
        / PREDICTION_DAYS
    )

    snapshot_panel["past_assignments_per_day"] = (
        snapshot_panel["past_assignments"]
        / OBSERVATION_DAYS
    )

    snapshot_panel["future_assignments_per_day"] = (
        snapshot_panel["future_assignments"]
        / PREDICTION_DAYS
    )

    snapshot_panel["future_login_decline"] = (
        safe_decline_rate(
            snapshot_panel["past_active_day_rate"],
            snapshot_panel["future_active_day_rate"],
        )
    )

    snapshot_panel["future_minutes_decline"] = (
        safe_decline_rate(
            snapshot_panel["past_minutes_per_day"],
            snapshot_panel["future_minutes_per_day"],
        )
    )

    snapshot_panel["future_assignment_decline"] = (
        safe_decline_rate(
            snapshot_panel["past_assignments_per_day"],
            snapshot_panel["future_assignments_per_day"],
        )
    )

    snapshot_panel["login_decline_flag"] = (
        snapshot_panel["future_login_decline"]
        >= LOGIN_DECLINE_THRESHOLD
    ).astype(int)

    snapshot_panel["minutes_decline_flag"] = (
        snapshot_panel["future_minutes_decline"]
        >= MINUTES_DECLINE_THRESHOLD
    ).astype(int)

    snapshot_panel["assignment_decline_flag"] = (
        snapshot_panel["future_assignment_decline"]
        >= ASSIGNMENT_DECLINE_THRESHOLD
    ).astype(int)

    snapshot_panel["decline_signal_count"] = (
        snapshot_panel["login_decline_flag"]
        + snapshot_panel["minutes_decline_flag"]
        + snapshot_panel["assignment_decline_flag"]
    )

    snapshot_panel["churn"] = np.where(
        snapshot_panel["eligible"] == 1,
        (
            snapshot_panel["decline_signal_count"]
            >= 2
        ).astype(int),
        np.nan,
    )

    modeling_panel = (
        snapshot_panel.loc[
            snapshot_panel["eligible"] == 1
        ]
        .copy()
    )

    modeling_panel["churn"] = (
        modeling_panel["churn"]
        .astype(int)
    )

    return modeling_panel[
        [
            "customer_id",
            "snapshot_date",
            "churn",
        ]
    ].copy()