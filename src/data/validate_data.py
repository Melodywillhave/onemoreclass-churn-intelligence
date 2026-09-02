import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_data import generate_latent_profiles

RAW_DATA_DIR = Path("data/raw")


def load_data():

    customers = pd.read_csv(
        RAW_DATA_DIR / "customers.csv",
        parse_dates=["signup_date"]
    )

    courses = pd.read_csv(
        RAW_DATA_DIR / "courses.csv"
    )

    activities = pd.read_csv(
        RAW_DATA_DIR / "activities.csv",
        parse_dates=["activity_date"]
    )

    transactions = pd.read_csv(
        RAW_DATA_DIR / "transactions.csv",
        parse_dates=["transaction_date"]
    )

    interactions = pd.read_csv(
        RAW_DATA_DIR / "interactions.csv",
        parse_dates=["interaction_date"]
    )

    return (
        customers,
        courses,
        activities,
        transactions,
        interactions
    )

def validate_basic_quality(
    customers,
    courses,
    activities,
    transactions,
    interactions
):

    print("\n=== BASIC DATA QUALITY ===")

    datasets = {
        "customers": customers,
        "courses": courses,
        "activities": activities,
        "transactions": transactions,
        "interactions": interactions,
    }

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        print("Shape:")
        print(df.shape)

        print("\nMissing values:")
        print(
            df.isna().sum()
        )

        print("\nDuplicate rows:")
        print(
            df.duplicated().sum()
        )

def validate_referential_integrity(
    customers,
    courses,
    activities,
    transactions,
    interactions
):

    print("\n=== REFERENTIAL INTEGRITY ===")

    customer_ids = set(
        customers["customer_id"]
    )

    course_ids = set(
        courses["course_id"]
    )

    activity_invalid = (
        ~activities["customer_id"]
        .isin(customer_ids)
    ).sum()

    transaction_customer_invalid = (
        ~transactions["customer_id"]
        .isin(customer_ids)
    ).sum()

    interaction_invalid = (
        ~interactions["customer_id"]
        .isin(customer_ids)
    ).sum()

    transaction_course_invalid = (
        ~transactions["course_id"]
        .isin(course_ids)
    ).sum()

    print(
        "Invalid activity customer IDs:",
        activity_invalid
    )

    print(
        "Invalid transaction customer IDs:",
        transaction_customer_invalid
    )

    print(
        "Invalid interaction customer IDs:",
        interaction_invalid
    )

    print(
        "Invalid transaction course IDs:",
        transaction_course_invalid
    )

def validate_course_prices(courses):

    print("\n=== COURSE PRICE VALIDATION ===")

    checks = {
        "XJTLU recorded": (
            courses.loc[
                courses["university"] == "XJTLU",
                "recorded_price"
            ].between(59, 299).all()
        ),

        "XJTLU live": (
            courses.loc[
                courses["university"] == "XJTLU",
                "live_price"
            ].between(699, 999).all()
        ),

        "UoL recorded": (
            courses.loc[
                courses["university"] == "UoL",
                "recorded_price"
            ].between(39, 199).all()
        ),

        "UoL live": (
            courses.loc[
                courses["university"] == "UoL",
                "live_price"
            ].between(399, 699).all()
        ),
    }

    for check, result in checks.items():
        print(
            f"{check}:",
            "PASS" if result else "FAIL"
        )
def validate_currency(
    customers,
    transactions
):

    print("\n=== CURRENCY VALIDATION ===")

    df = transactions.merge(
        customers[
            ["customer_id", "university"]
        ],
        on="customer_id",
        how="left"
    )

    invalid_xjtlu = df[
        (df["university"] == "XJTLU")
        & (df["currency"] != "RMB")
    ]

    invalid_uol = df[
        (df["university"] == "UoL")
        & (df["currency"] != "GBP")
    ]

    print(
        "Invalid XJTLU currency:",
        len(invalid_xjtlu)
    )

    print(
        "Invalid UoL currency:",
        len(invalid_uol)
    )

def validate_transaction_patterns(
    transactions
):

    print("\n=== TRANSACTION PATTERNS ===")

    live_share = (
        transactions
        .assign(
            is_live=(
                transactions["delivery_mode"]
                == "Live"
            ).astype(int)
        )
        .groupby("purchase_period")[
            "is_live"
        ]
        .mean()
        .reindex(
            [
                "semester_start",
                "midterm",
                "final"
            ]
        )
    )

    print("\nLive share by period:")
    print(live_share)

def validate_interactions(
    interactions
):

    print("\n=== INTERACTION VALIDATION ===")

    print("\nInteraction type distribution:")

    print(
        interactions[
            "interaction_type"
        ].value_counts(
            normalize=True
        )
    )

    print("\nResolution rate:")

    service = interactions[
        interactions[
            "issue_resolved"
        ].notna()
    ]

    print(
        service[
            "issue_resolved"
        ].mean()
    )

    print("\nAverage resolution time:")

    print(
        service[
            "resolution_time_hours"
        ].mean()
    )

    print("\nAverage satisfaction:")

    print(
        interactions[
            "satisfaction_score"
        ].mean()
    )


def validate_behavior_type_alignment(
    customers,
    activities,
    transactions,
    interactions,
):
    """
    Validation-only diagnostic.

    We temporarily expose hidden latent behavior_type to verify whether the
    observable data matches the intended synthetic decline pattern.
    This is NOT part of the final ML dataset and must never be used as a feature.

    The comparison is intentionally per-customer and window-based:
    recent_30d vs previous_30d, not a simple daily-average difference.
    This avoids dilution from zero-login days and mirrors future churn feature logic.
    """

    print("\n=== LATENT BEHAVIOR VALIDATION (DEBUG ONLY) ===")
    print("NOTE: behavior_type is intentionally exposed here for validation only.")
    print("Do not include this in the ML feature set.")

    latent_profiles = generate_latent_profiles(customers)
    customer_marks = customers[["customer_id"]].merge(
        latent_profiles,
        on="customer_id",
        how="left"
    )

    activity = activities.copy()
    activity["activity_date"] = pd.to_datetime(activity["activity_date"])

    end_date = activity["activity_date"].max()
    prior_window_end = end_date - pd.Timedelta(days=30)
    prior_window_start = prior_window_end - pd.Timedelta(days=30)

    recent_window = activity[
        activity["activity_date"] >= prior_window_end
    ].copy()
    previous_window = activity[
        (activity["activity_date"] >= prior_window_start)
        & (activity["activity_date"] < prior_window_end)
    ].copy()

    recent_activity = recent_window.groupby("customer_id").agg(
        recent_login_sum=("login_count", "sum"),
        recent_video_sum=("videos_watched", "sum"),
        recent_minutes_sum=("minutes_watched", "sum"),
        recent_courses_started=("courses_started", "sum"),
        recent_assignments_completed=("assignments_completed", "sum"),
    )

    previous_activity = previous_window.groupby("customer_id").agg(
        prior_login_sum=("login_count", "sum"),
        prior_video_sum=("videos_watched", "sum"),
        prior_minutes_sum=("minutes_watched", "sum"),
        prior_courses_started=("courses_started", "sum"),
        prior_assignments_completed=("assignments_completed", "sum"),
    )

    engagement = customer_marks.merge(
        recent_activity,
        on="customer_id",
        how="left"
    ).merge(
        previous_activity,
        on="customer_id",
        how="left"
    ).fillna(0)

    metrics = [
        ("login_sum", "recent_login_sum", "prior_login_sum"),
        ("video_sum", "recent_video_sum", "prior_video_sum"),
        ("minutes_sum", "recent_minutes_sum", "prior_minutes_sum"),
        ("courses_started", "recent_courses_started", "prior_courses_started"),
        ("assignments_completed", "recent_assignments_completed", "prior_assignments_completed"),
    ]

    for metric_name, recent_col, prior_col in metrics:
        prior_values = engagement[prior_col].replace(0, np.nan)
        engagement[f"{metric_name}_drop_rate"] = (
            (engagement[prior_col] - engagement[recent_col]) / prior_values
        ).fillna(0.0)

    issue_events = interactions.copy()
    issue_events["is_issue"] = issue_events["interaction_type"].isin(
        {"support_ticket", "complaint", "refund_request", "billing_inquiry"}
    )
    issue_events["interaction_date"] = pd.to_datetime(issue_events["interaction_date"])

    recent_interactions = issue_events[
        issue_events["interaction_date"] >= prior_window_end
    ].groupby("customer_id").agg(
        recent_issue_count=("is_issue", "sum"),
        recent_complaint_count=("complaint_flag", "sum"),
        recent_refund_count=("refund_request_flag", "sum"),
    )

    previous_interactions = issue_events[
        (issue_events["interaction_date"] >= prior_window_start)
        & (issue_events["interaction_date"] < prior_window_end)
    ].groupby("customer_id").agg(
        prior_issue_count=("is_issue", "sum"),
        prior_complaint_count=("complaint_flag", "sum"),
        prior_refund_count=("refund_request_flag", "sum"),
    )

    validation_df = engagement.merge(
        recent_interactions,
        on="customer_id",
        how="left"
    ).merge(
        previous_interactions,
        on="customer_id",
        how="left"
    ).fillna(0)

    validation_df["issue_increase"] = (
        validation_df["recent_issue_count"] - validation_df["prior_issue_count"]
    )
    validation_df["complaint_increase"] = (
        validation_df["recent_complaint_count"] - validation_df["prior_complaint_count"]
    )
    validation_df["refund_increase"] = (
        validation_df["recent_refund_count"] - validation_df["prior_refund_count"]
    )

    summary = validation_df.groupby("behavior_type").agg(
        login_drop_rate_mean=("login_sum_drop_rate", "mean"),
        video_drop_rate_mean=("video_sum_drop_rate", "mean"),
        minutes_drop_rate_mean=("minutes_sum_drop_rate", "mean"),
        issue_increase_mean=("issue_increase", "mean"),
        complaint_increase_mean=("complaint_increase", "mean"),
        refund_increase_mean=("refund_increase", "mean"),
    )

    print(summary.round(4))

    print("\n--- Expected business pattern ---")
    print("declining users should show a stronger recent-vs-prior drop in engagement and a stronger increase in service issues than stable users.")
    print("This is only a validation check; it is not a feature to feed into ML models.")

    return summary

if __name__ == "__main__":

    (
        customers,
        courses,
        activities,
        transactions,
        interactions
    ) = load_data()

    validate_basic_quality(
        customers,
        courses,
        activities,
        transactions,
        interactions
    )

    validate_referential_integrity(
        customers,
        courses,
        activities,
        transactions,
        interactions
    )

    validate_course_prices(
        courses
    )

    validate_currency(
        customers,
        transactions
    )

    validate_transaction_patterns(
        transactions
    )

    validate_interactions(
        interactions
    )

    validate_behavior_type_alignment(
        customers,
        activities,
        transactions,
        interactions,
    )