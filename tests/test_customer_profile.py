from __future__ import annotations

import pandas as pd

from src.intelligence.customer_profile import build_customer_profile


def test_build_customer_profile() -> None:
    predictions = pd.DataFrame(
        [
            {
                "customer_id": 100001,
                "snapshot_date": pd.Timestamp("2026-05-24"),
                "churn_probability": 0.80,
                "risk_rank": 1,
                "risk_percentile": 0.01,
                "risk_tier": "High",
                "primary_driver": "Engagement Decline",
                "secondary_driver": "Service Friction",
                "Engagement Decline": 1.2,
                "Low Recent Engagement": 0.1,
                "Inactivity / Recency": 0.0,
                "Service Friction": 0.8,
            }
        ]
    )

    features = pd.DataFrame(
        [
            {
                "customer_id": 100001,
                "snapshot_date": pd.Timestamp("2026-05-24"),
                "university": "XJTLU",
                "country": "China",
                "class_year": "Year 4",
                "academic_group": "Business & Economics",
                "plan_type": "Free",
                "acquisition_channel": "Social",
                "tenure_days": 172,
                "active_days_30d": 26,
                "logins_30d": 46,
                "minutes_30d": 1070.3,
                "assignments_30d": 11,
                "courses_started_30d": 2,
                "courses_completed_30d": 1,
                "days_since_last_active": 1,
                "active_days_trend_30d": 0.13,
                "logins_trend_30d": 0.33,
                "minutes_trend_30d": 0.47,
                "assignments_trend_30d": 0.39,
                "transaction_count": 1.0,
                "payment_success_rate": 1.0,
                "failed_payment_rate": 0.0,
                "pending_payment_rate": 0.0,
                "avg_discount_rate": 0.0,
                "discount_purchase_share": 0.0,
                "interaction_count": 8.0,
                "complaint_rate": 0.0,
                "refund_request_rate": 0.125,
                "resolved_issue_rate": 0.0,
                "avg_resolution_hours": 63.8,
                "avg_satisfaction_score": 3.5,
                "days_since_last_interaction": 0.0,
            }
        ]
    )

    profile = build_customer_profile(
        customer_id=100001,
        predictions=predictions,
        features=features,
    )

    assert profile["customer_id"] == 100001
    assert profile["snapshot_date"] == "2026-05-24"

    assert profile["risk"]["risk_tier"] == "High"
    assert profile["risk"]["risk_rank"] == 1

    assert (
        profile["drivers"]["primary"]
        == "Engagement Decline"
    )

    assert (
        profile["customer_context"]["university"]
        == "XJTLU"
    )

    assert (
        profile["engagement"]["minutes_30d"]
        == 1070.3
    )

    assert (
        profile["service"]["refund_request_rate"]
        == 0.125
    )