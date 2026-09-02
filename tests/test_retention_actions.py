from __future__ import annotations

from src.intelligence.retention_actions import (
    recommend_retention_action,
)


def test_engagement_decline_action() -> None:
    profile = {
        "customer_id": 100001,
        "risk": {
            "risk_tier": "High",
        },
        "drivers": {
            "primary": "Engagement Decline",
            "secondary": "Service Friction",
        },
        "engagement_trends": {
            "minutes_trend_30d": 0.50,
            "logins_trend_30d": 0.30,
            "assignments_trend_30d": 0.40,
        },
        "service": {
            "refund_request_rate": 0.10,
            "resolved_issue_rate": 0.25,
            "avg_resolution_hours": 60.0,
        },
        "engagement": {
            "days_since_last_active": 1,
        },
    }

    result = recommend_retention_action(profile)

    assert (
        result["recommended_action"]
        == "Re-engagement Outreach"
    )
    assert result["action_priority"] == "High"
    assert result["primary_driver"] == "Engagement Decline"


def test_service_friction_action() -> None:
    profile = {
        "customer_id": 100002,
        "risk": {
            "risk_tier": "High",
        },
        "drivers": {
            "primary": "Service Friction",
            "secondary": "No Secondary Risk Signal",
        },
        "engagement_trends": {
            "minutes_trend_30d": 0.0,
            "logins_trend_30d": 0.0,
            "assignments_trend_30d": 0.0,
        },
        "service": {
            "refund_request_rate": 0.20,
            "resolved_issue_rate": 0.20,
            "avg_resolution_hours": 72.0,
        },
        "engagement": {
            "days_since_last_active": 2,
        },
    }

    result = recommend_retention_action(profile)

    assert (
        result["recommended_action"]
        == "Priority Service Recovery"
    )
    assert result["primary_driver"] == "Service Friction"