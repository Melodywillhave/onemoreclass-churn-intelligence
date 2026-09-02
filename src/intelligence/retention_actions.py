from __future__ import annotations

from typing import Any


ACTION_MAP = {
    "Engagement Decline": {
        "action": "Re-engagement Outreach",
        "objective": "Restore recent learning engagement",
    },
    "Low Recent Engagement": {
        "action": "Personalized Learning Recommendation",
        "objective": "Increase ongoing platform engagement",
    },
    "Inactivity / Recency": {
        "action": "Win-back Outreach",
        "objective": "Bring the customer back to the platform",
    },
    "Service Friction": {
        "action": "Priority Service Recovery",
        "objective": "Resolve service issues before disengagement",
    },
    "No Strong Risk Signal": {
        "action": "Manual Risk Review",
        "objective": (
            "Review additional customer context because model risk is high "
            "but no dominant behavioral driver was identified"
    ),
},
}


def _format_percentage(value: Any) -> str | None:
    """Convert a decimal rate into a readable percentage."""

    if value is None:
        return None

    return f"{float(value) * 100:.1f}%"


def _build_supporting_signals(
    profile: dict[str, Any],
) -> list[str]:
    """Build human-readable evidence from the customer profile."""

    signals: list[str] = []

    trends = profile.get(
        "engagement_trends",
        {},
    )

    service = profile.get(
        "service",
        {},
    )

    minutes_trend = trends.get(
        "minutes_trend_30d"
    )

    if minutes_trend is not None and minutes_trend > 0.20:
        signals.append(
            "Minutes watched declined "
            f"{_format_percentage(minutes_trend)}"
        )

    logins_trend = trends.get(
        "logins_trend_30d"
    )

    if logins_trend is not None and logins_trend > 0.20:
        signals.append(
            "Login activity declined "
            f"{_format_percentage(logins_trend)}"
        )

    assignments_trend = trends.get(
        "assignments_trend_30d"
    )

    if (
        assignments_trend is not None
        and assignments_trend > 0.20
    ):
        signals.append(
            "Assignment activity declined "
            f"{_format_percentage(assignments_trend)}"
        )

    refund_rate = service.get(
        "refund_request_rate"
    )

    if refund_rate is not None and refund_rate > 0:
        signals.append(
            "Refund request rate is "
            f"{_format_percentage(refund_rate)}"
        )

    resolved_rate = service.get(
        "resolved_issue_rate"
    )

    if resolved_rate is not None and resolved_rate < 0.50:
        signals.append(
            "Less than half of recorded service issues were resolved"
        )

    resolution_hours = service.get(
        "avg_resolution_hours"
    )

    if (
        resolution_hours is not None
        and resolution_hours > 48
    ):
        signals.append(
            "Average service resolution time exceeds 48 hours"
        )

    days_inactive = profile.get(
        "engagement",
        {},
    ).get(
        "days_since_last_active"
    )

    if days_inactive is not None and days_inactive >= 7:
        signals.append(
            f"No recorded learning activity for {int(days_inactive)} days"
        )

    return signals


def recommend_retention_action(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Generate a structured retention recommendation."""

    if not profile:
        raise ValueError(
            "profile must not be empty."
        )

    customer_id = profile.get(
        "customer_id"
    )

    risk = profile.get(
        "risk",
        {},
    )

    drivers = profile.get(
        "drivers",
        {},
    )

    risk_tier = risk.get(
        "risk_tier"
    )

    primary_driver = drivers.get(
        "primary",
        "No Strong Risk Signal",
    )

    secondary_driver = drivers.get(
        "secondary",
        "No Secondary Risk Signal",
    )

    action_config = ACTION_MAP.get(
        primary_driver,
        ACTION_MAP["No Strong Risk Signal"],
    )

    priority_map = {
        "High": "High",
        "Medium": "Medium",
        "Low": "Low",
    }

    action_priority = priority_map.get(
        risk_tier,
        "Low",
    )

    supporting_signals = (
        _build_supporting_signals(profile)
    )

    return {
        "customer_id": customer_id,
        "risk_tier": risk_tier,
        "primary_driver": primary_driver,
        "secondary_driver": secondary_driver,
        "recommended_action": action_config[
            "action"
        ],
        "action_priority": action_priority,
        "objective": action_config[
            "objective"
        ],
        "supporting_signals": supporting_signals,
    }