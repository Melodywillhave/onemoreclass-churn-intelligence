from __future__ import annotations

from typing import Any


def build_retrieval_query(
    profile: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    """Build a retrieval query from customer risk and decision context."""

    if not profile:
        raise ValueError("profile must not be empty.")

    if not recommendation:
        raise ValueError("recommendation must not be empty.")

    risk = profile.get("risk", {})
    drivers = profile.get("drivers", {})

    risk_tier = risk.get("risk_tier")
    primary_driver = drivers.get("primary")
    secondary_driver = drivers.get("secondary")

    recommended_action = recommendation.get(
        "recommended_action"
    )

    supporting_signals = recommendation.get(
        "supporting_signals",
        [],
    )

    query_parts = [
        f"Risk tier: {risk_tier}.",
        f"Primary behavioral driver: {primary_driver}.",
        f"Secondary behavioral driver: {secondary_driver}.",
        f"Recommended retention action: {recommended_action}.",
    ]

    if supporting_signals:
        query_parts.append(
            "Customer risk signals: "
            + "; ".join(supporting_signals)
            + "."
        )

    return "\n".join(query_parts)