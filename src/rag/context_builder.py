from __future__ import annotations

from typing import Any

import pandas as pd


def build_rag_context(
    profile: dict[str, Any],
    recommendation: dict[str, Any],
    retrieved_chunks: pd.DataFrame,
) -> str:
    """Assemble grounded customer and knowledge context for the LLM."""

    if not profile:
        raise ValueError("profile must not be empty.")

    if not recommendation:
        raise ValueError("recommendation must not be empty.")

    if retrieved_chunks.empty:
        raise ValueError("retrieved_chunks must not be empty.")

    risk = profile.get("risk", {})
    drivers = profile.get("drivers", {})

    customer_lines = [
        f"Customer ID: {profile.get('customer_id')}",
        f"Risk tier: {risk.get('risk_tier')}",
        f"Churn probability: {risk.get('churn_probability')}",
        f"Primary driver: {drivers.get('primary')}",
        f"Secondary driver: {drivers.get('secondary')}",
        (
            "Recommended action: "
            f"{recommendation.get('recommended_action')}"
        ),
        (
            "Action objective: "
            f"{recommendation.get('objective')}"
        ),
    ]

    supporting_signals = recommendation.get(
        "supporting_signals",
        [],
    )

    if supporting_signals:
        customer_lines.append("Supporting signals:")

        for signal in supporting_signals:
            customer_lines.append(
                f"- {signal}"
            )

    knowledge_sections: list[str] = []

    for rank, row in enumerate(
        retrieved_chunks.itertuples(),
        start=1,
    ):
        knowledge_sections.append(
            "\n".join(
                [
                    f"[Knowledge {rank}]",
                    f"Source: {row.source}",
                    f"Section: {row.section_path}",
                    f"Retrieval score: {row.similarity_score:.4f}",
                    "",
                    row.content,
                ]
            )
        )

    context = "\n\n".join(
        [
            "CUSTOMER RISK CONTEXT",
            "\n".join(customer_lines),
            "RETRIEVED BUSINESS KNOWLEDGE",
            "\n\n".join(knowledge_sections),
        ]
    )

    return context