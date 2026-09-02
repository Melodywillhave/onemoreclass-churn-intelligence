from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

DRIVER_DOCUMENT_MAP = {
    "Engagement Decline": [
        "engagement_playbook.md",
        "retention_playbook.md",
        "offer_policy.md",
    ],
    "Low Recent Engagement": [
        "engagement_playbook.md",
        "retention_playbook.md",
        "offer_policy.md",
    ],
    "Service Friction": [
        "service_recovery_playbook.md",
        "retention_playbook.md",
        "offer_policy.md",
    ],
    "Inactivity / Recency": [
        "engagement_playbook.md",
        "retention_playbook.md",
    ],
    "No Strong Risk Signal": [
        "retention_playbook.md",
    ],
}


def _load_document(filename: str) -> str:
    """Load a knowledge-base document."""

    path = KNOWLEDGE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge document not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def retrieve_knowledge(
    recommendation: dict[str, Any],
) -> list[dict[str, str]]:
    """Retrieve relevant knowledge documents for a retention decision."""

    if not recommendation:
        raise ValueError("recommendation must not be empty.")

    primary_driver = recommendation.get("primary_driver")
    secondary_driver = recommendation.get("secondary_driver")

    filenames: list[str] = []

    for driver in [primary_driver, secondary_driver]:
        if driver in DRIVER_DOCUMENT_MAP:
            filenames.extend(
                DRIVER_DOCUMENT_MAP[driver]
            )

    # Preserve ranking while removing duplicate documents.
    filenames = list(dict.fromkeys(filenames))

    if not filenames:
        filenames = ["retention_playbook.md"]

    retrieved_documents = []

    for filename in filenames:
        retrieved_documents.append(
            {
                "source": filename,
                "content": _load_document(filename),
            }
        )

    return retrieved_documents