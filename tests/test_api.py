from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app as api_app
from src.persistence.database import (
    get_connection,
    initialize_database,
)


client = TestClient(api_app.app)


def setup_test_database(database_path: Path) -> None:
    """Create and seed an isolated SQLite database for API tests."""
    initialize_database(database_path)

    with get_connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO customer_risk (
                customer_id,
                snapshot_date,
                churn_probability,
                risk_rank,
                risk_percentile,
                risk_tier,
                primary_driver,
                secondary_driver
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                104431,
                "2026-05-24",
                0.73,
                1,
                0.01,
                "High",
                "Low Recent Engagement",
                "Service Friction",
            ),
        )

        connection.commit()


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_customer_risk_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "test_churn_intelligence.db"

    setup_test_database(database_path)

    def test_get_connection():
        return get_connection(database_path)

    monkeypatch.setattr(
        api_app,
        "get_connection",
        test_get_connection,
    )

    response = client.get(
        "/customers/104431/risk"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == 104431
    assert data["risk_tier"] == "High"
    assert data["risk_rank"] == 1
    assert 0 <= data["churn_probability"] <= 1


def test_customer_risk_not_found(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "test_churn_intelligence.db"

    setup_test_database(database_path)

    def test_get_connection():
        return get_connection(database_path)

    monkeypatch.setattr(
        api_app,
        "get_connection",
        test_get_connection,
    )

    response = client.get(
        "/customers/999999/risk"
    )

    assert response.status_code == 404