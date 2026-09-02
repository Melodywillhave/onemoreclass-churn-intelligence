from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_customer_risk_endpoint() -> None:
    response = client.get(
        "/customers/104431/risk"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == 104431
    assert data["risk_tier"] == "High"
    assert data["risk_rank"] == 1
    assert 0 <= data["churn_probability"] <= 1


def test_customer_risk_not_found() -> None:
    response = client.get(
        "/customers/999999/risk"
    )

    assert response.status_code == 404