from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.intelligence.customer_profile import build_customer_profile
from src.intelligence.retention_actions import recommend_retention_action
from src.persistence.database import get_connection
from src.rag.context_builder import build_rag_context
from src.rag.generator import generate_recommendation
from src.rag.prompt_builder import build_generation_prompt
from src.rag.query_builder import build_retrieval_query
from src.rag.schemas import RetentionRecommendation
from src.rag.semantic_retriever import retrieve_chunks

PREDICTIONS_PATH = Path(
    "artifacts/predictions/churn_predictions.parquet"
)

FEATURES_PATH = Path(
    "data/processed/modeling_features.parquet"
)


app = FastAPI(
    title="Customer Churn Intelligence API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API is running."""

    return {
        "status": "healthy",
    }


@app.get("/customers/{customer_id}/risk")
def get_customer_risk(customer_id: int) -> dict:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                customer_id,
                snapshot_date,
                churn_probability,
                risk_rank,
                risk_percentile,
                risk_tier,
                primary_driver,
                secondary_driver
            FROM customer_risk
            WHERE customer_id = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (customer_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} was not found.",
        )

    return {
        "customer_id": row["customer_id"],
        "snapshot_date": row["snapshot_date"],
        "churn_probability": row["churn_probability"],
        "risk_rank": row["risk_rank"],
        "risk_percentile": row["risk_percentile"],
        "risk_tier": row["risk_tier"],
        "primary_driver": row["primary_driver"],
        "secondary_driver": row["secondary_driver"],
    }


@app.post(
    "/recommendations/{customer_id}",
    response_model=RetentionRecommendation,
)
def generate_customer_recommendation(
    customer_id: int,
) -> RetentionRecommendation:
    with get_connection() as connection:
        cached_row = connection.execute(
            """
            SELECT recommendation_json
            FROM recommendations
            WHERE customer_id = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (customer_id,),
        ).fetchone()

    if cached_row is not None:
        recommendation = json.loads(
            cached_row["recommendation_json"]
        )

        return RetentionRecommendation.model_validate(
            recommendation
        )

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    try:
        profile = build_customer_profile(
            customer_id=customer_id,
            predictions=predictions,
            features=features,
        )

        deterministic_recommendation = (
            recommend_retention_action(profile)
        )

        query = build_retrieval_query(
            profile,
            deterministic_recommendation,
        )

        retrieved_chunks = retrieve_chunks(
            query=query,
            top_k=5,
        )

        context = build_rag_context(
            profile,
            deterministic_recommendation,
            retrieved_chunks,
        )

        prompt = build_generation_prompt(context)

        generated = generate_recommendation(prompt)

        recommendation = (
            RetentionRecommendation.model_validate(generated)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO recommendations (
                customer_id,
                snapshot_date,
                generated_at,
                recommendation_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                customer_id,
                profile["snapshot_date"],
                datetime.now(timezone.utc).isoformat(),
                json.dumps(recommendation.model_dump()),
            ),
        )

        connection.commit()

    return recommendation