from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.intelligence.customer_profile import build_customer_profile
from src.intelligence.retention_actions import recommend_retention_action
from src.rag.context_builder import build_rag_context
from src.rag.generator import generate_recommendation
from src.rag.prompt_builder import build_generation_prompt
from src.rag.query_builder import build_retrieval_query
from src.rag.semantic_retriever import retrieve_chunks

PREDICTIONS_PATH = Path(
    "artifacts/predictions/churn_predictions.parquet"
)
FEATURES_PATH = Path(
    "data/processed/modeling_features.parquet"
)
OUTPUT_DIR = Path(
    "artifacts/recommendations"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate a grounded retention recommendation."
    )

    parser.add_argument(
        "--customer-id",
        type=int,
        required=True,
        help="Customer ID to generate a recommendation for.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the end-to-end RAG recommendation pipeline."""

    args = parse_args()

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    profile = build_customer_profile(
        customer_id=args.customer_id,
        predictions=predictions,
        features=features,
    )

    recommendation = recommend_retention_action(
        profile
    )

    query = build_retrieval_query(
        profile=profile,
        recommendation=recommendation,
    )

    retrieved_chunks = retrieve_chunks(
        query=query,
        top_k=5,
    )

    context = build_rag_context(
        profile=profile,
        recommendation=recommendation,
        retrieved_chunks=retrieved_chunks,
    )

    prompt = build_generation_prompt(
        rag_context=context,
    )

    generated_recommendation = generate_recommendation(
        prompt=prompt,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"customer_{args.customer_id}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            generated_recommendation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            generated_recommendation,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved recommendation to: {output_path}"
    )


if __name__ == "__main__":
    main()