from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.intelligence.customer_profile import build_customer_profile
from src.intelligence.retention_actions import recommend_retention_action
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

OUTPUT_PATH = Path(
    "artifacts/evaluation/rag_evaluation_sample.json"
)

SUMMARY_OUTPUT_PATH = Path(
    "artifacts/evaluation/rag_evaluation_qwen25_7b.csv"
)

SAMPLES_PER_DRIVER = 1

TARGET_DRIVERS = [
    "Engagement Decline",
    "Low Recent Engagement",
    "Service Friction",
    "No Strong Risk Signal",
]

ALLOWED_KNOWLEDGE_SOURCES = {
    "retention_playbook.md",
    "engagement_playbook.md",
    "service_recovery_playbook.md",
    "offer_policy.md",
}
RESTRICTED_TACTIC_PATTERNS = [
    "offer a discount",
    "provide a discount",
    "give a discount",
    "offer credit",
    "provide credit",
    "give credit",
    "offer an incentive",
    "provide an incentive",
    "give an incentive",
    "offer a reward",
    "provide a reward",
    "give a reward",
    "send a promotion",
    "provide a promotion",
    "offer a promotion",
    "provide compensation",
    "offer compensation",
]


def select_evaluation_customers(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Select representative high-risk customers by primary driver."""

    high_risk = predictions.loc[
        predictions["risk_tier"] == "High"
    ].copy()

    samples = []

    for driver in TARGET_DRIVERS:
        driver_sample = (
            high_risk.loc[
                high_risk["primary_driver"] == driver
            ]
            .sort_values("risk_rank")
            .head(SAMPLES_PER_DRIVER)
        )

        samples.append(driver_sample)

    return pd.concat(
        samples,
        ignore_index=True,
    )


def evaluate_generated_recommendation(
    customer_id: int,
    expected_action: str,
    expected_evidence: list[str],
    generated: dict,
    retrieved_chunks: pd.DataFrame,
) -> dict:
    """Run deterministic quality checks on one LLM recommendation."""

    validated = RetentionRecommendation.model_validate(
        generated
    )

    generated_evidence = set(
        validated.supporting_evidence
    )

    expected_evidence_set = set(
        expected_evidence
    )

    evidence_grounded = generated_evidence.issubset(
        expected_evidence_set
    )

    if expected_evidence_set:
        evidence_coverage = (
            len(
                generated_evidence
                & expected_evidence_set
            )
            / len(expected_evidence_set)
        )
    else:
         evidence_coverage = (
            1.0 if not generated_evidence else 0.0
    )

    customer_id_match = (
        validated.customer_id == customer_id
    )

    action_match = (
        validated.recommended_action.action
        == expected_action
    )

    policy_sources_valid = all(
        rationale.source in ALLOWED_KNOWLEDGE_SOURCES
        for rationale in validated.policy_rationale
    )
    retrieved_pairs = set(
        zip(
            retrieved_chunks["source"],
            retrieved_chunks["section_path"],
        )
    )

    generated_pairs = {
        (
            rationale.source,
            rationale.section,
        )
        for rationale in validated.policy_rationale
    }

    citations_grounded = generated_pairs.issubset(
        retrieved_pairs
    )

    action_text = " ".join(
        [
            validated.recommended_action.details,
            validated.secondary_consideration,
        ]
    ).lower()


    matched_restricted_terms = [
        pattern
        for pattern in RESTRICTED_TACTIC_PATTERNS
        if pattern in action_text
    ]

    return {
    "schema_valid": True,
    "customer_id_match": customer_id_match,
    "action_match": action_match,
    "policy_sources_valid": policy_sources_valid,
    "citations_grounded": citations_grounded,
    "no_restricted_tactic": (
        len(matched_restricted_terms) == 0
    ),
    "matched_restricted_terms": matched_restricted_terms,
    "action_text": action_text,
    "evidence_grounded": evidence_grounded,
    "evidence_coverage": evidence_coverage,
    }


def main() -> None:
    """Generate RAG recommendations for evaluation customers."""

    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    features = pd.read_parquet(
        FEATURES_PATH
    )

    evaluation_customers = select_evaluation_customers(
        predictions
    )

    print("Evaluation sample:")
    print(
        evaluation_customers[
            [
                "customer_id",
                "risk_rank",
                "risk_tier",
                "primary_driver",
            ]
        ].to_string(index=False)
    )

    evaluation_results = []

    for row in evaluation_customers.itertuples():
        customer_id = int(row.customer_id)

        print(
            f"\nGenerating recommendation for "
            f"customer {customer_id}..."
        )

        profile = build_customer_profile(
            customer_id=customer_id,
            predictions=predictions,
            features=features,
        )

        deterministic_recommendation = (
            recommend_retention_action(profile)
        )

        query = build_retrieval_query(
            profile=profile,
            recommendation=deterministic_recommendation,
        )

        retrieved_chunks = retrieve_chunks(
            query=query,
            top_k=5,
        )

        context = build_rag_context(
            profile=profile,
            recommendation=deterministic_recommendation,
            retrieved_chunks=retrieved_chunks,
        )

        prompt = build_generation_prompt(
            rag_context=context,
        )

        generated = generate_recommendation(
            prompt=prompt,
        )

        checks = evaluate_generated_recommendation(
            customer_id=customer_id,
            expected_action=(
                deterministic_recommendation[
                    "recommended_action"
                ]
            ),
            expected_evidence=(
                deterministic_recommendation[
                    "supporting_signals"
                ]
            ),
            generated=generated,
            retrieved_chunks=retrieved_chunks,
        )

        evaluation_results.append(
        {
            "customer_id": customer_id,
            "risk_rank": int(row.risk_rank),
            "primary_driver": row.primary_driver,
            "expected_action": (
                deterministic_recommendation[
                    "recommended_action"
                ]
            ),
            "expected_evidence": (
                deterministic_recommendation[
                    "supporting_signals"
                ]
            ),
            "checks": checks,
            "generated_recommendation": generated,
            }
        )

    summary_rows = []

    for result in evaluation_results:
        checks = result["checks"]

        summary_rows.append(
            {
                "customer_id": result["customer_id"],
                "primary_driver": result["primary_driver"],
                "schema_valid": checks["schema_valid"],
                "customer_id_match": checks["customer_id_match"],
                "action_match": checks["action_match"],
                "policy_sources_valid": checks[
                    "policy_sources_valid"
                ],
                "citations_grounded": checks[
                    "citations_grounded"
                ],
                "no_restricted_tactic": checks[
                    "no_restricted_tactic"
                ],
                "evidence_grounded": checks[
                    "evidence_grounded"
                ],
                "evidence_coverage": checks[
                    "evidence_coverage"
                ],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    print("\nEvaluation summary:")
    print(
        summary_df.to_string(index=False)
    )

    print("\nPass rates:")
    print(
        summary_df[
            [
                "schema_valid",
                "customer_id_match",
                "action_match",
                "policy_sources_valid",
                "citations_grounded",
                "no_restricted_tactic",
                "evidence_grounded",
            ]
        ]
        .mean()
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    print(
        "\nAverage evidence coverage:",
        f"{summary_df['evidence_coverage'].mean():.1%}",
    )

    print("\nRestricted tactic review:")

    print("\nEvidence grounding review:")

    for result in evaluation_results:
        checks = result["checks"]

        if not checks["evidence_grounded"]:
            print(
                f"\nCustomer {result['customer_id']} "
                f"({result['primary_driver']})"
            )

            print(
                "Expected evidence:",
                result["expected_evidence"],
            )

            print(
                "Generated evidence:",
                result["generated_recommendation"][
                    "supporting_evidence"
                ],
            )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved evaluation summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )
    
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved evaluation results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()