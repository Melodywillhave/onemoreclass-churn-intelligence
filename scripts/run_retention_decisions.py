from pathlib import Path

import pandas as pd

from src.intelligence.customer_profile import build_customer_profile
from src.intelligence.retention_actions import recommend_retention_action

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate retention decisions for high-risk customers."""

    predictions_path = (
        PROJECT_ROOT
        / "artifacts"
        / "predictions"
        / "churn_predictions.parquet"
    )

    features_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_features.parquet"
    )

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "decisions"
    )

    output_path = (
        output_dir
        / "retention_decisions.parquet"
    )

    predictions = pd.read_parquet(
        predictions_path
    )

    features = pd.read_parquet(
        features_path
    )

    high_risk_customers = predictions.loc[
        predictions["risk_tier"] == "High",
        "customer_id",
    ].tolist()

    if not high_risk_customers:
        raise ValueError(
            "No high-risk customers were found."
        )

    rows: list[dict] = []

    for customer_id in high_risk_customers:
        profile = build_customer_profile(
            customer_id=customer_id,
            predictions=predictions,
            features=features,
        )

        recommendation = recommend_retention_action(
            profile
        )

        rows.append(
            {
                "customer_id": customer_id,
                "snapshot_date": profile[
                    "snapshot_date"
                ],
                "churn_probability": profile[
                    "risk"
                ][
                    "churn_probability"
                ],
                "risk_rank": profile[
                    "risk"
                ][
                    "risk_rank"
                ],
                "risk_tier": recommendation[
                    "risk_tier"
                ],
                "primary_driver": recommendation[
                    "primary_driver"
                ],
                "secondary_driver": recommendation[
                    "secondary_driver"
                ],
                "recommended_action": recommendation[
                    "recommended_action"
                ],
                "action_priority": recommendation[
                    "action_priority"
                ],
                "objective": recommendation[
                    "objective"
                ],
                "supporting_signals": recommendation[
                    "supporting_signals"
                ],
            }
        )

    decisions = pd.DataFrame(
        rows
    )

    decisions = decisions.sort_values(
        "risk_rank"
    ).reset_index(drop=True)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"High-risk customers: {len(decisions):,}"
    )

    print("\nRecommended action distribution:")
    print(
        decisions[
            "recommended_action"
        ].value_counts()
    )

    print("\nPrimary driver distribution:")
    print(
        decisions[
            "primary_driver"
        ].value_counts()
    )

    print(
        f"\nSaved retention decisions to: {output_path}"
    )


if __name__ == "__main__":
    main()