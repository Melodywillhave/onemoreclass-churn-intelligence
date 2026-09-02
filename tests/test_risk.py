from __future__ import annotations

import pandas as pd

from src.modeling.risk import assign_risk_tiers


def test_assign_risk_tiers() -> None:
    customer_ids = pd.Series(
        [
            100001,
            100002,
            100003,
            100004,
            100005,
            100006,
            100007,
            100008,
            100009,
            100010,
        ]
    )

    churn_probabilities = pd.Series(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
            0.10,
            0.05,
        ]
    )

    result = assign_risk_tiers(
        customer_ids=customer_ids,
        y_prob=churn_probabilities,
        high_risk_rate=0.20,
        medium_risk_rate=0.30,
    )

    assert len(result) == 10

    assert (
        result["risk_tier"]
        .value_counts()
        .to_dict()
        == {
            "Low": 5,
            "Medium": 3,
            "High": 2,
        }
    )

    assert result.iloc[0]["customer_id"] == 100001
    assert result.iloc[0]["risk_rank"] == 1
    assert result.iloc[0]["risk_tier"] == "High"

    assert result.iloc[-1]["customer_id"] == 100010
    assert result.iloc[-1]["risk_rank"] == 10
    assert result.iloc[-1]["risk_tier"] == "Low"