from __future__ import annotations

import numpy as np
import pandas as pd


def assign_risk_tiers(
    customer_ids: np.ndarray,
    y_prob: np.ndarray,
    high_risk_rate: float = 0.20,
    medium_risk_rate: float = 0.30,
) -> pd.DataFrame:
    """Assign customers to risk tiers using predicted-risk ranking."""

    customer_ids = np.asarray(customer_ids)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(customer_ids) != len(y_prob):
        raise ValueError(
            "customer_ids and y_prob must have the same length."
        )

    if len(customer_ids) == 0:
        raise ValueError(
            "customer_ids and y_prob must not be empty."
        )

    if not np.isfinite(y_prob).all():
        raise ValueError(
            "y_prob must contain only finite values."
        )

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(
            "y_prob must be between 0 and 1."
        )

    if not 0 < high_risk_rate < 1:
        raise ValueError(
            "high_risk_rate must be between 0 and 1."
        )

    if not 0 < medium_risk_rate < 1:
        raise ValueError(
            "medium_risk_rate must be between 0 and 1."
        )

    if high_risk_rate + medium_risk_rate >= 1:
        raise ValueError(
            "high_risk_rate + medium_risk_rate must be less than 1."
        )

    result = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "churn_probability": y_prob,
        }
    )

    result = result.sort_values(
        ["churn_probability", "customer_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    n_customers = len(result)

    result["risk_rank"] = np.arange(
        1,
        n_customers + 1,
    )

    result["risk_percentile"] = (
        result["risk_rank"] / n_customers
    )

    high_cutoff = int(
        np.ceil(
            n_customers * high_risk_rate
        )
    )

    medium_cutoff = int(
        np.ceil(
            n_customers
            * (
                high_risk_rate
                + medium_risk_rate
            )
        )
    )

    result["risk_tier"] = "Low"

    result.loc[
        result["risk_rank"] <= medium_cutoff,
        "risk_tier",
    ] = "Medium"

    result.loc[
        result["risk_rank"] <= high_cutoff,
        "risk_tier",
    ] = "High"

    return result