from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.persistence.database import get_connection, initialize_database


PREDICTIONS_PATH = Path(
    "artifacts/predictions/churn_predictions.parquet"
)


def main() -> None:
    initialize_database()

    predictions = pd.read_parquet(PREDICTIONS_PATH)

    customer_risk = predictions[
        [
            "customer_id",
            "snapshot_date",
            "churn_probability",
            "risk_rank",
            "risk_percentile",
            "risk_tier",
            "primary_driver",
            "secondary_driver",
        ]
    ].copy()

    customer_risk["snapshot_date"] = (
        pd.to_datetime(customer_risk["snapshot_date"])
        .dt.strftime("%Y-%m-%d")
    )

    with get_connection() as connection:
        connection.execute("DELETE FROM customer_risk")

        customer_risk.to_sql(
            "customer_risk",
            connection,
            if_exists="append",
            index=False,
        )

        connection.commit()

    print(
        f"Loaded {len(customer_risk):,} rows "
        "into customer_risk."
    )


if __name__ == "__main__":
    main()