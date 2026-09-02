from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.persistence.database import get_connection, initialize_database


DECISIONS_PATH = Path(
    "artifacts/decisions/retention_decisions.parquet"
)


def main() -> None:
    initialize_database()

    decisions = pd.read_parquet(DECISIONS_PATH)

    retention_decisions = decisions[
        [
            "customer_id",
            "snapshot_date",
            "risk_tier",
            "primary_driver",
            "secondary_driver",
            "recommended_action",
            "action_priority",
            "objective",
            "supporting_signals",
        ]
    ].copy()

    retention_decisions["snapshot_date"] = (
        pd.to_datetime(retention_decisions["snapshot_date"])
        .dt.strftime("%Y-%m-%d")
    )

    retention_decisions["supporting_signals"] = (
        retention_decisions["supporting_signals"]
        .apply(
            lambda value: json.dumps(
                value.tolist()
                if hasattr(value, "tolist")
                else value
            )
        )
    )

    with get_connection() as connection:
        connection.execute("DELETE FROM retention_decisions")

        retention_decisions.to_sql(
            "retention_decisions",
            connection,
            if_exists="append",
            index=False,
        )

        connection.commit()

    print(
        f"Loaded {len(retention_decisions):,} rows "
        "into retention_decisions."
    )


if __name__ == "__main__":
    main()