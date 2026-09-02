from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.persistence.database import get_connection, initialize_database


RECOMMENDATIONS_DIR = Path("artifacts/recommendations")


def main() -> None:
    initialize_database()

    recommendation_files = sorted(
        RECOMMENDATIONS_DIR.glob("customer_*.json")
    )

    if not recommendation_files:
        print("No recommendation files found.")
        return

    rows = []

    for file_path in recommendation_files:
        with file_path.open("r", encoding="utf-8") as file:
            recommendation = json.load(file)

        customer_id = int(recommendation["customer_id"])

        rows.append(
            (
                customer_id,
                None,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(recommendation),
            )
        )

    with get_connection() as connection:
        for (
            customer_id,
            _,
            generated_at,
            recommendation_json,
        ) in rows:
            risk_row = connection.execute(
                """
                SELECT snapshot_date
                FROM customer_risk
                WHERE customer_id = ?
                """,
                (customer_id,),
            ).fetchone()

            if risk_row is None:
                raise ValueError(
                    f"Customer {customer_id} was not found "
                    "in customer_risk."
                )

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
                    risk_row["snapshot_date"],
                    generated_at,
                    recommendation_json,
                ),
            )

        connection.commit()

    print(
        f"Loaded {len(rows):,} recommendations "
        "into recommendations."
    )


if __name__ == "__main__":
    main()