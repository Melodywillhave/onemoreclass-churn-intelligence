from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.labeling.churn import build_churn_labels


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

CUSTOMERS_PATH = RAW_DATA_DIR / "customers.csv"
ACTIVITIES_PATH = RAW_DATA_DIR / "activities.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "modeling_labels.csv"


def main() -> None:
    """Build and persist churn labels."""
    customers = pd.read_csv(
        CUSTOMERS_PATH,
        parse_dates=["signup_date"],
    )

    activities = pd.read_csv(
        ACTIVITIES_PATH,
        parse_dates=["activity_date"],
    )

    labels = build_churn_labels(
        customers=customers,
        activities=activities,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(labels):,} labels to "
        f"{OUTPUT_PATH}"
    )
    print(
        f"Snapshots: {labels['snapshot_date'].nunique()}"
    )
    print(
        f"Churn rate: {labels['churn'].mean():.2%}"
    )


if __name__ == "__main__":
    main()