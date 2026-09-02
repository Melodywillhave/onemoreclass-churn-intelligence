import numpy as np
import pandas as pd


def validate_feature_matrix(
    df: pd.DataFrame,
    snapshot_key=("customer_id", "snapshot_date"),
    target_col="churn",
    forbidden_features=None,
    expected_nullable=None,
) -> None:
    """
    Validate the final customer-snapshot feature matrix.

    Raises AssertionError if any data contract is violated.
    """

    forbidden_features = set(forbidden_features or [])
    expected_nullable = set(expected_nullable or [])

    snapshot_key = list(snapshot_key)

    # 1. Required columns
    required_cols = set(snapshot_key + [target_col])
    missing_required = required_cols - set(df.columns)

    assert not missing_required, (
        f"Missing required columns: {sorted(missing_required)}"
    )

    # 2. Unique customer-snapshot key
    duplicate_rows = df.duplicated(
        subset=snapshot_key
    ).sum()

    assert duplicate_rows == 0, (
        f"Found {duplicate_rows} duplicate customer-snapshot rows."
    )

    # 3. Target validity
    assert df[target_col].notna().all(), (
        f"Target column '{target_col}' contains missing values."
    )

    invalid_target_values = (
        set(df[target_col].unique())
        - {0, 1}
    )

    assert not invalid_target_values, (
        f"Unexpected target values: {invalid_target_values}"
    )

    # 4. Leakage protection
    leakage_cols = sorted(
        forbidden_features.intersection(df.columns)
    )

    assert not leakage_cols, (
        f"Forbidden leakage features found: {leakage_cols}"
    )

    # 5. Infinite numeric values
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    infinite_cols = [
        col
        for col in numeric_cols
        if np.isinf(df[col]).any()
    ]

    assert not infinite_cols, (
        f"Infinite values found in: {infinite_cols}"
    )

    # 6. Missing-value contract
    actual_nullable = set(
        df.columns[
            df.isna().any()
        ]
    )

    unexpected_nullable = sorted(
        actual_nullable
        - expected_nullable
    )

    assert not unexpected_nullable, (
        "Unexpected missing values found in: "
        f"{unexpected_nullable}"
    )

    # 7. Snapshot date validity
    assert pd.api.types.is_datetime64_any_dtype(
        df["snapshot_date"]
    ), "snapshot_date must be datetime dtype."

    # 8. Negative recency checks
    recency_cols = [
        col
        for col in df.columns
        if col.startswith("days_since_")
    ]

    negative_recency = [
        col
        for col in recency_cols
        if (df[col].dropna() < 0).any()
    ]

    assert not negative_recency, (
        f"Negative recency values found in: {negative_recency}"
    )