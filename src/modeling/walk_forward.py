from __future__ import annotations

from typing import Any

import pandas as pd


def generate_purged_walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "snapshot_date",
    prediction_horizon_days: int = 30,
    min_train_snapshots: int = 1,
) -> list[dict[str, Any]]:
    """Generate purged, expanding-window walk-forward validation folds.

    Each fold uses exactly one validation snapshot date and expands the training window
    to include all earlier snapshot dates whose complete future outcome window ends
    strictly before that validation snapshot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input customer-snapshot table with one row per (customer_id, snapshot_date).
    date_col : str, default "snapshot_date"
        Column containing the snapshot date.
    prediction_horizon_days : int, default 30
        Number of days used to define the future outcome window.
    min_train_snapshots : int, default 1
        Minimum number of eligible training snapshots required for a valid fold.

    Returns
    -------
    list[dict[str, Any]]
        A list of fold dictionaries. Each item contains:
        - ``fold``: integer fold number
        - ``train_df``: expanding training snapshot dataframe
        - ``val_df``: validation snapshot dataframe for one snapshot date
        - ``train_snapshots``: list of training snapshot dates
        - ``val_snapshot``: validation snapshot date
    """
    if df is None:
        raise ValueError("df must not be None.")
    if df.empty:
        raise ValueError("df must not be empty.")

    required_columns = [date_col, "customer_id", "churn"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}.")

    if prediction_horizon_days <= 0:
        raise ValueError("prediction_horizon_days must be positive.")
    if min_train_snapshots < 1:
        raise ValueError("min_train_snapshots must be greater than or equal to 1.")

    df_working = df.copy()
    df_working[date_col] = pd.to_datetime(df_working[date_col], errors="raise")

    duplicate_pairs = df_working.duplicated(subset=["customer_id", date_col], keep=False)
    if duplicate_pairs.any():
        raise ValueError(
            "Input dataframe contains duplicate (customer_id, snapshot_date) rows."
        )

    snapshot_dates = sorted(df_working[date_col].drop_duplicates().tolist())
    if not snapshot_dates:
        raise ValueError("No snapshot dates were found in the input dataframe.")

    folds: list[dict[str, Any]] = []
    for val_snapshot in snapshot_dates:
        eligible_train_snapshots = [
            train_snapshot
            for train_snapshot in snapshot_dates
            if train_snapshot + pd.Timedelta(days=prediction_horizon_days) < val_snapshot
        ]

        if len(eligible_train_snapshots) < min_train_snapshots:
            continue

        train_df = df_working[df_working[date_col].isin(eligible_train_snapshots)].copy()
        val_df = df_working[df_working[date_col] == val_snapshot].copy()

        if train_df.empty:
            raise ValueError("Eligible training dataframe must not be empty for a valid fold.")
        if val_df.empty:
            raise ValueError("Validation dataframe must not be empty for a valid fold.")

        max_train_snapshot = max(eligible_train_snapshots)
        if max_train_snapshot + pd.Timedelta(days=prediction_horizon_days) >= val_snapshot:
            raise ValueError(
                "Temporal leakage detected: max(train snapshot) + prediction_horizon_days "
                "must be strictly earlier than validation_snapshot."
            )

        train_df = train_df.sort_values(by=[date_col, "customer_id"]).reset_index(drop=True)
        val_df = val_df.sort_values(by=[date_col, "customer_id"]).reset_index(drop=True)

        folds.append(
            {
                "fold": len(folds) + 1,
                "train_df": train_df,
                "val_df": val_df,
                "train_snapshots": eligible_train_snapshots,
                "val_snapshot": val_snapshot,
            }
        )

    if not folds:
        raise ValueError("No valid walk-forward folds were generated.")

    return folds
