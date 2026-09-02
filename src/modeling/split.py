from __future__ import annotations

import pandas as pd


def _validate_split(
    split_df: pd.DataFrame,
    split_name: str,
    date_col: str,
) -> None:
    """Fail fast when a split is empty or contains invalid duplicate snapshot rows."""
    if split_df.empty:
        raise ValueError(f"{split_name} split must not be empty.")

    duplicate_pairs = split_df.duplicated(subset=["customer_id", date_col], keep=False)
    if duplicate_pairs.any():
        raise ValueError(
            f"{split_name} split contains duplicate (customer_id, {date_col}) rows."
        )


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp | str,
    val_end: pd.Timestamp | str,
    test_end: pd.Timestamp | str | None = None,
    date_col: str = "snapshot_date",
    prediction_horizon_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a customer-snapshot dataset into chronological train/validation/test sets.

    The split enforces a purge/embargo gap after each training and validation cut-off:
    observations inside the gap are excluded from the following split to avoid leakage from
    overlapping future windows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with one row per (customer_id, snapshot_date).
    train_end : pandas.Timestamp | str
        Final snapshot date included in the training split.
    val_end : pandas.Timestamp | str
        Final snapshot date included in the validation split.
    test_end : pandas.Timestamp | str, optional
        Final snapshot date included in the test split, by default None.
    date_col : str, default "snapshot_date"
        Column containing the snapshot timestamp.
    prediction_horizon_days : int, default 30
        Length of the purge period after each split boundary.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Chronologically ordered train, validation, and test dataframes.
    """
    if df is None:
        raise ValueError("Input dataframe must not be None.")
    if df.empty:
        raise ValueError("Input dataframe must not be empty.")

    required_columns = [date_col, "customer_id", "churn"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}.")

    if prediction_horizon_days < 0:
        raise ValueError("prediction_horizon_days must be non-negative.")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be earlier than val_end.")

    if test_end is not None:
        test_end_ts = pd.Timestamp(test_end)
        if val_end_ts >= test_end_ts:
            raise ValueError("val_end must be earlier than test_end when test_end is provided.")
    else:
        test_end_ts = None

    df_working = df.copy()
    df_working[date_col] = pd.to_datetime(df_working[date_col], errors="raise")

    # Enforce the embargo gap after the train and validation cut-offs.
    train_purge_start = train_end_ts + pd.Timedelta(days=prediction_horizon_days)
    val_purge_start = val_end_ts + pd.Timedelta(days=prediction_horizon_days)

    train_df = df_working.loc[df_working[date_col] <= train_end_ts].copy()
    val_df = df_working.loc[
        (df_working[date_col] > train_purge_start)
        & (df_working[date_col] <= val_end_ts)
    ].copy()

    if test_end_ts is None:
        test_df = df_working.loc[df_working[date_col] > val_purge_start].copy()
    else:
        test_df = df_working.loc[
            (df_working[date_col] > val_purge_start)
            & (df_working[date_col] <= test_end_ts)
        ].copy()

    # Fail fast on empty splits and duplicate snapshot pairs.
    _validate_split(train_df, "train", date_col)
    _validate_split(val_df, "validation", date_col)
    _validate_split(test_df, "test", date_col)

    train_df = train_df.sort_values(by=[date_col, "customer_id"]).reset_index(drop=True)
    val_df = val_df.sort_values(by=[date_col, "customer_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=[date_col, "customer_id"]).reset_index(drop=True)

    # Explicit temporal separation checks.
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train, validation, and test splits must all be non-empty.")

    max_train_date = train_df[date_col].max()
    min_val_date = val_df[date_col].min()
    if max_train_date + pd.Timedelta(days=prediction_horizon_days) >= min_val_date:
        raise ValueError(
            "Temporal leakage: the maximum training snapshot date plus the prediction horizon "
            "must be strictly earlier than the minimum validation snapshot date."
        )

    max_val_date = val_df[date_col].max()
    min_test_date = test_df[date_col].min()
    if max_val_date + pd.Timedelta(days=prediction_horizon_days) >= min_test_date:
        raise ValueError(
            "Temporal leakage: the maximum validation snapshot date plus the prediction horizon "
            "must be strictly earlier than the minimum test snapshot date."
        )

    return train_df, val_df, test_df
