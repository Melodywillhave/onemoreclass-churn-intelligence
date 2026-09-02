from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMNS = ["customer_id", "snapshot_date"]
TARGET_COLUMN = "churn"


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_columns: list[str] = ID_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictors and target without mutating the source dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input snapshot-level dataframe.
    target_col : str, default "churn"
        Binary target column.
    id_columns : list[str], default ["customer_id", "snapshot_date"]
        Identifier columns excluded from the feature matrix.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Predictor dataframe and the target series.
    """
    if df is None:
        raise ValueError("df must not be None.")
    if df.empty:
        raise ValueError("df must not be empty.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in the input dataframe.")

    missing_id_columns = [column for column in id_columns if column not in df.columns]
    if missing_id_columns:
        missing = ", ".join(missing_id_columns)
        raise ValueError(f"Input dataframe is missing required ID columns: {missing}.")

    y = df[target_col].copy()
    if y.isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    valid_values = {0, 1}
    if not y.isin(valid_values).all():
        raise ValueError(f"Target column '{target_col}' must contain only 0/1 values.")

    feature_columns = [column for column in df.columns if column not in set(id_columns) | {target_col}]
    if not feature_columns:
        raise ValueError("At least one predictor column must remain after excluding IDs and target.")

    X = df.loc[:, feature_columns].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing definition for numeric and categorical feature columns.

    This function only defines the sklearn preprocessing pipeline and does not fit it.
    The downstream training code is responsible for fitting with training data.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor matrix without ID columns or target.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessor that combines numeric and categorical transformers.
    """
    if X is None:
        raise ValueError("X must not be None.")
    if X.empty:
        raise ValueError("X must not be empty.")

    forbidden_columns = {"customer_id", "snapshot_date", "churn"}
    if forbidden_columns.intersection(X.columns):
        raise ValueError("X must not contain customer_id, snapshot_date, or churn columns.")

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature column.")

    numeric_features = []
    categorical_features = []

    for column in X.columns:
        series = X[column]
        is_boolean = pd.api.types.is_bool_dtype(series)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_boolean or is_numeric:
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    if not numeric_features and not categorical_features:
        raise ValueError("At least one numeric or categorical feature must exist in X.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor
