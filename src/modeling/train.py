from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import build_preprocessor, split_features_target


def build_logistic_regression_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build an unfitted logistic-regression pipeline using the project preprocessor."""
    if X_train is None:
        raise ValueError("X_train must not be None.")
    if X_train.empty:
        raise ValueError("X_train must not be empty.")

    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def fit_logistic_regression(train_df: pd.DataFrame) -> Pipeline:
    """Fit a logistic regression model on the training snapshot frame only."""
    if train_df is None:
        raise ValueError("train_df must not be None.")
    if train_df.empty:
        raise ValueError("train_df must not be empty.")

    X_train, y_train = split_features_target(train_df)
    pipeline = build_logistic_regression_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    return pipeline

def build_hist_gradient_boosting_pipeline(
    X_train: pd.DataFrame,
) -> Pipeline:
    """Build an unfitted histogram gradient boosting pipeline."""
    if X_train is None:
        raise ValueError("X_train must not be None.")
    if X_train.empty:
        raise ValueError("X_train must not be empty.")

    preprocessor = build_preprocessor(X_train)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def fit_hist_gradient_boosting(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Fit histogram gradient boosting on the training frame only."""
    if train_df is None:
        raise ValueError("train_df must not be None.")
    if train_df.empty:
        raise ValueError("train_df must not be empty.")

    X_train, y_train = split_features_target(train_df)

    pipeline = build_hist_gradient_boosting_pipeline(
        X_train
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline

def predict_probabilities(model: Any, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict churn probabilities for a dataframe and return true labels plus probabilities."""
    if model is None:
        raise ValueError("model must not be None.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("model must support predict_proba.")

    if df is None:
        raise ValueError("df must not be None.")
    if df.empty:
        raise ValueError("df must not be empty.")

    X_eval, y_true = split_features_target(df)
    y_prob = model.predict_proba(X_eval)[:, 1]

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return y_true.to_numpy(), y_prob


def baseline_probability(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the target labels and a constant churn-rate baseline for evaluation data."""
    if train_df is None:
        raise ValueError("train_df must not be None.")
    if train_df.empty:
        raise ValueError("train_df must not be empty.")
    if eval_df is None:
        raise ValueError("eval_df must not be None.")
    if eval_df.empty:
        raise ValueError("eval_df must not be empty.")

    _, y_train = split_features_target(train_df)
    baseline_rate = float(y_train.mean())

    _, y_true = split_features_target(eval_df)
    baseline_predictions = np.full(len(y_true), baseline_rate, dtype=float)

    if np.any((baseline_predictions < 0) | (baseline_predictions > 1)):
        raise ValueError("Baseline probabilities must be between 0 and 1.")

    return y_true.to_numpy(), baseline_predictions
