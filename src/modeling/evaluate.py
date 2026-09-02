from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.modeling.train import (
    baseline_probability,
    fit_hist_gradient_boosting,
    fit_logistic_regression,
    predict_probabilities,
)


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate probability predictions against binary churn labels."""
    if y_true is None or y_prob is None:
        raise ValueError("y_true and y_prob must not be None.")
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must not be empty.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob, dtype=float)

    if np.isnan(y_true_array).any() or np.isnan(y_prob_array).any():
        raise ValueError("y_true and y_prob must not contain missing values.")

    if not np.isin(y_true_array, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 values.")

    if np.any((y_prob_array < 0) | (y_prob_array > 1)):
        raise ValueError("y_prob must be between 0 and 1.")

    unique_classes = np.unique(y_true_array)
    if len(unique_classes) < 2:
        raise ValueError("ROC-AUC requires both classes to be present in y_true.")

    roc_auc = roc_auc_score(y_true_array, y_prob_array)
    pr_auc = average_precision_score(y_true_array, y_prob_array)
    loss = log_loss(y_true_array, y_prob_array, labels=[0, 1])
    brier = brier_score_loss(y_true_array, y_prob_array)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(loss),
        "brier_score": float(brier),
    }


def evaluate_walk_forward(folds: list[dict[str, Any]]) -> pd.DataFrame:
    """Evaluate baseline and logistic-regression models across ordered walk-forward folds."""
    if folds is None:
        raise ValueError("folds must not be None.")
    if len(folds) == 0:
        raise ValueError("At least one fold must be provided.")

    rows: list[dict[str, Any]] = []

    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("Each fold must be a dictionary.")  # noqa: TRY004

        required_keys = {"train_df", "val_df", "train_snapshots", "val_snapshot"}
        missing_keys = required_keys - set(fold.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Fold is missing required keys: {missing}.")

        train_df = fold["train_df"]
        val_df = fold["val_df"]
        train_snapshots = fold["train_snapshots"]
        val_snapshot = fold["val_snapshot"]

        if train_df is None or val_df is None:
            raise ValueError("Each fold must contain non-empty train_df and val_df.")
        if train_df.empty or val_df.empty:
            raise ValueError("Each fold must contain non-empty train_df and val_df.")

        # Baseline
        y_true_baseline, y_prob_baseline = baseline_probability(train_df, val_df)
        baseline_metrics = evaluate_probabilities(
            y_true_baseline,
            y_prob_baseline,
        )

        # Logistic Regression
        logit_model = fit_logistic_regression(train_df)
        y_true_logit, y_prob_logit = predict_probabilities(
            logit_model,
            val_df,
        )
        logistic_metrics = evaluate_probabilities(
            y_true_logit,
            y_prob_logit,
        )

        # HistGradientBoosting
        hgb_model = fit_hist_gradient_boosting(train_df)
        y_true_hgb, y_prob_hgb = predict_probabilities(
            hgb_model,
            val_df,
        )
        hgb_metrics = evaluate_probabilities(
            y_true_hgb,
            y_prob_hgb,
        )

        train_churn_rate = float(train_df["churn"].mean())
        val_churn_rate = float(val_df["churn"].mean())

        for model_name, y_true, y_prob, metrics in [
            (
                "baseline",
                y_true_baseline,
                y_prob_baseline,
                baseline_metrics,
            ),
            (
                "logistic_regression",
                y_true_logit,
                y_prob_logit,
                logistic_metrics,
            ),
            (
                "hist_gradient_boosting",
                y_true_hgb,
                y_prob_hgb,
                hgb_metrics,
            ),
        ]:
            rows.append(
                {
                    "fold": fold.get("fold", len(rows) + 1),
                    "model": model_name,
                    "train_snapshots": list(train_snapshots),
                    "val_snapshot": val_snapshot,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "train_churn_rate": train_churn_rate,
                    "val_churn_rate": val_churn_rate,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "log_loss": metrics["log_loss"],
                    "brier_score": metrics["brier_score"],
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No evaluation rows were generated.")

    results = results.sort_values(by=["fold", "model"]).reset_index(drop=True)
    return results
