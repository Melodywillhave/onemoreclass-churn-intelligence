from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate classification performance across probability thresholds."""

    if thresholds is None:
        thresholds = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
        ]

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    if len(y_true) == 0:
        raise ValueError("y_true and y_prob must not be empty.")

    rows = []

    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("Thresholds must be between 0 and 1.")

        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        predicted_positive_rate = float(y_pred.mean())

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate": predicted_positive_rate,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(rows)

def evaluate_top_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k_rates: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate retention targeting performance at different capacity levels."""

    if top_k_rates is None:
        top_k_rates = [
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
        ]

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")

    if len(y_true) == 0:
        raise ValueError("y_true and y_prob must not be empty.")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob must contain only finite values.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must be between 0 and 1.")

    total_customers = len(y_true)
    total_churners = int(y_true.sum())

    # Highest predicted churn risk first
    ranking = np.argsort(-y_prob)

    rows = []

    for rate in top_k_rates:
        if not 0 < rate <= 1:
            raise ValueError("Top-k rates must be between 0 and 1.")

        n_targeted = max(
            1,
            int(np.ceil(total_customers * rate)),
        )

        targeted_indices = ranking[:n_targeted]
        targeted_y_true = y_true[targeted_indices]

        captured_churners = int(targeted_y_true.sum())

        precision_at_k = captured_churners / n_targeted

        recall_at_k = (
            captured_churners / total_churners
            if total_churners > 0
            else 0.0
        )

        lift_at_k = (
            precision_at_k / (total_churners / total_customers)
            if total_churners > 0
            else 0.0
        )

        rows.append(
            {
                "target_rate": rate,
                "customers_targeted": n_targeted,
                "captured_churners": captured_churners,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_at_k": lift_at_k,
            }
        )

    return pd.DataFrame(rows)