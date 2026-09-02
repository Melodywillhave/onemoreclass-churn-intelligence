from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import split_features_target


def calculate_permutation_importance(
    model: Pipeline,
    evaluation_df: pd.DataFrame,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate global permutation importance on an evaluation dataset."""

    if model is None:
        raise ValueError("model must not be None.")

    if evaluation_df is None or evaluation_df.empty:
        raise ValueError("evaluation_df must not be empty.")

    if n_repeats <= 0:
        raise ValueError("n_repeats must be greater than 0.")

    X_eval, y_eval = split_features_target(
        evaluation_df
    )

    importance = permutation_importance(
        estimator=model,
        X=X_eval,
        y=y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    results = pd.DataFrame(
        {
            "feature": X_eval.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    results = results.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    results["importance_rank"] = (
        results.index + 1
    )

    return results[
        [
            "importance_rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ]