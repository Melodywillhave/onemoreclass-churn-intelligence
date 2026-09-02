from pathlib import Path

import pandas as pd

import src.modeling.threshold
import src.modeling.train
import src.modeling.walk_forward

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_features.parquet"
    )

    df = pd.read_parquet(data_path)

    folds = src.modeling.walk_forward.generate_purged_walk_forward_splits(
        df,
        prediction_horizon_days=30,
        min_train_snapshots=1,
    )

    all_results = []

    for fold_number, fold in enumerate(folds, start=1):
        train_df = fold["train_df"]
        val_df = fold["val_df"]

        model = src.modeling.train.fit_hist_gradient_boosting(
            train_df
        )

        y_true, y_prob = src.modeling.train.predict_probabilities(
            model,
            val_df,
        )

        fold_results = src.modeling.threshold.evaluate_top_k(
            y_true,
            y_prob,
            top_k_rates=[
                0.10,
                0.20,
                0.30,
            ],
        )

        fold_results.insert(
            0,
            "fold",
            fold_number,
        )

        fold_results.insert(
            1,
            "val_snapshot",
            fold["val_snapshot"],
        )

        fold_results.insert(
            2,
            "val_churn_rate",
            float(val_df["churn"].mean()),
        )

        all_results.append(
            fold_results
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\nTop-K Walk-Forward Evaluation\n")
    print(
        results.to_string(
            index=False
        )
    )

    output_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "walk_forward_top_k_hgb.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved Top-K results to: {output_path}"
    )


if __name__ == "__main__":
    main()