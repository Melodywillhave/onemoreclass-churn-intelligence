from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.evaluate import evaluate_walk_forward
from src.modeling.walk_forward import generate_purged_walk_forward_splits


def main() -> None:
    """Load the modeling feature table, evaluate purged walk-forward folds, and save results."""
    data_path = PROJECT_ROOT / "data" / "processed" / "modeling_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")

    df = pd.read_parquet(data_path)
    folds = generate_purged_walk_forward_splits(
        df=df,
        date_col="snapshot_date",
        prediction_horizon_days=30,
        min_train_snapshots=1,
    )

    metrics_df = evaluate_walk_forward(folds)
    print(metrics_df)

    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "walk_forward_metrics_logistic.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
