import numpy as np
import pandas as pd


def _build_transaction_history(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for snapshot_date in sorted(labels["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)

        customers_at_snapshot = labels.loc[
            labels["snapshot_date"] == snapshot_date,
            ["customer_id"],
        ]

        history = transactions.loc[
            transactions["transaction_date"] <= snapshot_date
        ].copy()

        history["paid_payment"] = (
            history["payment_status"].eq("Paid")
        ).astype(int)

        history["failed_payment"] = (
            history["payment_status"].eq("Failed")
        ).astype(int)

        history["pending_payment"] = (
            history["payment_status"].eq("Pending")
        ).astype(int)

        history["live_purchase"] = (
            history["delivery_mode"].eq("Live")
        ).astype(int)

        history["discount_purchase"] = (
            history["discount_used"].astype(int)
        )

        aggregated = (
            history
            .groupby("customer_id")
            .agg(
                transaction_count=("transaction_id", "count"),
                paid_transactions=("paid_payment", "sum"),
                failed_transactions=("failed_payment", "sum"),
                pending_transactions=("pending_payment", "sum"),
                avg_discount_rate=("discount_rate", "mean"),
                discount_purchase_share=("discount_purchase", "mean"),
                live_purchase_share=("live_purchase", "mean"),
                last_transaction_date=("transaction_date", "max"),
            )
            .reset_index()
        )

        features = customers_at_snapshot.merge(
            aggregated,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        features["snapshot_date"] = snapshot_date

        features["days_since_last_purchase"] = (
            snapshot_date
            - features["last_transaction_date"]
        ).dt.days

        rows.append(features)

    return pd.concat(rows, ignore_index=True)


def _build_currency_spend(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for snapshot_date in sorted(labels["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)

        customers_at_snapshot = labels.loc[
            labels["snapshot_date"] == snapshot_date,
            ["customer_id"],
        ]

        history = transactions.loc[
            (transactions["transaction_date"] <= snapshot_date)
            & (transactions["payment_status"].eq("Paid"))
        ].copy()

        spend = (
            history
            .pivot_table(
                index="customer_id",
                columns="currency",
                values="final_amount",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        spend.columns.name = None

        spend = spend.rename(
            columns={
                "RMB": "paid_spend_rmb",
                "GBP": "paid_spend_gbp",
            }
        )

        features = customers_at_snapshot.merge(
            spend,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        for col in [
            "paid_spend_rmb",
            "paid_spend_gbp",
        ]:
            if col not in features.columns:
                features[col] = 0

            features[col] = features[col].fillna(0)

        features["snapshot_date"] = snapshot_date

        rows.append(
            features[
                [
                    "customer_id",
                    "snapshot_date",
                    "paid_spend_rmb",
                    "paid_spend_gbp",
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_transaction_features(
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe transaction features at the customer-snapshot level.
    """

    labels = labels.copy()
    transactions = transactions.copy()

    labels["snapshot_date"] = pd.to_datetime(
        labels["snapshot_date"]
    )

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"]
    )

    features = _build_transaction_history(
        labels=labels,
        transactions=transactions,
    )

    spend_features = _build_currency_spend(
        labels=labels,
        transactions=transactions,
    )

    features = features.merge(
        spend_features,
        on=["customer_id", "snapshot_date"],
        how="left",
        validate="one_to_one",
    )

    features["has_purchase_history"] = (
        features["transaction_count"]
        .notna()
        .astype(int)
    )

    # Cumulative spend is well-defined as zero for customers
    # with no paid purchase history.
    spend_cols = [
        "paid_spend_rmb",
        "paid_spend_gbp",
    ]

    features[spend_cols] = (
        features[spend_cols]
        .fillna(0)
    )

    # Preserve NaN for transaction rates when no purchase history exists.
    features["payment_success_rate"] = np.where(
        features["transaction_count"].notna(),
        features["paid_transactions"]
        / features["transaction_count"],
        np.nan,
    )

    features["failed_payment_rate"] = np.where(
        features["transaction_count"].notna(),
        features["failed_transactions"]
        / features["transaction_count"],
        np.nan,
    )

    features["pending_payment_rate"] = np.where(
        features["transaction_count"].notna(),
        features["pending_transactions"]
        / features["transaction_count"],
        np.nan,
    )

    return features.drop(
        columns=["last_transaction_date"],
        errors="ignore",
    )