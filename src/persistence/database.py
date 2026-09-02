from __future__ import annotations

import sqlite3
from pathlib import Path

DATABASE_PATH = Path("data/database/churn_intelligence.db")


def get_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database(
    database_path: Path = DATABASE_PATH,
) -> None:
    """Create the application database tables if they do not exist."""
    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_risk (
                customer_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_rank INTEGER NOT NULL,
                risk_percentile REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                primary_driver TEXT NOT NULL,
                secondary_driver TEXT NOT NULL,
                PRIMARY KEY (customer_id, snapshot_date)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retention_decisions (
                customer_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                risk_tier TEXT NOT NULL,
                primary_driver TEXT NOT NULL,
                secondary_driver TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                action_priority TEXT NOT NULL,
                objective TEXT NOT NULL,
                supporting_signals TEXT NOT NULL,
                PRIMARY KEY (customer_id, snapshot_date)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                customer_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                recommendation_json TEXT NOT NULL,
                PRIMARY KEY (customer_id, snapshot_date)
            )
            """
        )

        connection.commit()