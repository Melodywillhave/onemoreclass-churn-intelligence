from __future__ import annotations

from src.persistence.database import DATABASE_PATH, initialize_database


def main() -> None:
    initialize_database()

    print(f"Database initialized: {DATABASE_PATH}")


if __name__ == "__main__":
    main()