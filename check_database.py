
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent / "caregivers.db"

if not db_path.exists():
    print("Database file not found:", db_path)
else:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """)

        tables = cursor.fetchall()

        print("Database:", db_path)
        print("\nTables found:")

        for (table_name,) in tables:
            print(f"\n- {table_name}")

            # Display column names without changing data.
            cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            )
            columns = cursor.fetchall()

            for column in columns:
                print("  Column:", column[1], "| Type:", column[2])
