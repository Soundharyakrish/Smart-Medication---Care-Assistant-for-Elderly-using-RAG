
# medication_store.py

import json
import os
import sqlite3
from pathlib import Path
from datetime import date, datetime


# =========================================================
# DATABASE PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Option 1: Set CAREGIVER_DB_PATH to the full path of your
# existing caregiver database using an environment variable.
#
# Option 2: If caregivers.db is beside this file, the default
# path below will be used.
#
# IMPORTANT: The database must already contain elderly_profiles.

_configured_db_path = os.getenv("CAREGIVER_DB_PATH", "").strip()

if _configured_db_path:
    DB_PATH = Path(_configured_db_path).expanduser().resolve()
else:
    DB_PATH = (BASE_DIR / "caregivers.db").resolve()


VALID_DOSE_STATUSES = {"Pending", "Taken", "Skipped"}


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    """
    Connect to the existing caregiver database.

    Do not silently create a new empty database if the configured
    database file does not exist.
    """
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Caregiver database not found: {DB_PATH}\n"
            "Set CAREGIVER_DB_PATH to the full path of your "
            "existing caregiver database."
        )

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=15
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")

    return connection


def _parse_iso_date(value, field_name="date"):
    """Validate a date strictly in YYYY-MM-DD format."""
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    if parsed.isoformat() != value:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        )

    return parsed


def _parse_times_json(times_json):
    """Read and validate the stored list of daily reminder times."""
    if not isinstance(times_json, str):
        raise ValueError(
            "Stored medication times must be JSON text."
        )

    try:
        times = json.loads(times_json)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            "Medication schedule contains invalid JSON."
        )

    if not isinstance(times, list) or not times:
        raise ValueError(
            "Medication schedule must contain at least one time."
        )

    valid_times = []

    for value in times:
        if not isinstance(value, str):
            raise ValueError(
                "Medication times must be strings."
            )

        try:
            parsed = datetime.strptime(value, "%H:%M")
        except ValueError:
            raise ValueError(
                "Medication times must use HH:MM format."
            )

        if parsed.strftime("%H:%M") != value:
            raise ValueError(
                "Medication times must use HH:MM format."
            )

        valid_times.append(value)

    if len(valid_times) != len(set(valid_times)):
        raise ValueError(
            "Medication schedule contains duplicate times."
        )

    return sorted(valid_times)


# =========================================================
# DATABASE INITIALIZATION - NON-DESTRUCTIVE
# =========================================================

def init_medication_db():
    """
    Initialize medication tables in the existing caregiver database.

    This function does not delete existing tables or records.
    The existing elderly_profiles table must already be present.
    """
    with get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
            """).fetchall()
        }

        if "elderly_profiles" not in tables:
            raise RuntimeError(
                "The elderly_profiles table was not found in "
                f"the configured database: {DB_PATH}. "
                "Check that CAREGIVER_DB_PATH points to the "
                "correct existing caregiver database."
            )

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(elderly_profiles)"
            ).fetchall()
        }

        if "id" not in columns:
            raise RuntimeError(
                "Expected elderly_profiles.id. "
                "Adjust medication_store.py to match your schema."
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elderly_id INTEGER NOT NULL,
                medicine_name TEXT NOT NULL,
                dosage TEXT NOT NULL,
                instructions TEXT DEFAULT '',
                times_json TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS medication_doses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                medication_id INTEGER NOT NULL,
                elderly_id INTEGER NOT NULL,
                scheduled_date TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (
                        status IN ('Pending', 'Taken', 'Skipped')
                    ),
                updated_at TEXT NOT NULL,

                UNIQUE (
                    medication_id,
                    scheduled_date,
                    scheduled_time
                ),

                FOREIGN KEY (medication_id)
                    REFERENCES medications(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_medications_elderly
            ON medications(elderly_id, active)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doses_elderly_date
            ON medication_doses(elderly_id, scheduled_date)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doses_medication_date
            ON medication_doses(medication_id, scheduled_date)
        """)


# =========================================================
# PROFILE LOOKUP
# =========================================================

def profile_exists(elderly_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT id
            FROM elderly_profiles
            WHERE id = ?
        """, (elderly_id,)).fetchone()

        return row is not None


# =========================================================
# MEDICATION READ OPERATIONS
# =========================================================

def get_medications(elderly_id, include_inactive=False):
    sql = """
        SELECT *
        FROM medications
        WHERE elderly_id = ?
    """
    params = [elderly_id]

    if not include_inactive:
        sql += " AND active = 1"

    sql += " ORDER BY medicine_name COLLATE NOCASE"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def get_medication(medication_id, elderly_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT *
            FROM medications
            WHERE id = ? AND elderly_id = ?
        """, (medication_id, elderly_id)).fetchone()

        return dict(row) if row else None


# =========================================================
# CREATE MEDICATION
# =========================================================

def create_medication(
    elderly_id,
    medicine_name,
    dosage,
    instructions,
    times_json,
    start_date,
    end_date
):
    # Validate before writing to the database.
    _parse_times_json(times_json)
    start = _parse_iso_date(start_date, "Start date")

    if end_date:
        end = _parse_iso_date(end_date, "End date")

        if end < start:
            raise ValueError(
                "End date cannot be before start date."
            )

    with get_connection() as conn:
        profile = conn.execute(
            "SELECT id FROM elderly_profiles WHERE id = ?",
            (elderly_id,)
        ).fetchone()

        if not profile:
            raise ValueError("Elderly profile not found.")

        cursor = conn.execute("""
            INSERT INTO medications (
                elderly_id,
                medicine_name,
                dosage,
                instructions,
                times_json,
                start_date,
                end_date,
                active,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
        """, (
            elderly_id,
            medicine_name.strip(),
            dosage.strip(),
            instructions.strip(),
            times_json,
            start_date,
            end_date,
            datetime.now().isoformat(timespec="seconds")
        ))

        return cursor.lastrowid


# =========================================================
# UPDATE MEDICATION
# =========================================================

def update_medication(
    medication_id,
    elderly_id,
    medicine_name,
    dosage,
    instructions,
    times_json,
    start_date,
    end_date
):
    _parse_times_json(times_json)
    start = _parse_iso_date(start_date, "Start date")

    if end_date:
        end = _parse_iso_date(end_date, "End date")

        if end < start:
            raise ValueError(
                "End date cannot be before start date."
            )

    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE medications
            SET medicine_name = ?,
                dosage = ?,
                instructions = ?,
                times_json = ?,
                start_date = ?,
                end_date = ?
            WHERE id = ?
              AND elderly_id = ?
              AND active = 1
        """, (
            medicine_name.strip(),
            dosage.strip(),
            instructions.strip(),
            times_json,
            start_date,
            end_date,
            medication_id,
            elderly_id
        ))

        # Existing dose records are deliberately retained.
        return cursor.rowcount > 0


# =========================================================
# DEACTIVATE MEDICATION
# =========================================================

def deactivate_medication(medication_id, elderly_id):
    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE medications
            SET active = 0
            WHERE id = ?
              AND elderly_id = ?
              AND active = 1
        """, (medication_id, elderly_id))

        # Soft deactivation preserves dose history.
        return cursor.rowcount > 0


# =========================================================
# RELIABLE DAILY DOSE GENERATION
# =========================================================

def get_daily_doses(elderly_id, target_date=None):
    """
    Return doses for a date and create missing scheduled records.

    Repeated calls do not create duplicate doses.
    Existing Taken/Skipped statuses are not overwritten.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    requested_day = _parse_iso_date(
        target_date,
        "Requested date"
    )

    target_date = requested_day.isoformat()
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        profile = conn.execute("""
            SELECT id
            FROM elderly_profiles
            WHERE id = ?
        """, (elderly_id,)).fetchone()

        if profile is None:
            raise ValueError("Elderly profile not found.")

        medications = conn.execute("""
            SELECT *
            FROM medications
            WHERE elderly_id = ?
              AND active = 1
            ORDER BY id
        """, (elderly_id,)).fetchall()

        for medication in medications:
            start = _parse_iso_date(
                medication["start_date"],
                "Medication start date"
            )

            end_value = medication["end_date"]

            if end_value:
                end = _parse_iso_date(
                    end_value,
                    "Medication end date"
                )
            else:
                end = None

            if requested_day < start:
                continue

            if end is not None and requested_day > end:
                continue

            times = _parse_times_json(
                medication["times_json"]
            )

            for scheduled_time in times:
                conn.execute("""
                    INSERT OR IGNORE INTO medication_doses (
                        medication_id,
                        elderly_id,
                        scheduled_date,
                        scheduled_time,
                        status,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, 'Pending', ?)
                """, (
                    medication["id"],
                    elderly_id,
                    target_date,
                    scheduled_time,
                    now
                ))

        rows = conn.execute("""
            SELECT
                d.id AS dose_id,
                d.medication_id,
                d.elderly_id,
                d.scheduled_date,
                d.scheduled_time,
                d.status,
                m.medicine_name,
                m.dosage,
                m.instructions
            FROM medication_doses AS d
            JOIN medications AS m
              ON m.id = d.medication_id
            WHERE d.elderly_id = ?
              AND d.scheduled_date = ?
              AND m.active = 1
              AND EXISTS (
                  SELECT 1
                  FROM json_each(m.times_json) AS jt
                  WHERE jt.value = d.scheduled_time
              )
              AND d.scheduled_date >= m.start_date
              AND (
                  m.end_date = ''
                  OR d.scheduled_date <= m.end_date
              )
            ORDER BY d.scheduled_time, d.id
        """, (elderly_id, target_date)).fetchall()

        return [dict(row) for row in rows]


# =========================================================
# DOSE STATUS UPDATE
# =========================================================

def update_dose_status(dose_id, elderly_id, status):
    if status not in VALID_DOSE_STATUSES:
        raise ValueError(
            "Status must be Pending, Taken, or Skipped."
        )

    with get_connection() as conn:
        cursor = conn.execute("""
            UPDATE medication_doses
            SET status = ?,
                updated_at = ?
            WHERE id = ?
              AND elderly_id = ?
        """, (
            status,
            datetime.now().isoformat(timespec="seconds"),
            dose_id,
            elderly_id
        ))

        return cursor.rowcount > 0


# =========================================================
# DOSE HISTORY
# =========================================================

def get_dose_history(elderly_id, limit=100):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError(
            "History limit must be an integer."
        )

    limit = max(1, min(limit, 200))

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                d.id AS dose_id,
                d.medication_id,
                d.scheduled_date,
                d.scheduled_time,
                d.status,
                d.updated_at,
                m.medicine_name,
                m.dosage
            FROM medication_doses AS d
            JOIN medications AS m
              ON m.id = d.medication_id
            WHERE d.elderly_id = ?
            ORDER BY
                d.scheduled_date DESC,
                d.scheduled_time DESC,
                d.id DESC
            LIMIT ?
        """, (elderly_id, limit)).fetchall()

        return [dict(row) for row in rows]


# =========================================================
# INITIALIZE MEDICATION TABLES
# =========================================================

# This requires the existing caregiver database to be available.
# It creates only missing medication tables and indexes.
# It does not delete existing records.
init_medication_db()
