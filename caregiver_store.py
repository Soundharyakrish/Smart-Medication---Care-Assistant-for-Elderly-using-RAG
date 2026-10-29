
# caregiver_store.py

import os
import sqlite3
from pathlib import Path


# =========================================================
# DATABASE PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

_configured_path = os.getenv("CAREGIVER_DB_PATH", "").strip()

if _configured_path:
    DATABASE_PATH = Path(_configured_path).expanduser().resolve()
else:
    DATABASE_PATH = (BASE_DIR / "caregivers.db").resolve()


CONTACT_ROLES = {"primary", "backup", "doctor"}


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=15
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_database():
    """
    Create caregiver tables if they do not already exist.

    WARNING:
    SQLite creates a database file if it does not exist.
    Confirm DATABASE_PATH points to the intended database
    before allowing initialization.
    """
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS elderly_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS caregiver_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                elderly_id INTEGER NOT NULL,
                role TEXT NOT NULL
                    CHECK (
                        role IN ('primary', 'backup', 'doctor')
                    ),
                name TEXT NOT NULL,
                relationship TEXT DEFAULT '',
                phone TEXT NOT NULL,
                organization TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (elderly_id)
                    REFERENCES elderly_profiles(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_contacts_elderly
            ON caregiver_contacts(elderly_id)
        """)


# =========================================================
# HELPER
# =========================================================

def row_to_dict(row):
    return dict(row) if row is not None else None


# =========================================================
# CAREGIVER PROFILE OPERATIONS
# =========================================================

def list_profiles():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT p.*,
                (
                    SELECT COUNT(*)
                    FROM caregiver_contacts c
                    WHERE c.elderly_id = p.id
                ) AS contact_count
            FROM elderly_profiles p
            ORDER BY p.name COLLATE NOCASE
        """).fetchall()

        return [dict(row) for row in rows]


def get_profile(elderly_id):
    with get_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM elderly_profiles
            WHERE id = ?
        """, (elderly_id,)).fetchone()

        return row_to_dict(row)


def create_profile(name, age=None, notes=""):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO elderly_profiles (name, age, notes)
            VALUES (?, ?, ?)
        """, (name, age, notes))

        return cursor.lastrowid


def update_profile(elderly_id, name, age=None, notes=""):
    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE elderly_profiles
            SET name = ?, age = ?, notes = ?
            WHERE id = ?
        """, (name, age, notes, elderly_id))

        return cursor.rowcount > 0


def delete_profile(elderly_id):
    with get_connection() as connection:
        cursor = connection.execute("""
            DELETE FROM elderly_profiles
            WHERE id = ?
        """, (elderly_id,))

        return cursor.rowcount > 0


# =========================================================
# CAREGIVER CONTACT OPERATIONS
# =========================================================

def list_contacts(elderly_id):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM caregiver_contacts
            WHERE elderly_id = ?
            ORDER BY
                CASE role
                    WHEN 'primary' THEN 1
                    WHEN 'backup' THEN 2
                    WHEN 'doctor' THEN 3
                    ELSE 4
                END,
                name COLLATE NOCASE
        """, (elderly_id,)).fetchall()

        return [dict(row) for row in rows]


def get_contact(contact_id, elderly_id):
    with get_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM caregiver_contacts
            WHERE id = ? AND elderly_id = ?
        """, (contact_id, elderly_id)).fetchone()

        return row_to_dict(row)


def create_contact(
    elderly_id,
    role,
    name,
    phone,
    relationship="",
    organization="",
    notes=""
):
    if role not in CONTACT_ROLES:
        raise ValueError(
            "Role must be primary, backup, or doctor."
        )

    with get_connection() as connection:
        profile = connection.execute("""
            SELECT id
            FROM elderly_profiles
            WHERE id = ?
        """, (elderly_id,)).fetchone()

        if profile is None:
            raise ValueError("Elderly profile not found.")

        cursor = connection.execute("""
            INSERT INTO caregiver_contacts (
                elderly_id,
                role,
                name,
                phone,
                relationship,
                organization,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            elderly_id,
            role,
            name,
            phone,
            relationship,
            organization,
            notes
        ))

        return cursor.lastrowid


def update_contact(
    contact_id,
    elderly_id,
    role,
    name,
    phone,
    relationship="",
    organization="",
    notes=""
):
    if role not in CONTACT_ROLES:
        raise ValueError(
            "Role must be primary, backup, or doctor."
        )

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE caregiver_contacts
            SET role = ?,
                name = ?,
                phone = ?,
                relationship = ?,
                organization = ?,
                notes = ?
            WHERE id = ? AND elderly_id = ?
        """, (
            role,
            name,
            phone,
            relationship,
            organization,
            notes,
            contact_id,
            elderly_id
        ))

        return cursor.rowcount > 0


def delete_contact(contact_id, elderly_id):
    with get_connection() as connection:
        cursor = connection.execute("""
            DELETE FROM caregiver_contacts
            WHERE id = ? AND elderly_id = ?
        """, (contact_id, elderly_id))

        return cursor.rowcount > 0


# =========================================================
# INITIALIZE DATABASE TABLES
# =========================================================

# Creates the caregiver tables when this module is imported.
init_database()
