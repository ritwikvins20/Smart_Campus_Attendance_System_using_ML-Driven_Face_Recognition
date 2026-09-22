"""
auth.py — Authentication helpers for Face Recognition Attendance System.

WARNING: SHA-256 without salt is prototype-only.
In production, replace with salted bcrypt or argon2 via the `bcrypt`
or `argon2-cffi` library.
"""

import hashlib
import sqlite3

import db

# ---------------------------------------------------------------------------
# Hardcoded prototype users  (username → (hashed_password, role))
# ---------------------------------------------------------------------------
# Passwords were hashed with: hashlib.sha256(password.encode()).hexdigest()
#
# staff1   / staff123   → role: staff
# cr1      / cr123      → role: staff
# faculty1 / fac123     → role: faculty
# ---------------------------------------------------------------------------

_SEED_USERS = [
    ("staff1",   "staff123",  "staff"),
    ("cr1",      "cr123",     "staff"),
    ("faculty1", "fac123",    "faculty"),
]


def hash_password(plain: str) -> str:
    """
    Return SHA-256 hex digest of a plaintext password.
    NOTE: Prototype-only — no salt. Use bcrypt/argon2 in production.
    """
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def seed_users() -> None:
    """
    Insert the prototype user accounts into the `users` table if it is empty.
    Called once at application startup.
    """
    conn = db.get_db_connection()
    cursor = conn.cursor()

    # Create users table if it doesn't exist yet
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL
        )
    """)

    # Only seed when empty to avoid re-inserting on every restart
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        for username, password, role in _SEED_USERS:
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, hash_password(password), role),
            )
        print("auth.py: Prototype users seeded into `users` table.")

    conn.commit()
    conn.close()


def verify_login(username: str, password: str):
    """
    Check credentials against the `users` table.

    Returns:
        str: Role ('staff' or 'faculty') if credentials are valid.
        None: If username or password is incorrect.
    """
    if not username or not password:
        return None

    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password_hash, role FROM users WHERE username = ?",
        (username.strip(),),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    expected_hash = row["password_hash"]
    actual_hash = hash_password(password)

    if actual_hash == expected_hash:
        return row["role"]

    return None
