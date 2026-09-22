"""
reset_db.py — Database reset script for Face Recognition Attendance System.

Preserves: students table and all stored face embeddings.
Recreates:  attendance table (with new approval_status column) and users table.
"""

import os
import glob
import sqlite3


def reset_database():
    # 1. Find .db file in project folder
    db_files = glob.glob("*.db")
    if not db_files:
        print("Error: No .db file found in the project folder.")
        return

    db_path = "attendance.db" if "attendance.db" in db_files else db_files[0]
    print(f"Found database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 2. Count & clear attendance (preserve students + embeddings)
    # ------------------------------------------------------------------
    try:
        cursor.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        attendance_count = 0

    # Drop old attendance table (may be missing approval_status column)
    cursor.execute("DROP TABLE IF EXISTS attendance")

    # Recreate attendance with new schema
    cursor.execute("""
        CREATE TABLE attendance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      INTEGER NOT NULL,
            date            TEXT    NOT NULL,
            time            TEXT    NOT NULL,
            status          TEXT    NOT NULL,
            confidence      REAL,
            approval_status TEXT    NOT NULL DEFAULT 'pending',
            marked_by       TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)
    print(f"Deleted {attendance_count} row(s) from 'attendance' table. Table recreated with new schema.")

    # ------------------------------------------------------------------
    # 3. Clear users table
    # ------------------------------------------------------------------
    try:
        cursor.execute("SELECT COUNT(*) FROM users")
        users_count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM users")
        print(f"Deleted {users_count} row(s) from 'users' table.")
    except sqlite3.OperationalError:
        # Table doesn't exist yet — create it
        users_count = 0
        print("'users' table not found — will be created at app startup.")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL
        )
    """)

    # ------------------------------------------------------------------
    # 4. Reset autoincrement sequences for recreated tables
    # ------------------------------------------------------------------
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('attendance', 'users')")
    except sqlite3.OperationalError:
        pass  # sqlite_sequence may not exist if no AUTOINCREMENT rows yet

    # ------------------------------------------------------------------
    # 5. Verify students table is intact
    # ------------------------------------------------------------------
    cursor.execute("SELECT COUNT(*) FROM students")
    students_count = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    print(f"'students' table preserved: {students_count} student(s) with embeddings intact.")
    print("Database cleaned. Ready for fresh registrations.")


if __name__ == "__main__":
    reset_database()
