import sqlite3
import os

def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    return conn

def init_database():
    """Initialize the database with required tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            year INTEGER NOT NULL,
            section TEXT NOT NULL,
            face_embedding BLOB
        )
    ''')
    
    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def check_database_exists():
    """Check if the database file exists, create it if not."""
    db_path = 'attendance.db'
    
    if not os.path.exists(db_path):
        print("Database not found. Creating new database...")
        init_database()
    else:
        print("Database already exists!")
    
    return os.path.exists(db_path)

if __name__ == "__main__":
    # Test the database initialization
    check_database_exists()
