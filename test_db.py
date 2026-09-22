import db

# Test the database initialization
if db.check_database_exists():
    print("Database initialized successfully!")
else:
    print("Database initialization failed!")
