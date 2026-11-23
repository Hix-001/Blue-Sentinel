import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .security import hash_password

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "users.db"

DEFAULT_USERS = [
    {
        "username": "COM101",
        "password": "hix001com",
        "role": "COMMANDER",
        "callsign": "Commander Hix"
    },
    {
        "username": "ANA202",
        "password": "hix001ana",
        "role": "ANALYST",
        "callsign": "Analyst Vance"
    },
    {
        "username": "OPR303",
        "password": "hix001opr",
        "role": "OPERATOR",
        "callsign": "Operator Chen"
    }
]

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes users table and seeds the 3 default operational accounts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            callsign TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Seed default users if not already present
    for u in DEFAULT_USERS:
        cursor.execute("SELECT id FROM users WHERE username = ?", (u["username"],))
        if not cursor.fetchone():
            pwd_hash, salt = hash_password(u["password"])
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, role, callsign, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (u["username"], pwd_hash, salt, u["role"], u["callsign"], now)
            )
            print(f"[Auth Engine] Seeded security account: {u['username']} [{u['role']}]")
    conn.commit()
    conn.close()

def get_user_by_username(username: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
