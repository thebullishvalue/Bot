"""
PRAGYAM Bot — User Analytics & Database
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQLite-based user tracking, request logging, and analytics.

Key fixes:
    • Auto-detects writable DB path (handles read-only mounts like Render)
    • WAL journal mode for safe concurrent access (bot thread + dashboard)
    • Connection timeout to avoid indefinite hangs
    • Explicit init_db() — not called on import
"""

import sqlite3
import json
import os
import shutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pragyam.db")

# ─── Smart DB Path Resolution ───
# On platforms like Render, the source directory (/mount/src/) is read-only.
# We try the source dir first, then fall back to a writable location.

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_NAME = "pragyam_bot.db"


def _resolve_db_path() -> str:
    """Find a writable path for the database file."""
    # 1. Prefer the source directory (local dev, Docker with writable mount)
    preferred = os.path.join(_SCRIPT_DIR, _DB_NAME)
    if _is_writable_path(preferred):
        return preferred

    # 2. Check env var override (for custom persistent volumes)
    env_path = os.environ.get("PRAGYAM_DB_PATH")
    if env_path:
        os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
        return env_path

    # 3. Fallback to /tmp (writable on virtually all platforms)
    fallback = os.path.join("/tmp", _DB_NAME)

    # If the source dir has an existing DB, copy it to the writable location
    # so we don't lose historical data on redeploy
    if os.path.exists(preferred) and not os.path.exists(fallback):
        try:
            shutil.copy2(preferred, fallback)
            logger.info(f"Copied existing DB from {preferred} → {fallback}")
        except Exception as e:
            logger.warning(f"Could not copy DB to fallback: {e}")

    logger.info(f"Using writable DB path: {fallback}")
    return fallback


def _is_writable_path(filepath: str) -> bool:
    """Check if we can write to the given filepath."""
    try:
        dirpath = os.path.dirname(filepath) or "."
        if not os.path.exists(dirpath):
            return False
        # Try creating/opening a test file in that directory
        test_file = os.path.join(dirpath, ".db_write_test")
        with open(test_file, 'w') as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except (OSError, PermissionError):
        return False


DB_PATH = _resolve_db_path()
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode and timeout."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent readers + one writer without blocking
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")
    return _local.conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            first_seen TEXT NOT NULL,
            last_active TEXT NOT NULL,
            total_requests INTEGER DEFAULT 0,
            is_blocked INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            investment_style TEXT,
            capital REAL,
            status TEXT DEFAULT 'pending',
            positions INTEGER,
            total_value REAL,
            regime TEXT,
            selection_mode TEXT,
            strategies TEXT,
            duration_seconds REAL,
            error_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS bot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            source TEXT,
            message TEXT NOT NULL,
            user_id INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id);
        CREATE INDEX IF NOT EXISTS idx_requests_time ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_time ON bot_logs(timestamp);
    """)
    conn.commit()
    logger.info(f"Database ready at: {DB_PATH}")


# ─── Write Operations (with error handling) ───

def register_user(user_id: int, username: str = None, first_name: str = None, last_name: str = None):
    """Register or update a user. Silently logs errors instead of crashing."""
    try:
        conn = _get_conn()
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO users (user_id, username, first_name, last_name, first_seen, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = COALESCE(excluded.username, users.username),
                first_name = COALESCE(excluded.first_name, users.first_name),
                last_name = COALESCE(excluded.last_name, users.last_name),
                last_active = excluded.last_active
        """, (user_id, username, first_name, last_name, now, now))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"register_user failed (user_id={user_id}): {e}")


def log_request_start(user_id: int, investment_style: str, capital: float) -> int:
    """Log a new portfolio request. Returns request_id (0 on failure)."""
    try:
        conn = _get_conn()
        now = datetime.now().isoformat()
        cur = conn.execute("""
            INSERT INTO requests (user_id, timestamp, investment_style, capital, status)
            VALUES (?, ?, ?, ?, 'running')
        """, (user_id, now, investment_style, capital))
        conn.execute(
            "UPDATE users SET total_requests = total_requests + 1, last_active = ? WHERE user_id = ?",
            (now, user_id)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        logger.error(f"log_request_start failed: {e}")
        return 0


def log_request_complete(request_id: int, positions: int, total_value: float, regime: str,
                         selection_mode: str, strategies: list, duration: float):
    """Mark a request as successfully completed."""
    if request_id == 0:
        return
    try:
        conn = _get_conn()
        conn.execute("""
            UPDATE requests SET status='success', positions=?, total_value=?, regime=?,
            selection_mode=?, strategies=?, duration_seconds=? WHERE id=?
        """, (positions, total_value, regime, selection_mode, json.dumps(strategies), duration, request_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"log_request_complete failed (id={request_id}): {e}")


def log_request_error(request_id: int, error_msg: str, duration: float):
    """Mark a request as failed."""
    if request_id == 0:
        return
    try:
        conn = _get_conn()
        conn.execute("""
            UPDATE requests SET status='error', error_message=?, duration_seconds=? WHERE id=?
        """, (error_msg, duration, request_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"log_request_error failed (id={request_id}): {e}")


def add_log(level: str, source: str, message: str, user_id: int = None):
    """Write an entry to the bot_logs table. Never raises."""
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO bot_logs (timestamp, level, source, message, user_id)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), level, source, message, user_id))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"add_log failed: {e}")


# ─── Analytics Queries ───

def get_all_users() -> List[Dict]:
    try:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM users ORDER BY last_active DESC").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error(f"get_all_users failed: {e}")
        return []


def get_all_requests(limit=100) -> List[Dict]:
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT r.*, u.username, u.first_name FROM requests r
            LEFT JOIN users u ON r.user_id = u.user_id
            ORDER BY r.timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error(f"get_all_requests failed: {e}")
        return []


def get_user_requests(user_id: int, limit=50) -> List[Dict]:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM requests WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error(f"get_user_requests failed: {e}")
        return []


def get_recent_logs(limit=200) -> List[Dict]:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM bot_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        logger.error(f"get_recent_logs failed: {e}")
        return []


def get_dashboard_stats() -> Dict:
    """Aggregate stats for the dashboard overview page."""
    defaults = {
        'total_users': 0, 'total_requests': 0, 'success_requests': 0,
        'error_requests': 0, 'running_requests': 0, 'today_requests': 0,
        'today_users': 0, 'avg_duration_seconds': 0, 'success_rate': 0,
        'style_breakdown': {}, 'avg_capital': 0, 'min_capital': 0,
        'max_capital': 0, 'total_capital_processed': 0, 'hourly_distribution': {},
    }
    try:
        conn = _get_conn()

        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_requests = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        success_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='success'").fetchone()[0]
        error_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='error'").fetchone()[0]
        running_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='running'").fetchone()[0]

        today = datetime.now().strftime('%Y-%m-%d')
        today_requests = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        today_users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM requests WHERE timestamp LIKE ?", (f"{today}%",)
        ).fetchone()[0]

        avg_duration = conn.execute(
            "SELECT AVG(duration_seconds) FROM requests WHERE status='success'"
        ).fetchone()[0] or 0

        # Style breakdown
        style_rows = conn.execute(
            "SELECT investment_style, COUNT(*) as cnt FROM requests GROUP BY investment_style"
        ).fetchall()
        style_breakdown = {r['investment_style']: r['cnt'] for r in style_rows if r['investment_style']}

        # Capital stats
        cap_stats = conn.execute("""
            SELECT AVG(capital) as avg_cap, MIN(capital) as min_cap,
                   MAX(capital) as max_cap, SUM(capital) as total_cap
            FROM requests WHERE status='success'
        """).fetchone()

        # Hourly distribution
        hour_rows = conn.execute("""
            SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, COUNT(*) as cnt
            FROM requests GROUP BY hour ORDER BY hour
        """).fetchall()
        hourly = {r['hour']: r['cnt'] for r in hour_rows}

        return {
            'total_users': total_users,
            'total_requests': total_requests,
            'success_requests': success_requests,
            'error_requests': error_requests,
            'running_requests': running_requests,
            'today_requests': today_requests,
            'today_users': today_users,
            'avg_duration_seconds': avg_duration,
            'success_rate': (success_requests / total_requests * 100) if total_requests > 0 else 0,
            'style_breakdown': style_breakdown,
            'avg_capital': (cap_stats['avg_cap'] or 0) if cap_stats else 0,
            'min_capital': (cap_stats['min_cap'] or 0) if cap_stats else 0,
            'max_capital': (cap_stats['max_cap'] or 0) if cap_stats else 0,
            'total_capital_processed': (cap_stats['total_cap'] or 0) if cap_stats else 0,
            'hourly_distribution': hourly,
        }
    except sqlite3.Error as e:
        logger.error(f"get_dashboard_stats failed: {e}")
        return defaults
