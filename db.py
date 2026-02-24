"""
PRAGYAM Bot - User Analytics & Database
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQLite-based user tracking, request logging, and analytics.
"""

import sqlite3
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pragyam_bot.db")
_local = threading.local()


def _get_conn():
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
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


def register_user(user_id: int, username: str = None, first_name: str = None, last_name: str = None):
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


def log_request_start(user_id: int, investment_style: str, capital: float) -> int:
    conn = _get_conn()
    now = datetime.now().isoformat()
    cur = conn.execute("""
        INSERT INTO requests (user_id, timestamp, investment_style, capital, status)
        VALUES (?, ?, ?, ?, 'running')
    """, (user_id, now, investment_style, capital))
    conn.execute("UPDATE users SET total_requests = total_requests + 1, last_active = ? WHERE user_id = ?", (now, user_id))
    conn.commit()
    return cur.lastrowid


def log_request_complete(request_id: int, positions: int, total_value: float, regime: str,
                         selection_mode: str, strategies: list, duration: float):
    conn = _get_conn()
    conn.execute("""
        UPDATE requests SET status='success', positions=?, total_value=?, regime=?,
        selection_mode=?, strategies=?, duration_seconds=? WHERE id=?
    """, (positions, total_value, regime, selection_mode, json.dumps(strategies), duration, request_id))
    conn.commit()


def log_request_error(request_id: int, error_msg: str, duration: float):
    conn = _get_conn()
    conn.execute("""
        UPDATE requests SET status='error', error_message=?, duration_seconds=? WHERE id=?
    """, (error_msg, duration, request_id))
    conn.commit()


def add_log(level: str, source: str, message: str, user_id: int = None):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO bot_logs (timestamp, level, source, message, user_id) VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), level, source, message, user_id))
    conn.commit()


# ─── Analytics Queries ───

def get_all_users() -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM users ORDER BY last_active DESC").fetchall()
    return [dict(r) for r in rows]


def get_all_requests(limit=100) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("""
        SELECT r.*, u.username, u.first_name FROM requests r
        LEFT JOIN users u ON r.user_id = u.user_id
        ORDER BY r.timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_user_requests(user_id: int, limit=50) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM requests WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_recent_logs(limit=200) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM bot_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> Dict:
    conn = _get_conn()
    
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_requests = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    success_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='success'").fetchone()[0]
    error_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='error'").fetchone()[0]
    running_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE status='running'").fetchone()[0]
    
    today = datetime.now().strftime('%Y-%m-%d')
    today_requests = conn.execute("SELECT COUNT(*) FROM requests WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()[0]
    today_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM requests WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()[0]
    
    avg_duration = conn.execute("SELECT AVG(duration_seconds) FROM requests WHERE status='success'").fetchone()[0] or 0
    
    # Style breakdown
    style_rows = conn.execute("SELECT investment_style, COUNT(*) as cnt FROM requests GROUP BY investment_style").fetchall()
    style_breakdown = {r['investment_style']: r['cnt'] for r in style_rows}
    
    # Capital stats
    cap_stats = conn.execute("SELECT AVG(capital) as avg_cap, MIN(capital) as min_cap, MAX(capital) as max_cap, SUM(capital) as total_cap FROM requests WHERE status='success'").fetchone()
    
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
        'avg_capital': cap_stats['avg_cap'] or 0 if cap_stats else 0,
        'min_capital': cap_stats['min_cap'] or 0 if cap_stats else 0,
        'max_capital': cap_stats['max_cap'] or 0 if cap_stats else 0,
        'total_capital_processed': cap_stats['total_cap'] or 0 if cap_stats else 0,
        'hourly_distribution': hourly,
    }


# Init on import
init_db()
