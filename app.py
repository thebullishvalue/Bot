"""
PRAGYAM — Entry Point
━━━━━━━━━━━━━━━━━━━━━
Hemrek Capital Portfolio Intelligence Distribution System

This is the Streamlit entry point. It:
  1. Configures shared logging (including bot.log file)
  2. Initializes the database
  3. Starts the Telegram bot in a daemon thread
  4. Renders the admin dashboard

Streamlit Cloud: set main file to app.py
Local dev:      streamlit run app.py
"""

import os
import sys
import threading
import logging

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


# ─── Writable log path ───
def _writable_log_path() -> str:
    """Find a writable location for bot.log."""
    # Try source dir first (works in most local/Docker setups)
    preferred = os.path.join(SCRIPT_DIR, "bot.log")
    try:
        with open(preferred, 'a') as f:
            pass  # Just test if we can open for append
        return preferred
    except (OSError, PermissionError):
        pass
    # Fallback to /tmp
    return "/tmp/bot.log"


LOG_FILE = _writable_log_path()

# ─── Logging (one central setup for the whole process) ───
log_format = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
    force=True  # Override any previous basicConfig (critical for reliability)
)
logger = logging.getLogger("pragyam.launcher")

for name in ['httpx', 'httpcore', 'telegram.ext', 'urllib3', 'yfinance']:
    logging.getLogger(name).setLevel(logging.WARNING)

logger.info(f"Log file: {LOG_FILE}")


# ─── Initialize Database ───
from db import init_db
init_db()


# ─── Bot Thread (once per process) ───
import db as _db_module


def _ensure_bot_running():
    """Start the Telegram bot in a daemon thread. Only starts once per process.
    Checks thread liveness — restarts if the thread crashed."""

    existing = getattr(_db_module, '_bot_thread', None)
    if existing is not None and existing.is_alive():
        return

    lock = getattr(_db_module, '_bot_lock', None)
    if lock is None:
        _db_module._bot_lock = threading.Lock()
        lock = _db_module._bot_lock

    with lock:
        existing = getattr(_db_module, '_bot_thread', None)
        if existing is not None and existing.is_alive():
            return

        try:
            from bot import main as bot_main
            t = threading.Thread(target=bot_main, name="telegram-bot", daemon=True)
            t.start()
            _db_module._bot_thread = t
            logger.info("━━ Telegram Bot thread started ━━")
        except Exception as e:
            logger.error(f"Bot failed to start: {e}", exc_info=True)


_ensure_bot_running()


# ─── Dashboard (Streamlit UI) ───
# Import and call the render function — no exec(), no subprocess.
# set_page_config is inside render() so it's the first st call.
from dashboard import render as _render_dashboard
_render_dashboard(log_file_path=LOG_FILE)
