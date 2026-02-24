"""
PRAGYAM — Single Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hemrek Capital Portfolio Intelligence Distribution System

Run this file to start everything:
    python app.py

Architecture:
    • Telegram Bot runs in a daemon thread (dies with the main process)
    • Streamlit Dashboard runs as the foreground subprocess (serves the UI)
    • SQLite DB is shared via WAL mode for safe concurrent access
"""

import subprocess
import sys
import os
import threading
import logging
import signal
import atexit
import time

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

PID_FILE = os.path.join(SCRIPT_DIR, "bot.pid")
LOCK_FILE = os.path.join(SCRIPT_DIR, "bot.lock")

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
)
logger = logging.getLogger("pragyam.launcher")

# Quiet noisy loggers
for name in ['httpx', 'httpcore', 'telegram.ext', 'urllib3', 'yfinance']:
    logging.getLogger(name).setLevel(logging.WARNING)


# ─── Cleanup ───
def _cleanup():
    """Remove PID and stale lock files on exit."""
    for f in [PID_FILE, LOCK_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass

atexit.register(_cleanup)


def _write_pid():
    """Write current PID so dashboard can detect bot status."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
    except OSError as e:
        logger.warning(f"Could not write PID file: {e}")


# ─── Bot Thread ───
def _run_bot():
    """Start the Telegram bot (runs in a daemon thread)."""
    try:
        from bot import main as bot_main
        logger.info("━━ Telegram Bot starting ━━")
        bot_main()
    except Exception as e:
        logger.error(f"Bot thread crashed: {e}", exc_info=True)


def start_bot_thread() -> threading.Thread:
    """Launch the bot in a daemon thread that dies with the main process."""
    t = threading.Thread(target=_run_bot, name="telegram-bot", daemon=True)
    t.start()
    logger.info(f"Bot thread started (thread={t.name})")
    return t


# ─── Dashboard (foreground) ───
def start_dashboard():
    """Start Streamlit dashboard as the main foreground process."""
    logger.info("━━ Admin Dashboard starting ━━")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--theme.base", "dark",
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--server.port", os.environ.get("PORT", "8501"),
        "--browser.gatherUsageStats", "false",
    ]
    try:
        subprocess.run(cmd, check=False, cwd=SCRIPT_DIR)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")


# ─── Main ───
if __name__ == "__main__":
    # Clean stale files from previous runs
    _cleanup()

    # Initialize DB (creates tables if needed)
    try:
        from db import init_db
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init failed: {e}", exc_info=True)

    # Write PID for dashboard status display
    _write_pid()

    # Handle signals gracefully
    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received — cleaning up")
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start bot in background thread
    bot_thread = start_bot_thread()

    # Give bot a moment to initialize before dashboard takes over
    time.sleep(2)

    # Start dashboard in foreground (blocks until exit)
    start_dashboard()

    # Cleanup on exit
    _cleanup()
