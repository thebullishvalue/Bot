"""
PRAGYAM — Single Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hemrek Capital Portfolio Intelligence Distribution System

Works in TWO modes automatically:

  1. Streamlit Cloud / `streamlit run app.py`
     → Bot starts as a daemon thread
     → Dashboard renders directly (we're already inside Streamlit)

  2. Local / `python app.py`
     → Bot starts as a daemon thread
     → Launches `streamlit run dashboard.py` as a subprocess
"""

import os
import sys
import threading
import logging

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
)
logger = logging.getLogger("pragyam.launcher")

for name in ['httpx', 'httpcore', 'telegram.ext', 'urllib3', 'yfinance']:
    logging.getLogger(name).setLevel(logging.WARNING)


# ─── Detect Streamlit ───
def _inside_streamlit() -> bool:
    """Check if we're being executed by Streamlit's script runner."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ─── Initialize Database ───
from db import init_db
init_db()


# ─── Bot Thread (starts ONCE per Python process) ───
# CRITICAL: Streamlit re-executes this script on every user interaction,
# so module-level variables here get reset. We store the flag on an imported
# module (db) because Python caches imported modules — their attributes persist.
import db as _db_module


def _ensure_bot_running():
    """Start the Telegram bot thread exactly once per process lifetime."""
    if getattr(_db_module, '_bot_thread_started', False):
        return  # Already running from a previous Streamlit rerun

    # Use a lock in case of race conditions during first render
    _lock = getattr(_db_module, '_bot_lock', None)
    if _lock is None:
        _db_module._bot_lock = threading.Lock()
        _lock = _db_module._bot_lock

    with _lock:
        if getattr(_db_module, '_bot_thread_started', False):
            return
        try:
            from bot import main as bot_main
            t = threading.Thread(target=bot_main, name="telegram-bot", daemon=True)
            t.start()
            _db_module._bot_thread_started = True
            logger.info("━━ Telegram Bot thread started ━━")

            # Write PID so dashboard sidebar can show bot status
            try:
                with open(os.path.join(SCRIPT_DIR, "bot.pid"), 'w') as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
        except Exception as e:
            logger.error(f"Bot failed to start: {e}", exc_info=True)


# Start bot (guarded — only once per process)
_ensure_bot_running()


# ─── Route to correct mode ───
if _inside_streamlit():
    # MODE 1: Streamlit Cloud / `streamlit run app.py`
    # We're already inside Streamlit — render the dashboard directly
    # by executing dashboard.py in this script's context.
    _dashboard_path = os.path.join(SCRIPT_DIR, "dashboard.py")
    with open(_dashboard_path) as _f:
        exec(compile(_f.read(), _dashboard_path, "exec"))

else:
    # MODE 2: Local dev / `python app.py`
    # Launch Streamlit as a subprocess (foreground, blocks until exit)
    if __name__ == "__main__":
        import subprocess
        import signal
        import time
        import atexit

        PID_FILE = os.path.join(SCRIPT_DIR, "bot.pid")
        LOCK_FILE = os.path.join(SCRIPT_DIR, "bot.lock")

        def _cleanup():
            for f in [PID_FILE, LOCK_FILE]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except OSError:
                    pass

        atexit.register(_cleanup)

        def _signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            _cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        logger.info("━━ Admin Dashboard starting (subprocess) ━━")
        time.sleep(2)  # Let bot initialize

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
            logger.info("Stopped by user")
        finally:
            _cleanup()
