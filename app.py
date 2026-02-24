import subprocess
import sys
import os
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pragyam.launcher")

def start_bot():
    """Start the Telegram bot in a thread."""
    try:
        from bot import main as bot_main  # Import the bot's main function
        logger.info("Starting Telegram Bot...")
        bot_main()  # This runs the bot's polling loop
    except ImportError as e:
        logger.error(f"Failed to import bot.py: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Bot failed: {e}")
        sys.exit(1)

def start_dashboard():
    """Start the Streamlit dashboard."""
    try:
        logger.info("Starting Admin Dashboard...")
        # Run Streamlit via subprocess (mimics 'streamlit run')
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "dark"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error starting dashboard: {e}")
        sys.exit(1)

def main():
    # Start bot in a background thread
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    
    # Give bot a moment to initialize
    time.sleep(2)
    
    # Start dashboard in the main thread (blocks until exit)
    start_dashboard()
    
    # Optional: Wait for bot thread (though daemon=True means it exits with main)
    bot_thread.join()

if __name__ == "__main__":
    main()
