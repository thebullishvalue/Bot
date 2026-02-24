import subprocess
import sys
import os
import multiprocessing
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pragyam.launcher")

def start_bot():
    """Start the Telegram bot in a separate process."""
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
        # Run Streamlit via subprocess (mimics 'streamlit run') without port/address
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--theme.base", "dark"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Start bot in a separate process
    bot_process = multiprocessing.Process(target=start_bot)
    bot_process.start()
    
    # Give bot a moment to initialize
    time.sleep(2)
    
    # Start dashboard in the main process (blocks until exit)
    start_dashboard()
    
    # Wait for bot process (though it will terminate with the app)
    bot_process.join()
