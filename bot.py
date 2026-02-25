"""
PRAGYAM Telegram Bot
━━━━━━━━━━━━━━━━━━━━
Minimal, professional Telegram interface for Pragyam Portfolio Intelligence.

Concurrency model:
    • PTB handles multiple users simultaneously (concurrent_updates=True)
    • Engine runs in a dedicated ThreadPoolExecutor (10 workers = 10 parallel portfolios)
    • Bot polling works in both main thread and daemon thread (Streamlit Cloud)
"""

import os
import sys
import io
import asyncio
import logging
import time
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ConversationHandler, filters, ContextTypes
)
from telegram.constants import ChatAction, ParseMode

# ─── Suppress PTB per_message warning (we intentionally use per_message=False) ───
warnings.filterwarnings("ignore", message=".*per_message.*", category=UserWarning)

# ─── Setup Path ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

# ─── Configure Logging ───
log_format = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "bot.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pragyam.bot")

for name in ['httpx', 'httpcore', 'telegram.ext', 'urllib3', 'yfinance']:
    logging.getLogger(name).setLevel(logging.WARNING)

# ─── Import Modules ───
from db import (
    init_db, register_user, log_request_start, log_request_complete,
    log_request_error, add_log
)
from portfolio_image import generate_portfolio_image

# Load environment variables
load_dotenv()

# ─── Bot Config ───
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("CRITICAL: TELEGRAM_BOT_TOKEN environment variable is not set.")

# ─── Engine Thread Pool ───
# Dedicated pool for heavy portfolio generation
MAX_ENGINE_WORKERS = 10
ENGINE_POOL = ThreadPoolExecutor(max_workers=MAX_ENGINE_WORKERS, thread_name_prefix="pragyam-engine")

# Track active engine jobs for capacity reporting
import threading as _threading
_active_jobs = _threading.Semaphore(MAX_ENGINE_WORKERS)


def _engine_has_capacity() -> bool:
    """Check if the engine pool can accept another job right now."""
    # Try acquiring without blocking — if we can, release immediately and return True
    acquired = _active_jobs.acquire(blocking=False)
    if acquired:
        _active_jobs.release()
        return True
    return False


def _get_queue_depth() -> int:
    """Approximate number of jobs waiting in the queue."""
    try:
        return ENGINE_POOL._work_queue.qsize()
    except Exception:
        return 0

# Conversation states
SELECT_STYLE, ENTER_CAPITAL, CONFIRM = range(3)

SWING_CAPITAL_PRESETS = {
    '₹1L': 100000, '₹2.5L': 250000, '₹5L': 500000,
    '₹10L': 1000000, '₹25L': 2500000, '₹50L': 5000000,
    '₹1Cr': 10000000, '₹5Cr': 50000000,
}

SIP_CAPITAL_PRESETS = {
    '₹10K': 10000, '₹20K': 20000, '₹25K': 25000, '₹50K': 50000,
    '₹1L': 100000, '₹2L': 200000, '₹2.5L': 250000, '₹5L': 500000,
}

# ─── Message Templates ───

WELCOME_MSG = """
PRAGYAM
प्रज्ञम | Portfolio Intelligence
━━━━━━━━━━━━━━━━━━
Welcome to Pragyam — our institutional-grade portfolio curation engine.

How to use:
1️⃣  Select your investment style
2️⃣  Enter your capital amount
3️⃣  Receive your curated portfolio
"""

STYLE_MSG = "📈 <b>Select Investment Style:</b>"

CAPITAL_MSG = """
💰 <b>Enter Capital Amount (₹):</b>

Style: <b>{style}</b>
<i>Min: ₹10,000</i>
"""

CONFIRM_MSG = """
💼 <b>Confirm Details:</b>

<b>Style:</b> {style}
<b>Capital:</b> ₹{capital}

Proceed?
"""

PROCESSING_MSG = """
⏳ <b>Curating your portfolio...</b>

The engine is currently running walk-forward optimizations across 90+ strategies. 
<i>This process takes 5-8 minutes.</i>

You will receive a notification here once your portfolio is ready.
"""


# ═══════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler — logs errors cleanly."""
    logger.error(f"Exception while handling update: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Something went wrong. Please try /start again.",
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass


# ═══════════════════════════════════════
# HANDLERS
# ═══════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the bot, sends welcome msg, and initiates portfolio flow."""
    user = update.effective_user
    register_user(user.id, user.username, user.first_name, user.last_name)
    add_log("INFO", "bot", f"User started bot: @{user.username}", user.id)

    await update.message.reply_text(
        WELCOME_MSG,
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove()
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Swing Trading", callback_data="style_Swing Trading")],
        [InlineKeyboardButton("📊 SIP Investment", callback_data="style_SIP Investment")],
    ])

    await update.message.reply_text(STYLE_MSG, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    return SELECT_STYLE


async def style_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    style = query.data.replace("style_", "")
    context.user_data['investment_style'] = style

    presets = SIP_CAPITAL_PRESETS if "SIP" in style else SWING_CAPITAL_PRESETS

    buttons = []
    row = []
    for label, val in presets.items():
        row.append(InlineKeyboardButton(label, callback_data=f"cap_{val}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("✏️ Custom Amount", callback_data="cap_custom")])

    await query.edit_message_text(
        CAPITAL_MSG.format(style=style),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons)
    )
    return ENTER_CAPITAL


async def capital_preset_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.replace("cap_", "")

    if data == "custom":
        await query.edit_message_text(
            "💰 <b>Enter Custom Capital Amount (₹)</b>\n\n<i>Reply with a number (e.g., 500000)</i>",
            parse_mode=ParseMode.HTML
        )
        return ENTER_CAPITAL

    context.user_data['capital'] = int(data)
    return await _show_confirmation(query, context)


async def capital_text_entered(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().replace(',', '').replace('₹', '').replace(' ', '')

    try:
        capital = float(text)
        if capital < 10000:
            await update.message.reply_text("⚠️ Minimum capital is ₹10,000:")
            return ENTER_CAPITAL

        context.user_data['capital'] = capital

        style = context.user_data['investment_style']
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Generate", callback_data="confirm_yes"),
             InlineKeyboardButton("❌ Cancel", callback_data="confirm_no")]
        ])
        await update.message.reply_text(
            CONFIRM_MSG.format(style=style, capital=f"{capital:,.0f}"),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )
        return CONFIRM

    except ValueError:
        await update.message.reply_text("⚠️ Invalid format. Please enter numbers only:")
        return ENTER_CAPITAL


async def _show_confirmation(query, context):
    style = context.user_data['investment_style']
    capital = context.user_data['capital']

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Generate", callback_data="confirm_yes"),
         InlineKeyboardButton("❌ Cancel", callback_data="confirm_no")]
    ])

    await query.edit_message_text(
        CONFIRM_MSG.format(style=style, capital=f"{capital:,.0f}"),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )
    return CONFIRM


async def confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "confirm_no":
        await query.edit_message_text("❌ Cancelled.", parse_mode=ParseMode.HTML)
        return ConversationHandler.END

    # ─── RUN THE ENGINE ───
    user = query.from_user
    style = context.user_data['investment_style']
    capital = context.user_data['capital']

    request_id = log_request_start(user.id, style, capital)
    add_log("INFO", "engine", f"Portfolio requested: {style} ₹{capital:,.0f}", user.id)
    start_time = time.time()

    # ─── Capacity check ───
    queue_depth = _get_queue_depth()
    if not _engine_has_capacity() or queue_depth > 0:
        queue_msg = (
            f"⏳ <b>Curating your portfolio...</b>\n\n"
            f"The engine is busy with other requests. "
            f"<b>Queue position: ~{queue_depth + 1}</b>\n\n"
            f"<i>Your portfolio will be generated as soon as a slot opens. "
            f"This may take 10-15 minutes. You'll be notified here.</i>"
        )
        status_msg = await query.edit_message_text(queue_msg, parse_mode=ParseMode.HTML)
    else:
        status_msg = await query.edit_message_text(PROCESSING_MSG, parse_mode=ParseMode.HTML)

    # Run engine in dedicated thread pool — wrapped in semaphore for tracking
    def _run_engine():
        _active_jobs.acquire()
        try:
            from engine import run_pragyam_pipeline
            return run_pragyam_pipeline(style, capital, callback=lambda msg, pct: None)
        finally:
            _active_jobs.release()

    try:
        loop = asyncio.get_event_loop()
        portfolio_df, metadata = await loop.run_in_executor(ENGINE_POOL, _run_engine)

        duration = time.time() - start_time

        if portfolio_df is not None and not portfolio_df.empty:
            metadata['capital'] = capital

            regime = metadata.get('regime', {}).get('name', 'N/A')
            sel_mode = metadata.get('phases', {}).get('selection', {}).get('mode', 'N/A')
            strats = metadata.get('phases', {}).get('selection', {}).get('strategies', [])
            total_val = metadata.get('phases', {}).get('curation', {}).get('total_value', 0)

            log_request_complete(request_id, len(portfolio_df), total_val, regime, sel_mode, strats, duration)

            img_bytes = generate_portfolio_image(portfolio_df, metadata)

            summary = (
                f"✅ <b>Portfolio Ready</b>\n\n"
                f"<b>Style:</b> {style}\n"
                f"<b>Regime:</b> {regime}\n"
                f"<b>Capital:</b> ₹{capital:,.0f}\n"
                f"<b>Invested:</b> ₹{total_val:,.0f}\n"
                f"<b>Positions:</b> {len(portfolio_df)}\n"
                f"<b>Duration:</b> {duration:.0f}s"
            )

            await status_msg.delete()
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=io.BytesIO(img_bytes),
                caption=summary,
                parse_mode=ParseMode.HTML
            )

            add_log("INFO", "engine", f"Portfolio delivered: {len(portfolio_df)} positions in {duration:.0f}s", user.id)

        else:
            error_msg = metadata.get('error', 'Unknown error')
            log_request_error(request_id, error_msg, duration)
            await status_msg.edit_text(
                f"❌ <b>Failed:</b> {error_msg}\n\nTry /start again.",
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        duration = time.time() - start_time
        log_request_error(request_id, str(e), duration)
        logger.error(f"Portfolio generation failed for user {user.id}: {e}", exc_info=True)
        try:
            await status_msg.edit_text(
                "❌ <b>An error occurred processing your request.</b>\nTry /start again.",
                parse_mode=ParseMode.HTML
            )
        except Exception:
            pass

    return ConversationHandler.END


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ <b>How to Use</b>\n\n"
        "1. /start to begin\n"
        "2. Choose your style & capital\n"
        "3. Wait 5-8 minutes for the engine to curate\n"
        "4. Receive your strategy-optimized portfolio",
        parse_mode=ParseMode.HTML
    )


async def fallback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Tap /start to begin.", parse_mode=ParseMode.HTML)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# ═══════════════════════════════════════
# BOT BUILDER
# ═══════════════════════════════════════

def _build_app() -> Application:
    """Build the PTB Application with all handlers configured."""
    app = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)   # Process multiple users in parallel
        .build()
    )

    app.add_error_handler(error_handler)

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', cmd_start),
            CommandHandler('portfolio', cmd_start),
        ],
        states={
            SELECT_STYLE: [CallbackQueryHandler(style_selected, pattern=r'^style_')],
            ENTER_CAPITAL: [
                CallbackQueryHandler(capital_preset_selected, pattern=r'^cap_'),
                MessageHandler(filters.TEXT & ~filters.COMMAND, capital_text_entered),
            ],
            CONFIRM: [CallbackQueryHandler(confirm_handler, pattern=r'^confirm_')],
        },
        fallbacks=[
            CommandHandler('cancel', cancel),
            MessageHandler(filters.COMMAND, fallback_handler),
        ],
        per_message=False,
    )

    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback_handler))

    return app


# ═══════════════════════════════════════
# MAIN — Thread-aware startup
# ═══════════════════════════════════════

def main():
    """Start the bot. Auto-detects main thread vs daemon thread."""
    init_db()
    logger.info("Starting PRAGYAM Telegram Bot...")

    if threading.current_thread() is threading.main_thread():
        # Standalone: python bot.py — use the convenience method (has signal handlers)
        app = _build_app()
        app.run_polling(drop_pending_updates=True)
    else:
        # Daemon thread from app.py / Streamlit — manual async loop
        _run_in_thread()


def _run_in_thread():
    """Run the bot polling loop in a non-main thread.
    
    Key details:
        • Creates its own event loop (threads don't have one by default)
        • Calls delete_webhook() FIRST to kill any lingering previous session
          and prevent the "Conflict: terminated by other getUpdates" error
        • No signal handlers (they're main-thread-only in Python)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = _build_app()

    async def _run():
        try:
            await app.initialize()

            # ─── Force takeover: kill any previous polling session ───
            # Without this, redeploying on Render/Streamlit Cloud causes
            # "Conflict: terminated by other getUpdates request" because
            # the old process might still be polling for a few seconds.
            await app.bot.delete_webhook(drop_pending_updates=True)
            logger.info("Cleared previous webhook/polling session")

            await app.start()
            await app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
            )
            logger.info("PRAGYAM Bot is polling for updates...")

            # Keep alive until daemon thread is killed with the process
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Bot polling error: {e}", exc_info=True)
        finally:
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                pass

    try:
        loop.run_until_complete(_run())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()


if __name__ == '__main__':
    main()
