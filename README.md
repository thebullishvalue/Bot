# PRAGYAM — Portfolio Intelligence Bot
### Hemrek Capital Distribution System

Institutional-grade portfolio curation engine delivered via Telegram bot with a real-time admin dashboard.

---

## Architecture

```
app.py  ←  Single entry point (python app.py)
  ├── Bot Thread (daemon)   →  bot.py  →  engine.py  →  strategies.py
  │                                    →  db.py (SQLite + WAL)
  └── Dashboard (foreground) →  streamlit run dashboard.py
                                       →  db.py (SQLite + WAL)
```

- **`app.py`** — The only file you run. Starts the Telegram bot in a background daemon thread and launches the Streamlit admin dashboard as the foreground process.
- **`bot.py`** — Telegram bot with conversation flow: style selection → capital input → confirmation → engine run → portfolio image delivery.
- **`dashboard.py`** — Streamlit admin dashboard with overview metrics, user management, request logs, terminal output, and analytics.
- **`db.py`** — SQLite database with WAL mode for safe concurrent access. Auto-detects writable path (handles read-only mounts like Render).
- **`engine.py`** — Headless 4-phase pipeline: data fetch → regime detection → strategy selection → walk-forward → curation.
- **`backdata.py`** — Market data download and indicator calculation via yfinance.
- **`strategies.py`** — 90+ quantitative trading strategies.
- **`backtest_engine.py`** — Unified walk-forward backtest engine.
- **`strategy_selection.py`** — Dynamic strategy selection using market breadth data.
- **`portfolio_image.py`** — Generates clean portfolio table images via matplotlib.
- **`charts.py`** — Chart generation utilities.

---

## Quick Start

### 1. Environment Setup

```bash
# Create .env file with your Telegram bot token
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
# or
./run.sh
```

This starts:
- ✅ Telegram bot (background thread)
- ✅ Admin dashboard at http://localhost:8501

### 3. Use the Bot

Open your Telegram bot and send `/start`. Follow the flow:
1. Select investment style (Swing Trading / SIP Investment)
2. Choose or enter capital amount
3. Confirm and wait 5-8 minutes for the engine
4. Receive your curated portfolio image

---

## Deployment (Render / Cloud)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | ✅ | Telegram Bot API token |
| `PORT` | ❌ | Dashboard port (default: 8501) |
| `PRAGYAM_DB_PATH` | ❌ | Custom DB file path (for persistent volumes) |

### Render Configuration

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`
- **Health Check Path:** `/` (Streamlit serves on the configured port)

### Database Path Resolution

The app automatically finds a writable location for the SQLite database:

1. **Source directory** — preferred (works in Docker, local dev)
2. **`PRAGYAM_DB_PATH` env var** — for persistent volumes on cloud platforms
3. **`/tmp/pragyam_bot.db`** — fallback (works on Render and similar read-only source mounts)

> **Note:** On Render, the source directory (`/mount/src/`) is read-only. The app auto-detects this and falls back to `/tmp/`. For persistent data across deploys, attach a persistent disk and set `PRAGYAM_DB_PATH=/var/data/pragyam_bot.db`.

---

## File Structure

```
Bot/
├── app.py                 # Entry point — starts bot + dashboard
├── bot.py                 # Telegram bot handlers
├── dashboard.py           # Streamlit admin dashboard
├── db.py                  # SQLite database layer
├── engine.py              # Headless portfolio generation pipeline
├── backdata.py            # Market data & indicators
├── strategies.py          # Trading strategies library
├── backtest_engine.py     # Walk-forward backtest engine
├── strategy_selection.py  # Dynamic strategy selection
├── portfolio_image.py     # Portfolio image generator
├── charts.py              # Chart utilities
├── symbols.txt            # ETF/stock universe (30 symbols)
├── requirements.txt       # Python dependencies
├── run.sh                 # Shell launcher
├── .env                   # Environment variables (not committed)
└── README.md              # This file
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `readonly database` | Source dir is read-only | Set `PRAGYAM_DB_PATH` to a writable path or let the app auto-fallback to `/tmp/` |
| `Bot lock already acquired` | Stale lock file from a crash | Delete `bot.lock` — the new version handles this automatically |
| `No error handlers registered` | Missing error handler | Fixed in v4.0 — error handler is now registered |
| Bot not responding | Token not set | Check `.env` has `TELEGRAM_BOT_TOKEN` |
| Dashboard shows "Bot not running" | PID file missing/stale | Restart with `python app.py` |
| Health check 503 | Dashboard not ready in time | Increase health check timeout to 120s |

---

## Version History

- **v4.0.0** — Architecture rewrite: single entry point, thread-based bot, WAL mode DB, auto-writable path detection, error handler, clean client handover
- **v3.2.0** — Dashboard improvements, bot control panel
- **v3.0.0** — Initial Telegram bot + dashboard integration

---

*Hemrek Capital — Quantitative Portfolio Intelligence*
