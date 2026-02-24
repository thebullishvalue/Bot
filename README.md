# PRAGYAM Telegram Bot
### Portfolio Intelligence Distribution System | Hemrek Capital

---

## Overview

A professional Telegram bot that serves as the distribution channel for the **Pragyam** portfolio curation engine. Users interact with the bot to receive regime-aware, walk-forward curated portfolios — delivered as institutional-grade images.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    PRAGYAM BOT SYSTEM                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐ │
│  │  Telegram    │───▶│   Bot.py    │───▶│  Engine.py   │ │
│  │  Users       │    │  (Handler)  │    │  (Pipeline)  │ │
│  └─────────────┘    └─────────────┘    └──────────────┘ │
│         ▲                  │                    │        │
│         │           ┌──────▼──────┐    ┌───────▼──────┐ │
│         │           │   DB.py     │    │ Strategies   │ │
│         │           │  (SQLite)   │    │ Backdata     │ │
│         │           └──────┬──────┘    │ Backtest     │ │
│         │                  │           └──────────────┘ │
│         │           ┌──────▼──────┐                     │
│  ┌──────┴────┐      │ Dashboard   │                     │
│  │ Portfolio  │      │ (Streamlit) │                     │
│  │ Image Gen  │      └─────────────┘                     │
│  └───────────┘                                           │
└──────────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `bot.py` | Telegram bot — user interaction, conversation flow |
| `engine.py` | Headless Pragyam pipeline — 4-phase portfolio generation |
| `portfolio_image.py` | Professional portfolio image renderer (PIL) |
| `dashboard.py` | Streamlit admin dashboard — users, logs, analytics |
| `db.py` | SQLite database — user tracking, request logging |
| `run.sh` | Launch script for bot + dashboard |
| `strategies.py` | 90+ quantitative strategies (from Pragyam) |
| `backdata.py` | Market data fetching & indicator calculation |
| `backtest_engine.py` | Walk-forward backtesting engine |
| `strategy_selection.py` | Trigger-based strategy selection |
| `symbols.txt` | 30 ETF/sector instrument universe |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start both bot + dashboard
chmod +x run.sh
./run.sh both

# Or start individually
./run.sh bot         # Telegram bot only
./run.sh dashboard   # Admin dashboard only
./run.sh stop        # Stop all services
```

## User Flow (Telegram)

```
/start  →  Welcome message
/portfolio  →  Select Style (Swing/SIP)
           →  Enter Capital (presets or custom)
           →  Confirm parameters
           →  ⏳ Engine runs (3-5 min)
           →  📸 Portfolio image delivered
           →  📋 Top 5 holdings summary
```

## Admin Dashboard

Access at `http://localhost:8501` after starting:

- **Overview** — Key metrics, capital analytics, recent activity
- **Users** — All registered bot users with activity stats
- **Requests** — Full request log with status, timing, regime info
- **Terminal** — Live bot logs with color-coded output
- **Analytics** — Usage patterns, capital distribution, hourly activity

## Engine Pipeline

The 4-phase pipeline mirrors the full Pragyam system:

1. **Data Fetching** — Downloads price data for 30 instruments via yfinance
2. **Strategy Selection** — Backtests 90+ strategies with trigger-based methodology, selects top 4
3. **Walk-Forward Evaluation** — Pure walk-forward curation quality assessment
4. **Portfolio Curation** — Final weighted portfolio with position sizing

## Configuration

- **Bot Token**: Set in `bot.py` (line: `TOKEN = ...`)
- **Capital Range**: ₹10,000 to ₹10,00,00,000
- **Instruments**: Edit `symbols.txt`
- **Max Positions**: 30 (hardcoded in engine)
- **Walk-Forward Window**: 50 days

---

*Hemrek Capital © 2025 | Pragyam v3.2.0*
