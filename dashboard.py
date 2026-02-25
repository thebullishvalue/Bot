"""
PRAGYAM Bot — Admin Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Streamlit dashboard showing real-time bot activity, user stats,
terminal output, and analytics for the Pragyam distribution system.
"""

import streamlit as st
import pandas as pd
import time
import os
import sys
import html as html_mod
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from db import (
    get_all_users, get_all_requests, get_recent_logs,
    get_dashboard_stats, get_user_requests, init_db, DB_PATH
)

# Initialize database (idempotent — only runs once per process)
init_db()

# ─── Page Config ───
st.set_page_config(
    page_title="Pragyam Bot — Admin Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary: #FFC300;
        --bg: #0F0F0F;
        --card: #1A1A1A;
        --border: #2A2A2A;
        --text: #EAEAEA;
        --muted: #888888;
        --green: #10b981;
        --red: #ef4444;
        --cyan: #06b6d4;
    }
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: var(--bg); }
    [data-testid="stSidebar"] { background-color: #141414; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .block-container { padding-top: 2rem; max-width: 95%; }
    
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card h4 {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 8px 0 0 0;
        font-weight: 600;
    }
    .metric-card.gold h2 { color: var(--primary); }
    .metric-card.green h2 { color: var(--green); }
    .metric-card.red h2 { color: var(--red); }
    .metric-card.cyan h2 { color: var(--cyan); }
    .metric-card.white h2 { color: var(--text); }
    
    .terminal-box {
        background: #0a0a0a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #b0b0b0;
        max-height: 600px;
        overflow-y: auto;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .terminal-box .log-info { color: #06b6d4; }
    .terminal-box .log-warn { color: #f59e0b; }
    .terminal-box .log-error { color: #ef4444; }
    .terminal-box .log-time { color: #666; }
    .terminal-box .log-user { color: #FFC300; }
    
    .section-title {
        color: var(--primary);
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.5rem 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .status-success { background: rgba(16,185,129,0.2); color: #10b981; }
    .status-error { background: rgba(239,68,68,0.2); color: #ef4444; }
    .status-running { background: rgba(255,195,0,0.2); color: #FFC300; }
    .status-pending { background: rgba(136,136,136,0.2); color: #888; }
    
    .header-banner {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 1.5rem;
    }
    .header-banner h1 {
        color: var(--primary);
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .header-banner p {
        color: var(--muted);
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }
    
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _mc(value, label, cls="white"):
    return f'<div class="metric-card {cls}"><h2>{value}</h2><h4>{label}</h4></div>'


def _status_badge(status):
    cls = {'success': 'status-success', 'error': 'status-error', 'running': 'status-running'}.get(status, 'status-pending')
    return f'<span class="status-badge {cls}">{status}</span>'


def _get_bot_thread_status() -> tuple:
    """Check if bot thread is alive by inspecting the db module's stored thread ref."""
    try:
        import db as _db_mod
        thread = getattr(_db_mod, '_bot_thread', None)
        if thread is not None and thread.is_alive():
            return True, f"Bot thread alive ({thread.name})"
        elif thread is not None:
            return False, "Bot thread died"
        else:
            return False, "Bot thread not started"
    except Exception:
        return False, "Unknown"


def _get_engine_status() -> dict:
    """Get engine pool stats if available."""
    try:
        from bot import ENGINE_POOL, MAX_ENGINE_WORKERS, _active_jobs, _get_queue_depth
        queue = _get_queue_depth()
        # Estimate active workers from semaphore
        # Semaphore._value gives remaining permits
        active = MAX_ENGINE_WORKERS - _active_jobs._value
        return {
            'max_workers': MAX_ENGINE_WORKERS,
            'active': max(0, active),
            'queued': queue,
        }
    except Exception:
        return {'max_workers': 0, 'active': 0, 'queued': 0}


def format_log_html(logs):
    """Format DB logs into terminal-style HTML."""
    lines = []
    for log in reversed(logs):
        ts = html_mod.escape(str(log.get('timestamp', ''))[:19])
        level = html_mod.escape(str(log.get('level', 'INFO')))
        src = html_mod.escape(str(log.get('source', '')))
        msg = html_mod.escape(str(log.get('message', '')))
        uid = log.get('user_id', '')

        level_cls = {'INFO': 'log-info', 'WARNING': 'log-warn', 'ERROR': 'log-error'}.get(level, 'log-info')

        uid_part = f' <span class="log-user">[uid:{uid}]</span>' if uid else ''
        line = f'<span class="log-time">{ts}</span> <span class="{level_cls}">[{level:>7}]</span> <span style="color:#777">[{src}]</span>{uid_part} {msg}'
        lines.append(line)
    return "\n".join(lines[-200:])


def read_bot_log(n_lines=300) -> str:
    """Read the bot.log file for terminal display."""
    log_path = os.path.join(SCRIPT_DIR, "bot.log")
    if not os.path.exists(log_path):
        return "<span style='color:#888;'>No bot.log file found. The bot writes logs here once it starts processing requests.</span>"
    try:
        with open(log_path, 'r', errors='replace') as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as e:
        return f"<span style='color:#ef4444;'>Error reading bot.log: {e}</span>"


def colorize_log_lines(raw_text: str) -> str:
    """HTML-escape raw log text, then add color spans for known patterns."""
    lines = raw_text.split('\n')
    colored = []
    for line in lines:
        # CRITICAL: HTML-escape FIRST to prevent broken rendering
        safe_line = html_mod.escape(line)

        # Then apply color classes based on content patterns
        if 'ERROR' in line or 'error' in line.lower():
            colored.append(f'<span style="color: #ef4444;">{safe_line}</span>')
        elif 'WARNING' in line:
            colored.append(f'<span style="color: #f59e0b;">{safe_line}</span>')
        elif 'COMPLETE' in line or 'SUCCESS' in line or 'delivered' in line:
            colored.append(f'<span style="color: #10b981;">{safe_line}</span>')
        elif 'started bot' in line.lower() or 'User started' in line:
            colored.append(f'<span style="color: #FFC300;">{safe_line}</span>')
        elif 'PHASE' in line or 'polling' in line.lower():
            colored.append(f'<span style="color: #06b6d4;">{safe_line}</span>')
        elif 'Bot thread' in line or 'Telegram Bot' in line:
            colored.append(f'<span style="color: #10b981;">{safe_line}</span>')
        else:
            colored.append(safe_line)
    return "\n".join(colored)


# ═══════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #FFC300;">PRAGYAM</div>
        <div style="color: #888; font-size: 0.7rem; margin-top: 2px;">BOT ADMIN DASHBOARD</div>
        <div style="color: #555; font-size: 0.65rem; margin-top: 2px;">Hemrek Capital</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "👥 Users", "📋 Requests", "🖥️ Terminal", "📈 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if st.button("🔄 Refresh Now", width="stretch"):
        st.rerun()

    st.markdown("---")

    # ─── Bot & Engine Status ───
    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

    bot_alive, bot_status_msg = _get_bot_thread_status()
    if bot_alive:
        st.success(f"🤖 {bot_status_msg}")
    else:
        st.warning(f"🤖 {bot_status_msg}")

    engine_info = _get_engine_status()
    if engine_info['max_workers'] > 0:
        active = engine_info['active']
        queued = engine_info['queued']
        total = engine_info['max_workers']
        st.markdown(
            f'<div style="color: #aaa; font-size: 0.75rem; margin-top: 4px;">'
            f'⚙️ Engine: <b style="color: #FFC300;">{active}/{total}</b> active'
            f'{f" · <b style=&quot;color: #f59e0b;&quot;>{queued}</b> queued" if queued > 0 else ""}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.caption("Bot is managed by app.py — restart the app to restart the bot.")

    st.markdown(
        f'<div style="color: #444; font-size: 0.6rem; margin-top: 8px;">DB: {DB_PATH}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        f'<div style="color: #555; font-size: 0.65rem; text-align: center;">'
        f'v4.1.0 | {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════
# PAGES
# ═══════════════════════════════════════

if page == "📊 Overview":
    st.markdown("""
    <div class="header-banner">
        <h1>📊 Dashboard Overview</h1>
        <p>Real-time monitoring of the Pragyam distribution bot</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_dashboard_stats()

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(_mc(stats['total_users'], "Total Users", "gold"), unsafe_allow_html=True)
    with c2: st.markdown(_mc(stats['total_requests'], "Total Requests", "white"), unsafe_allow_html=True)
    with c3: st.markdown(_mc(f"{stats['success_rate']:.0f}%", "Success Rate", "green"), unsafe_allow_html=True)
    with c4: st.markdown(_mc(stats['today_requests'], "Today's Requests", "cyan"), unsafe_allow_html=True)
    with c5: st.markdown(_mc(f"{stats['avg_duration_seconds']:.0f}s", "Avg Duration", "white"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_mc(stats['success_requests'], "Successful", "green"), unsafe_allow_html=True)
    with c2: st.markdown(_mc(stats['error_requests'], "Failed", "red"), unsafe_allow_html=True)
    with c3: st.markdown(_mc(stats['running_requests'], "Running", "gold"), unsafe_allow_html=True)
    with c4: st.markdown(_mc(stats['today_users'], "Active Today", "cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Capital stats
    st.markdown('<div class="section-title">💰 Capital Analytics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_mc(f"₹{stats['total_capital_processed']:,.0f}", "Total Processed", "gold"), unsafe_allow_html=True)
    with c2: st.markdown(_mc(f"₹{stats['avg_capital']:,.0f}", "Avg Capital", "white"), unsafe_allow_html=True)
    with c3: st.markdown(_mc(f"₹{stats['min_capital']:,.0f}", "Min Capital", "white"), unsafe_allow_html=True)
    with c4: st.markdown(_mc(f"₹{stats['max_capital']:,.0f}", "Max Capital", "white"), unsafe_allow_html=True)

    # Style breakdown + Recent Activity
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">📈 Style Breakdown</div>', unsafe_allow_html=True)
        breakdown = stats.get('style_breakdown', {})
        if breakdown:
            df_style = pd.DataFrame([
                {'Style': k, 'Requests': v} for k, v in breakdown.items() if k
            ])
            if not df_style.empty:
                st.bar_chart(df_style.set_index('Style'))
            else:
                st.info("No requests yet")
        else:
            st.info("No requests yet")

    with c2:
        st.markdown('<div class="section-title">📋 Recent Activity</div>', unsafe_allow_html=True)
        recent = get_all_requests(limit=8)
        if recent:
            for req in recent:
                status_html = _status_badge(req.get('status', 'unknown'))
                name = req.get('first_name', '') or req.get('username', '') or f"User {req.get('user_id', '?')}"
                style = req.get('investment_style', 'N/A')
                cap = req.get('capital', 0) or 0
                ts = req.get('timestamp', '')[:16]
                st.markdown(
                    f'{status_html} <span style="color: #eee; font-size: 0.85rem;">'
                    f'<b>{html_mod.escape(str(name))}</b> — {html_mod.escape(str(style))} — ₹{cap:,.0f} — '
                    f'<span style="color: #666">{ts}</span></span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No requests yet")


elif page == "👥 Users":
    st.markdown("""
    <div class="header-banner">
        <h1>👥 User Management</h1>
        <p>All registered users of the Pragyam Telegram bot</p>
    </div>
    """, unsafe_allow_html=True)

    users = get_all_users()
    if users:
        df = pd.DataFrame(users)
        df['first_seen'] = pd.to_datetime(df['first_seen']).dt.strftime('%Y-%m-%d %H:%M')
        df['last_active'] = pd.to_datetime(df['last_active']).dt.strftime('%Y-%m-%d %H:%M')

        display_cols = ['user_id', 'username', 'first_name', 'last_name', 'total_requests', 'first_seen', 'last_active']
        available_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[available_cols], width="stretch", hide_index=True)
        st.caption(f"Total: {len(users)} users")
    else:
        st.info("No users registered yet. Start the bot and interact with it.")


elif page == "📋 Requests":
    st.markdown("""
    <div class="header-banner">
        <h1>📋 Request Log</h1>
        <p>All portfolio generation requests</p>
    </div>
    """, unsafe_allow_html=True)

    limit = st.selectbox("Show last", [25, 50, 100, 200], index=1)
    requests_data = get_all_requests(limit=limit)

    if requests_data:
        df = pd.DataFrame(requests_data)

        # Format columns
        if 'capital' in df.columns:
            df['capital_fmt'] = df['capital'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
        if 'total_value' in df.columns:
            df['value_fmt'] = df['total_value'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
        if 'duration_seconds' in df.columns:
            df['duration'] = df['duration_seconds'].apply(lambda x: f"{x:.0f}s" if pd.notna(x) else "—")
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%m-%d %H:%M')

        display_cols = ['time', 'first_name', 'investment_style', 'capital_fmt', 'status',
                        'positions', 'value_fmt', 'regime', 'selection_mode', 'duration']
        available = [c for c in display_cols if c in df.columns]

        st.dataframe(df[available], width="stretch", hide_index=True)

        # Summary stats
        c1, c2, c3 = st.columns(3)
        with c1:
            success = (df['status'] == 'success').sum()
            st.metric("Successful", success)
        with c2:
            errors = (df['status'] == 'error').sum()
            st.metric("Failed", errors)
        with c3:
            avg_d = df['duration_seconds'].dropna().mean()
            st.metric("Avg Duration", f"{avg_d:.0f}s" if pd.notna(avg_d) else "N/A")
    else:
        st.info("No requests yet")


elif page == "🖥️ Terminal":
    st.markdown("""
    <div class="header-banner">
        <h1>🖥️ Terminal Output</h1>
        <p>Live bot logs and engine output</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["**Bot Log File**", "**Database Logs**"])

    with tab1:
        n_lines = st.slider("Lines to show", 50, 500, 200, 50)
        log_content = read_bot_log(n_lines)

        if log_content.startswith("<span"):
            # Already HTML (error/empty message from read_bot_log)
            terminal_html = log_content
        else:
            # Raw log text — colorize with proper HTML escaping
            terminal_html = colorize_log_lines(log_content)

        st.markdown(
            f'<div class="terminal-box">{terminal_html}</div>',
            unsafe_allow_html=True
        )

    with tab2:
        db_logs = get_recent_logs(limit=200)
        if db_logs:
            log_html = format_log_html(db_logs)
            st.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No database logs yet. Bot activity (start events, portfolio requests, errors) appears here.")


elif page == "📈 Analytics":
    st.markdown("""
    <div class="header-banner">
        <h1>📈 Analytics</h1>
        <p>Usage patterns and distribution metrics</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_dashboard_stats()
    requests_data = get_all_requests(limit=500)

    if requests_data:
        df = pd.DataFrame(requests_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">📅 Requests Over Time</div>', unsafe_allow_html=True)
            df['date'] = df['timestamp'].dt.date
            daily = df.groupby('date').size().reset_index(name='requests')
            st.line_chart(daily.set_index('date'))

        with c2:
            st.markdown('<div class="section-title">⏰ Hourly Distribution</div>', unsafe_allow_html=True)
            df['hour'] = df['timestamp'].dt.hour
            hourly = df.groupby('hour').size().reset_index(name='requests')
            st.bar_chart(hourly.set_index('hour'))

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">💰 Capital Distribution</div>', unsafe_allow_html=True)
            valid_cap = df[df['capital'].notna() & (df['capital'] > 0)]['capital']
            if not valid_cap.empty:
                bins = [0, 100000, 500000, 1000000, 5000000, 10000000, 100000000]
                labels = ['<1L', '1-5L', '5-10L', '10-50L', '50L-1Cr', '>1Cr']
                df['cap_bucket'] = pd.cut(df['capital'], bins=bins, labels=labels, include_lowest=True)
                cap_dist = df['cap_bucket'].value_counts().sort_index()
                st.bar_chart(cap_dist)

        with c2:
            st.markdown('<div class="section-title">⏱️ Duration Distribution</div>', unsafe_allow_html=True)
            valid_dur = df[df['duration_seconds'].notna()]['duration_seconds']
            if not valid_dur.empty:
                st.bar_chart(valid_dur.value_counts(bins=10).sort_index())

        # Regime distribution
        st.markdown('<div class="section-title">🧠 Market Regime Distribution</div>', unsafe_allow_html=True)
        regime_counts = df[df['regime'].notna()]['regime'].value_counts()
        if not regime_counts.empty:
            st.bar_chart(regime_counts)
    else:
        st.info("No data for analytics yet. Generate some portfolios first!")


# Auto-refresh
if auto_refresh:
    time.sleep(10)
    st.rerun()
