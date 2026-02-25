"""
PRAGYAM Bot — Admin Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Streamlit dashboard for monitoring the Pragyam distribution system.

Usage:
    Called as: from dashboard import render; render(log_file_path=...)
    Or standalone: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import time
import os
import sys
import html as html_mod
import threading
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from db import (
    get_all_users, get_all_requests, get_recent_logs,
    get_dashboard_stats, init_db, DB_PATH
)


# ═══════════════════════════════════════
# CSS
# ═══════════════════════════════════════

_CSS = """
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
    .metric-card h2 { font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.2; }
    .metric-card h4 {
        color: var(--muted); font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 1px; margin: 8px 0 0 0; font-weight: 600;
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
        color: var(--primary); font-size: 1.1rem; font-weight: 700;
        margin: 1.5rem 0 0.5rem 0; text-transform: uppercase; letter-spacing: 1px;
    }
    
    .status-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    }
    .status-success { background: rgba(16,185,129,0.2); color: #10b981; }
    .status-error { background: rgba(239,68,68,0.2); color: #ef4444; }
    .status-running { background: rgba(255,195,0,0.2); color: #FFC300; }
    .status-pending { background: rgba(136,136,136,0.2); color: #888; }
    
    .header-banner {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        border: 1px solid var(--border); border-left: 4px solid var(--primary);
        border-radius: 8px; padding: 20px 24px; margin-bottom: 1.5rem;
    }
    .header-banner h1 { color: var(--primary); font-size: 1.6rem; font-weight: 800; margin: 0; }
    .header-banner p { color: var(--muted); font-size: 0.85rem; margin: 4px 0 0 0; }
    
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
</style>
"""


# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════

def _mc(value, label, cls="white"):
    return f'<div class="metric-card {cls}"><h2>{value}</h2><h4>{label}</h4></div>'


def _badge(status):
    cls = {'success': 'status-success', 'error': 'status-error', 'running': 'status-running'}.get(status, 'status-pending')
    return f'<span class="status-badge {cls}">{status}</span>'


def _esc(text):
    """HTML-escape a string safely."""
    return html_mod.escape(str(text)) if text else ""


def _get_bot_status():
    """Check if the bot thread is alive."""
    try:
        import db as _db
        t = getattr(_db, '_bot_thread', None)
        if t is not None and t.is_alive():
            return True, f"Running ({t.name})"
        elif t is not None:
            return False, "Thread died — restart app"
        return False, "Not started"
    except Exception:
        return False, "Unknown"


def _get_engine_info():
    """Get engine pool utilization."""
    try:
        from bot import MAX_ENGINE_WORKERS, _active_jobs, _get_queue_depth
        active = MAX_ENGINE_WORKERS - _active_jobs._value
        return max(0, active), _get_queue_depth(), MAX_ENGINE_WORKERS
    except Exception:
        return 0, 0, 0


def _read_log_file(path, n_lines=300):
    """Read and colorize log file content."""
    if not path or not os.path.exists(path):
        return '<span style="color:#666;">No log output yet. Send /start to the bot on Telegram to generate activity.</span>'
    try:
        with open(path, 'r', errors='replace') as f:
            lines = f.readlines()
        if not lines:
            return '<span style="color:#666;">Log file is empty. Waiting for bot activity...</span>'
        
        colored = []
        for line in lines[-n_lines:]:
            safe = _esc(line.rstrip())
            if not safe:
                continue
            if 'ERROR' in line:
                colored.append(f'<span style="color:#ef4444;">{safe}</span>')
            elif 'WARNING' in line:
                colored.append(f'<span style="color:#f59e0b;">{safe}</span>')
            elif 'polling' in line.lower() or 'Bot thread' in line or 'started' in line.lower():
                colored.append(f'<span style="color:#10b981;">{safe}</span>')
            elif 'Portfolio' in line or 'delivered' in line:
                colored.append(f'<span style="color:#FFC300;">{safe}</span>')
            elif 'PHASE' in line:
                colored.append(f'<span style="color:#06b6d4;">{safe}</span>')
            else:
                colored.append(safe)
        return "\n".join(colored)
    except Exception as e:
        return f'<span style="color:#ef4444;">Error reading log: {_esc(str(e))}</span>'


def _format_db_logs(logs):
    """Format database log entries as terminal HTML."""
    if not logs:
        return '<span style="color:#666;">No activity logged yet. Interact with the bot to see entries.</span>'
    lines = []
    for log in reversed(logs):
        ts = _esc(str(log.get('timestamp', ''))[:19])
        level = str(log.get('level', 'INFO'))
        src = _esc(str(log.get('source', '')))
        msg = _esc(str(log.get('message', '')))
        uid = log.get('user_id', '')
        cls = {'INFO': 'log-info', 'WARNING': 'log-warn', 'ERROR': 'log-error'}.get(level, 'log-info')
        uid_part = f' <span class="log-user">[uid:{uid}]</span>' if uid else ''
        lines.append(
            f'<span class="log-time">{ts}</span> '
            f'<span class="{cls}">[{_esc(level):>7}]</span> '
            f'<span style="color:#777">[{src}]</span>'
            f'{uid_part} {msg}'
        )
    return "\n".join(lines[-200:])


# ═══════════════════════════════════════
# RENDER (main entry point)
# ═══════════════════════════════════════

def render(log_file_path: str = None):
    """Render the complete admin dashboard.
    
    Args:
        log_file_path: Path to the bot.log file (set by app.py).
                       Falls back to SCRIPT_DIR/bot.log if not provided.
    """
    if log_file_path is None:
        log_file_path = os.path.join(SCRIPT_DIR, "bot.log")

    init_db()

    # ─── Page Config (must be first st command) ───
    st.set_page_config(
        page_title="Pragyam — Admin Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    # ─── Sidebar ───
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

        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
        if st.button("🔄 Refresh Now", key="refresh_btn"):
            st.rerun()

        st.markdown("---")

        # System status
        st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

        bot_alive, bot_msg = _get_bot_status()
        if bot_alive:
            st.success(f"🤖 Bot: {bot_msg}")
        else:
            st.warning(f"🤖 Bot: {bot_msg}")

        active, queued, total = _get_engine_info()
        if total > 0:
            status_parts = [f"⚙️ Engine: **{active}/{total}** active"]
            if queued > 0:
                status_parts.append(f"**{queued}** queued")
            st.caption(" · ".join(status_parts))

        st.markdown("---")
        st.markdown(
            f'<div style="color:#444; font-size:0.6rem;">DB: {_esc(DB_PATH)}<br>'
            f'Log: {_esc(log_file_path)}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="color:#555; font-size:0.65rem; text-align:center; margin-top:8px;">'
            f'v4.1 | {datetime.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True
        )

    # ─── Pages ───
    if page == "📊 Overview":
        _page_overview()
    elif page == "👥 Users":
        _page_users()
    elif page == "📋 Requests":
        _page_requests()
    elif page == "🖥️ Terminal":
        _page_terminal(log_file_path)
    elif page == "📈 Analytics":
        _page_analytics()

    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ═══════════════════════════════════════
# PAGES
# ═══════════════════════════════════════

def _page_overview():
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
    with c4: st.markdown(_mc(stats['today_requests'], "Today", "cyan"), unsafe_allow_html=True)
    with c5: st.markdown(_mc(f"{stats['avg_duration_seconds']:.0f}s", "Avg Duration", "white"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_mc(stats['success_requests'], "Successful", "green"), unsafe_allow_html=True)
    with c2: st.markdown(_mc(stats['error_requests'], "Failed", "red"), unsafe_allow_html=True)
    with c3: st.markdown(_mc(stats['running_requests'], "Running", "gold"), unsafe_allow_html=True)
    with c4: st.markdown(_mc(stats['today_users'], "Active Today", "cyan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Capital analytics
    st.markdown('<div class="section-title">💰 Capital Analytics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(_mc(f"₹{stats['total_capital_processed']:,.0f}", "Total Processed", "gold"), unsafe_allow_html=True)
    with c2: st.markdown(_mc(f"₹{stats['avg_capital']:,.0f}", "Avg Capital", "white"), unsafe_allow_html=True)
    with c3: st.markdown(_mc(f"₹{stats['min_capital']:,.0f}", "Min Capital", "white"), unsafe_allow_html=True)
    with c4: st.markdown(_mc(f"₹{stats['max_capital']:,.0f}", "Max Capital", "white"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">📈 Style Breakdown</div>', unsafe_allow_html=True)
        breakdown = stats.get('style_breakdown', {})
        if breakdown and any(v > 0 for v in breakdown.values()):
            df_style = pd.DataFrame([{'Style': k, 'Requests': v} for k, v in breakdown.items() if k])
            if not df_style.empty:
                st.bar_chart(df_style.set_index('Style'))
            else:
                st.info("No data yet")
        else:
            st.info("No requests yet. Send /start to the bot to generate a portfolio.")

    with c2:
        st.markdown('<div class="section-title">📋 Recent Activity</div>', unsafe_allow_html=True)
        recent = get_all_requests(limit=8)
        if recent:
            for req in recent:
                name = req.get('first_name') or req.get('username') or f"User {req.get('user_id', '?')}"
                style = req.get('investment_style', 'N/A')
                cap = req.get('capital', 0) or 0
                ts = str(req.get('timestamp', ''))[:16]
                st.markdown(
                    f'{_badge(req.get("status", "unknown"))} '
                    f'<span style="color:#eee; font-size:0.85rem;">'
                    f'<b>{_esc(name)}</b> — {_esc(style)} — ₹{cap:,.0f} — '
                    f'<span style="color:#666">{ts}</span></span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No requests yet")


def _page_users():
    st.markdown("""
    <div class="header-banner">
        <h1>👥 User Management</h1>
        <p>All registered users of the Pragyam Telegram bot</p>
    </div>
    """, unsafe_allow_html=True)

    users = get_all_users()
    if users:
        df = pd.DataFrame(users)
        for col in ['first_seen', 'last_active']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
        display = [c for c in ['user_id', 'username', 'first_name', 'last_name', 'total_requests', 'first_seen', 'last_active'] if c in df.columns]
        st.dataframe(df[display], hide_index=True)
        st.caption(f"Total: {len(users)} users")
    else:
        st.info("No users registered yet. Start the bot and send /start.")


def _page_requests():
    st.markdown("""
    <div class="header-banner">
        <h1>📋 Request Log</h1>
        <p>All portfolio generation requests</p>
    </div>
    """, unsafe_allow_html=True)

    limit = st.selectbox("Show last", [25, 50, 100, 200], index=1)
    rows = get_all_requests(limit=limit)

    if rows:
        df = pd.DataFrame(rows)
        if 'capital' in df.columns:
            df['capital_fmt'] = df['capital'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
        if 'total_value' in df.columns:
            df['value_fmt'] = df['total_value'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "—")
        if 'duration_seconds' in df.columns:
            df['duration'] = df['duration_seconds'].apply(lambda x: f"{x:.0f}s" if pd.notna(x) else "—")
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%m-%d %H:%M')

        display = [c for c in ['time', 'first_name', 'investment_style', 'capital_fmt', 'status',
                                'positions', 'value_fmt', 'regime', 'selection_mode', 'duration'] if c in df.columns]
        st.dataframe(df[display], hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Successful", int((df['status'] == 'success').sum()))
        with c2: st.metric("Failed", int((df['status'] == 'error').sum()))
        with c3:
            avg_d = df['duration_seconds'].dropna().mean()
            st.metric("Avg Duration", f"{avg_d:.0f}s" if pd.notna(avg_d) else "—")
    else:
        st.info("No requests yet")


def _page_terminal(log_file_path):
    st.markdown("""
    <div class="header-banner">
        <h1>🖥️ Terminal Output</h1>
        <p>Live bot logs and engine output</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📄 **Bot Log File**", "🗃️ **Database Activity**"])

    with tab1:
        n_lines = st.slider("Lines to show", 50, 500, 200, 50)
        html_content = _read_log_file(log_file_path, n_lines)
        st.markdown(f'<div class="terminal-box">{html_content}</div>', unsafe_allow_html=True)
        st.caption(f"Reading from: {log_file_path}")

    with tab2:
        db_logs = get_recent_logs(limit=200)
        html_content = _format_db_logs(db_logs)
        st.markdown(f'<div class="terminal-box">{html_content}</div>', unsafe_allow_html=True)


def _page_analytics():
    st.markdown("""
    <div class="header-banner">
        <h1>📈 Analytics</h1>
        <p>Usage patterns and distribution metrics</p>
    </div>
    """, unsafe_allow_html=True)

    rows = get_all_requests(limit=500)
    if not rows:
        st.info("No data for analytics yet. Generate some portfolios first!")
        return

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">📅 Requests Over Time</div>', unsafe_allow_html=True)
        daily = df.groupby(df['timestamp'].dt.date).size().reset_index(name='requests')
        daily.columns = ['date', 'requests']
        st.line_chart(daily.set_index('date'))

    with c2:
        st.markdown('<div class="section-title">⏰ Hourly Distribution</div>', unsafe_allow_html=True)
        hourly = df.groupby(df['timestamp'].dt.hour).size().reset_index(name='requests')
        hourly.columns = ['hour', 'requests']
        st.bar_chart(hourly.set_index('hour'))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">💰 Capital Distribution</div>', unsafe_allow_html=True)
        valid_cap = df[df['capital'].notna() & (df['capital'] > 0)]['capital']
        if not valid_cap.empty:
            bins = [0, 100000, 500000, 1000000, 5000000, 10000000, 100000000]
            labels = ['<1L', '1-5L', '5-10L', '10-50L', '50L-1Cr', '>1Cr']
            df['cap_bucket'] = pd.cut(df['capital'], bins=bins, labels=labels, include_lowest=True)
            st.bar_chart(df['cap_bucket'].value_counts().sort_index())

    with c2:
        st.markdown('<div class="section-title">⏱️ Duration Distribution</div>', unsafe_allow_html=True)
        valid_dur = df[df['duration_seconds'].notna()]['duration_seconds']
        if not valid_dur.empty:
            st.bar_chart(valid_dur.value_counts(bins=10).sort_index())

    st.markdown('<div class="section-title">🧠 Market Regime Distribution</div>', unsafe_allow_html=True)
    regime_counts = df[df['regime'].notna()]['regime'].value_counts()
    if not regime_counts.empty:
        st.bar_chart(regime_counts)


# ─── Standalone mode (streamlit run dashboard.py) ───
if __name__ == "__main__":
    render()
