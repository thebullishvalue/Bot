#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRAGYAM Bot — Launch Script
# Hemrek Capital Portfolio Intelligence Distribution System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GOLD='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRAGYAM — Portfolio Intelligence Bot"
echo "  Hemrek Capital Distribution System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

case "${1:-both}" in
    bot)
        echo -e "${CYAN}Starting Telegram Bot only...${NC}"
        python3 bot.py
        ;;
    dashboard)
        echo -e "${CYAN}Starting Admin Dashboard only...${NC}"
        streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --theme.base dark
        ;;
    both)
        echo -e "${GREEN}Starting Telegram Bot...${NC}"
        python3 bot.py &
        BOT_PID=$!
        echo $BOT_PID > bot.pid
        echo -e "${GREEN}Bot started (PID: $BOT_PID)${NC}"
        
        sleep 2
        
        echo -e "${GREEN}Starting Admin Dashboard...${NC}"
        streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --theme.base dark &
        DASH_PID=$!
        echo -e "${GREEN}Dashboard started (PID: $DASH_PID)${NC}"
        
        echo ""
        echo -e "${GOLD}━━━ ALL SYSTEMS RUNNING ━━━${NC}"
        echo -e "  Bot PID:       ${CYAN}$BOT_PID${NC}"
        echo -e "  Dashboard PID: ${CYAN}$DASH_PID${NC}"
        echo -e "  Dashboard URL: ${CYAN}http://localhost:8501${NC}"
        echo ""
        echo -e "${GOLD}Press Ctrl+C to stop all services${NC}"
        
        trap "echo -e '\n${RED}Shutting down...${NC}'; kill $BOT_PID $DASH_PID 2>/dev/null; rm -f bot.pid; exit 0" SIGINT SIGTERM
        
        wait
        ;;
    stop)
        echo -e "${RED}Stopping all services...${NC}"
        if [ -f bot.pid ]; then
            kill $(cat bot.pid) 2>/dev/null
            rm -f bot.pid
            echo "Bot stopped"
        fi
        pkill -f "streamlit run dashboard.py" 2>/dev/null
        echo "Dashboard stopped"
        echo -e "${GREEN}All services stopped.${NC}"
        ;;
    *)
        echo "Usage: $0 {bot|dashboard|both|stop}"
        echo ""
        echo "  bot        — Start Telegram bot only"
        echo "  dashboard  — Start admin dashboard only"  
        echo "  both       — Start bot + dashboard (default)"
        echo "  stop       — Stop all running services"
        ;;
esac
