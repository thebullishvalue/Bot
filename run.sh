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
NC='\033[0m'

echo -e "${GOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRAGYAM — Portfolio Intelligence Bot"
echo "  Hemrek Capital Distribution System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

echo -e "${GREEN}Starting Pragyam (bot + dashboard)...${NC}"
echo ""

# Single entry point — app.py manages both bot (thread) and dashboard (subprocess)
python3 app.py
