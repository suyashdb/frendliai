#!/usr/bin/env bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}=== FriendliAI Reasoning Gateway – Demo ===${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $MOCK_PID $GATEWAY_PID 2>/dev/null || true
    wait $MOCK_PID $GATEWAY_PID 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Install deps if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt --quiet
fi

# Start mock upstream
echo -e "${YELLOW}Starting mock upstream on :9000...${NC}"
python mock_upstream.py &
MOCK_PID=$!
sleep 1

# Verify mock is up
if ! curl -sf http://localhost:9000/health > /dev/null; then
    echo -e "${RED}Mock upstream failed to start.${NC}"
    exit 1
fi
echo -e "${GREEN}Mock upstream running (PID $MOCK_PID)${NC}"

# Start gateway
echo -e "${YELLOW}Starting gateway on :8000...${NC}"
GW_UPSTREAM_BASE_URL="http://localhost:9000/v1" \
GW_SUMMARISER_BASE_URL="http://localhost:9000/v1" \
GW_SUMMARISER_MODEL="gpt-4o-mini" \
GW_LOG_LEVEL="WARNING" \
python -m gateway.server &
GATEWAY_PID=$!
sleep 2

# Verify gateway is up
if ! curl -sf http://localhost:8000/health > /dev/null; then
    echo -e "${RED}Gateway failed to start.${NC}"
    exit 1
fi
echo -e "${GREEN}Gateway running (PID $GATEWAY_PID)${NC}"
echo ""

# Run client
echo -e "${GREEN}=== Running client ===${NC}"
echo ""
python client.py \
    --gateway http://localhost:8000 \
    --model deepseek-r1 \
    --prompt "What is the capital of France?"

echo ""
echo -e "${GREEN}=== Demo complete ===${NC}"
