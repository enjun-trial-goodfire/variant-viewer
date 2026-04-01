#!/usr/bin/env bash
# Rebuild frontend + restart server in one command.
# Usage: bash dev.sh [--build-data]
set -euo pipefail
cd "$(dirname "$0")"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# Optionally rebuild data
if [[ "${1:-}" == "--build-data" ]]; then
    echo "==> Building DuckDB..."
    uv run vv build --dev 5000 --probe probe_v10
fi

# Rebuild frontend
echo "==> Building frontend..."
cd frontend && bun run build && cd ..

# Kill any existing server
pkill -f "uv run vv serve" 2>/dev/null || true
pkill -f "uvicorn.*8501" 2>/dev/null || true
sleep 1

# Start server
echo "==> Starting server on :8501..."
uv run vv serve --port 8501 &
SERVER_PID=$!
sleep 2

# Tunnel
echo "==> Creating tunnel..."
source /mnt/polished-lake/scripts/config.sh
$SCRIPTS_DIR/bin/tunnel-url 8501

echo "==> Server PID: $SERVER_PID"
echo "==> Done. Ctrl+C to stop."
wait $SERVER_PID
