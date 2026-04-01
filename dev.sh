#!/usr/bin/env bash
# Full rebuild + restart in one command.
# Usage:
#   bash dev.sh                     # rebuild frontend + restart server
#   bash dev.sh --data              # also rebuild transform + DuckDB
#   bash dev.sh --data --dev 5000   # rebuild data with N variants
#   bash dev.sh --probe probe_v12   # specify probe version
set -euo pipefail
cd "$(dirname "$0")"

export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

PROBE="${PROBE:-probe_v12}"
DEV=""
BUILD_DATA=false
BUILD_FLAGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --data) BUILD_DATA=true; shift ;;
    --dev) DEV="--dev $2"; shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --umap) BUILD_FLAGS="$BUILD_FLAGS --umap"; shift ;;
    --neighbors) BUILD_FLAGS="$BUILD_FLAGS --neighbors"; shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if $BUILD_DATA; then
  echo "==> Transform (probe=$PROBE $DEV)..."
  uv run vv transform --probe "$PROBE" $DEV
  echo "==> Build DuckDB $BUILD_FLAGS..."
  uv run vv build --probe "$PROBE" $BUILD_FLAGS
fi

echo "==> Building frontend..."
cd frontend && bun run build && cd ..

# Kill existing server
fuser -k 8501/tcp 2>/dev/null || true
sleep 1

echo "==> Starting server on :8501..."
uv run vv serve --port 8501 &
SERVER_PID=$!
sleep 2

echo "==> Creating tunnel..."
source /mnt/polished-lake/scripts/config.sh
$SCRIPTS_DIR/bin/tunnel-url 8501

echo "==> Server PID: $SERVER_PID"
echo "==> Done. Ctrl+C to stop."
wait $SERVER_PID
