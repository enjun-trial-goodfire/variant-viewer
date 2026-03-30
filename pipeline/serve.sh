#!/bin/bash
#SBATCH --job-name=variant-viewer
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/variant_viewer_%j.out
#
# Build to /tmp + serve from there. Single job — no rsync needed.
# Usage: sbatch webapp/serve.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/mnt/polished-lake/scripts}"
PORT="${1:-8501}"
mkdir -p outputs

# Build to /tmp (fast)
uv run --frozen vv build
BUILD_DIR=$(grep '^STAGING_DIR=' "outputs/variant_viewer_${SLURM_JOB_ID}.out" 2>/dev/null | tail -1 | cut -d= -f2) || true

if [ -z "${BUILD_DIR:-}" ] || [ ! -d "${BUILD_DIR:-}" ]; then
    echo "ERROR: build failed, no staging dir found in output"
    exit 1
fi

# Serve + tunnel
cd "$BUILD_DIR"
echo "Serving $BUILD_DIR on port $PORT"
python3 -m http.server "$PORT" &
sleep 2
if ! kill -0 $! 2>/dev/null; then
    echo "ERROR: http.server failed to start (port $PORT in use?)"
    exit 1
fi
"$SCRIPTS_DIR/bin/tunnel-url" "$PORT"
echo "scancel $SLURM_JOB_ID to stop."
wait
