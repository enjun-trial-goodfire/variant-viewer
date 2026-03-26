#!/bin/bash
#SBATCH --job-name=variant-viewer
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/variant_viewer_%j.out
#
# Build to /tmp + serve from there. Single job — no rsync needed.
# Usage: sbatch webapp/serve.sh

set -e
cd "${SLURM_SUBMIT_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/mnt/polished-lake/scripts}"
PORT="${1:-8501}"

# Build to /tmp (fast)
uv run python webapp/build.py --no-sync
BUILD_DIR=$(grep '^STAGING_DIR=' "outputs/variant_viewer_${SLURM_JOB_ID}.out" | tail -1 | cut -d= -f2)

if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: build failed, no staging dir"
    exit 1
fi

# Serve + tunnel
cd "$BUILD_DIR"
echo "Serving $BUILD_DIR on port $PORT"
python3 -m http.server "$PORT" &
sleep 2
"$SCRIPTS_DIR/bin/tunnel-url" "$PORT"
echo "scancel $SLURM_JOB_ID to stop."
wait
