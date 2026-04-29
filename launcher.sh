#!/bin/bash
#
# loragrad launcher — runs once on container start, then keeps the service
# alive with a heartbeat so Railway treats the deploy as healthy and the
# logs stay reachable.
#
# Behavior:
#   - First boot (no /data/runs/routed.bin): train both modes, save to volume.
#   - Subsequent boots: print existing report + samples, skip retraining.

set -e

cd /workspace

mkdir -p /data/runs

# The trainer writes to ./runs/. Symlink so artifacts land on the volume.
rm -rf runs
ln -s /data/runs runs

STEPS="${LORAGRAD_STEPS:-10000}"
SEED="${LORAGRAD_SEED:-42}"

echo "════════════════════════════════════════════════════════"
echo "  loragrad on Railway"
echo "  steps=${STEPS}  seed=${SEED}  uname=$(uname -a)"
echo "  cpus=$(nproc)  mem=$(awk '/MemTotal/ {print $2 " " $3}' /proc/meminfo 2>/dev/null)"
echo "════════════════════════════════════════════════════════"

if [ ! -f /data/runs/routed.bin ]; then
    echo
    echo "── First run — training both modes ──"
    ./examples/train_loragrad --routed  --steps="${STEPS}" --seed="${SEED}" --name=routed
    ./examples/train_loragrad --control --steps="${STEPS}" --seed="${SEED}" --name=control
    echo
    echo "── Training complete. Saved to /data/runs/. ──"
else
    echo
    echo "── Already trained on this volume. Skipping training. ──"
    ls -la /data/runs/
fi

echo
echo "════════════════════════════════════════════════════════"
echo "  Heartbeat loop. Service stays alive; ^C to stop."
echo "════════════════════════════════════════════════════════"

while true; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] heartbeat ($(uptime -p 2>/dev/null || uptime))"
    sleep 600
done
