#!/usr/bin/env bash
# ablation_worker.sh <gpu_uuid> <queue_file>
# Pops cells (abl:cam) from the shared queue_file (flock-atomic) and runs each via
# ablation_run.sh until the queue is empty. Launch one worker per free GPU; all share the queue.
set -uo pipefail
cd "$(dirname "$0")"
GPU="${1:?gpu}"; Q="${2:?queue file}"
mkdir -p ablation_logs
pop() {
  exec 200>"$Q.lock"; flock -x 200
  local line; line=$(head -n1 "$Q" 2>/dev/null)
  [ -n "$line" ] && sed -i '1d' "$Q"
  flock -u 200; printf '%s' "$line"
}
while :; do
  CELL=$(pop); [ -z "$CELL" ] && break
  IFS=':' read -r abl cam <<< "$CELL"
  echo "[$(date +%H:%M:%S)] worker $GPU -> $abl:$cam" >> ablation_logs/run.log
  bash ablation_run.sh "$abl" "$cam" "$GPU"
done
echo "[$(date +%H:%M:%S)] worker ${GPU:0:12} drained" >> ablation_logs/run.log
