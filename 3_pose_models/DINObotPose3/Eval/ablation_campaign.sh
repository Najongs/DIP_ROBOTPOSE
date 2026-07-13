#!/usr/bin/env bash
# Job scheduler: run a queue of ablation cells across a GPU pool (one cell per GPU at a time,
# refilling as each finishes). Usage:
#   ablation_campaign.sh "<gpu1,gpu2,...>" abl:cam abl:cam ...
set -uo pipefail
cd "$(dirname "$0")"
IFS=',' read -ra GPUS <<< "${1:?gpu csv}"; shift
CELLS=("$@")
declare -A PID
i=0
mkdir -p ablation_logs
echo "[$(date +%H:%M:%S)] CAMPAIGN start: ${#CELLS[@]} cells on ${#GPUS[@]} gpus" >> ablation_logs/run.log
while (( i < ${#CELLS[@]} )) || (( ${#PID[@]} > 0 )); do
  for g in "${GPUS[@]}"; do
    if [[ -n "${PID[$g]:-}" ]] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    unset 'PID[$g]' 2>/dev/null || true
    if (( i < ${#CELLS[@]} )); then
      IFS=':' read -r abl cam <<< "${CELLS[$i]}"; ((i++))
      setsid bash ablation_run.sh "$abl" "$cam" "$g" >/dev/null 2>&1 < /dev/null &
      PID[$g]=$!
    fi
  done
  sleep 15
done
echo "[$(date +%H:%M:%S)] CAMPAIGN done" >> ablation_logs/run.log
