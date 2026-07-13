#!/usr/bin/env bash
# job_pool.sh "<gpu_csv>" "cmd with {GPU}" ...   — runs cmds across GPUs, one per GPU, refilling.
set -uo pipefail; cd "$(dirname "$0")"
IFS=',' read -ra GPUS <<< "${1:?gpus}"; shift; CMDS=("$@"); declare -A PID; i=0
while (( i < ${#CMDS[@]} )) || (( ${#PID[@]} )); do
  for g in "${GPUS[@]}"; do
    if [[ -n "${PID[$g]:-}" ]] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    unset "PID[$g]" 2>/dev/null || true
    if (( i < ${#CMDS[@]} )); then c="${CMDS[$i]//\{GPU\}/$g}"; ((i++)); setsid bash -c "$c" >/dev/null 2>&1 </dev/null & PID[$g]=$!; fi
  done; sleep 15
done; echo "[$(date +%H:%M:%S)] job_pool done" >> occl_logs/pool.log
