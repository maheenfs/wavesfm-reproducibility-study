#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-ngwn06}"
LOCAL_PORT="${2:-8765}"
REMOTE_PORT="${3:-8765}"
REMOTE_BIND="${4:-localhost}"

SPEC="${LOCAL_PORT}:${REMOTE_BIND}:${REMOTE_PORT}"
BACKOFFS=(5 10 30 60 300 900 1200)
ATTEMPT=0

kill_stale_listener() {
  local pids pid cmd
  pids="$(lsof -tiTCP:${LOCAL_PORT} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    cmd="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    if [[ "${cmd}" == *ssh* ]] || [[ "${cmd}" == *autossh* ]]; then
      echo "[tunnel-watch] killing stale listener pid=${pid} cmd=${cmd}"
      kill "${pid}" 2>/dev/null || true
    fi
  done <<< "${pids}"
}

next_backoff() {
  local idx=$1
  if (( idx < ${#BACKOFFS[@]} )); then
    echo "${BACKOFFS[$idx]}"
  else
    echo "${BACKOFFS[-1]}"
  fi
}

echo "[tunnel-watch] host=${HOST} local_port=${LOCAL_PORT} remote=${REMOTE_BIND}:${REMOTE_PORT}"

while true; do
  echo "[tunnel-watch] opening ssh -o ControlMaster=no -o ControlPath=none -N -L ${SPEC} ${HOST}"
  set +e
  ssh -o ControlMaster=no -o ControlPath=none -N -L "${SPEC}" "${HOST}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    ATTEMPT=0
    sleep 1
    continue
  fi

  if lsof -tiTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    kill_stale_listener
  fi

  backoff="$(next_backoff "${ATTEMPT}")"
  echo "[tunnel-watch] ssh exited rc=${rc}; retrying in ${backoff}s"
  sleep "${backoff}"
  ((ATTEMPT+=1))
done
