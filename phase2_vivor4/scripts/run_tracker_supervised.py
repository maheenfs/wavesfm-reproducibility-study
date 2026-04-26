from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = PROJECT_ROOT / "phase2_vivor4"
TRACKER_SCRIPT = PHASE2_ROOT / "scripts" / "wait_for_radcom_and_run_next.py"
STATUS_PATH = PHASE2_ROOT / "automation_logs" / "after_radcom_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WavesFM tracker under a restart loop with stale-heartbeat detection."
    )
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--label", type=str, default="tracker")
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stale-seconds", type=float, default=900.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=120.0)
    parser.add_argument("--graceful-stop-seconds", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=50)
    parser.add_argument("tracker_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def normalize_tracker_args(raw_args: list[str], session_root: Path) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]

    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        part = args[idx]
        if part == "--session-root":
            idx += 2
            continue
        if part == "--reuse-current-session":
            idx += 1
            continue
        normalized.append(part)
        idx += 1

    return ["--session-root", str(session_root), *normalized]


def tracker_command(session_root: Path, tracker_args: list[str]) -> list[str]:
    return [sys.executable, str(TRACKER_SCRIPT), *normalize_tracker_args(tracker_args, session_root)]


def read_status_age_seconds(expected_session_root: Path) -> float | None:
    if not STATUS_PATH.exists():
        return None
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return max(0.0, time.time() - STATUS_PATH.stat().st_mtime)

    payload_session = payload.get("session_root")
    if str(payload_session or "") != str(expected_session_root):
        return None

    updated = payload.get("updated_at_utc")
    if isinstance(updated, str) and updated:
        try:
            return max(0.0, time.time() - datetime.fromisoformat(updated).timestamp())
        except Exception:
            pass
    return max(0.0, time.time() - STATUS_PATH.stat().st_mtime)


def terminate_process_group(proc: subprocess.Popen[bytes], *, graceful_stop_seconds: float, log_path: Path, reason: str) -> int:
    append_log(log_path, f"Stopping tracker pid={proc.pid}: {reason}")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait(timeout=1)

    deadline = time.time() + max(graceful_stop_seconds, 0.0)
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return rc
        time.sleep(0.2)

    append_log(log_path, f"Escalating tracker pid={proc.pid} to SIGKILL")
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.wait(timeout=5)


def main() -> int:
    args = parse_args()
    session_root = args.session_root.expanduser().resolve()
    log_path = session_root / "supervisor.log"
    command = tracker_command(session_root, list(args.tracker_args))

    append_log(log_path, f"Supervisor start label={args.label}")
    append_log(log_path, "Tracker command: " + " ".join(command))

    restart_count = 0
    attempt = 0

    while True:
        attempt += 1
        append_log(log_path, f"Launch attempt {attempt}")
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        launched_at = time.time()
        stop_reason: str | None = None

        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    break

                time.sleep(max(args.poll_seconds, 1.0))
                age_seconds = read_status_age_seconds(session_root)
                if (
                    age_seconds is not None
                    and age_seconds > max(args.stale_seconds, 0.0)
                    and (time.time() - launched_at) >= max(args.startup_grace_seconds, 0.0)
                ):
                    stop_reason = f"status heartbeat stale for {age_seconds:.1f}s"
                    rc = terminate_process_group(
                        proc,
                        graceful_stop_seconds=args.graceful_stop_seconds,
                        log_path=log_path,
                        reason=stop_reason,
                    )
                    break
        except KeyboardInterrupt:
            stop_reason = "supervisor interrupted"
            terminate_process_group(
                proc,
                graceful_stop_seconds=args.graceful_stop_seconds,
                log_path=log_path,
                reason=stop_reason,
            )
            append_log(log_path, "Supervisor interrupted by user")
            return 130

        if rc == 0:
            append_log(log_path, "Tracker completed successfully")
            return 0

        restart_count += 1
        append_log(
            log_path,
            f"Tracker exited rc={rc} ({stop_reason or 'process error'}); restart {restart_count}/{args.max_restarts}",
        )
        if args.max_restarts >= 0 and restart_count > args.max_restarts:
            append_log(log_path, "Restart budget exhausted")
            return int(rc or 1)

        time.sleep(max(args.restart_delay_seconds, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
