#!/bin/bash
# run_step7_detached.sh
# ----------------------
# Launches workers/step7_worker.py in a FULLY detached session so that the
# OpenWork shell tool returns immediately and a tool timeout can NEVER kill
# the worker mid-run (critical for long executions: 100k+ docs, many hours).
#
# How it works:
#   - Python's subprocess.Popen(start_new_session=True) puts the worker in a
#     brand-new session/process group, reparented to init once the tiny launcher
#     exits. No file descriptor of the shell tool is inherited (stdin=/dev/null,
#     stdout/stderr -> the log file), so the tool call returns right away.
#   - The worker is checkpoint-based (data/02_intermediate/step7_checkpoints/):
#     rerunning this script resumes from the last completed vector.
#
# Usage:
#   ./run_step7_detached.sh
#   cat data/07_logs/step7.pid   # PID of the worker
#
# Log: data/07_logs/step7_run.log

set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE}/data/07_logs"
LOG_FILE="${LOG_DIR}/step7_run.log"
PID_FILE="${LOG_DIR}/step7.pid"

mkdir -p "${LOG_DIR}"

python3 - "$WORKSPACE" "$LOG_FILE" "$PID_FILE" <<'PYEOF'
import subprocess, sys, os

workspace, log_file, pid_file = sys.argv[1], sys.argv[2], sys.argv[3]

log_fh = open(log_file, "ab", buffering=0)

p = subprocess.Popen(
    ["./run_worker.sh", "workers/step7_worker.py"],
    cwd=workspace,
    stdin=subprocess.DEVNULL,
    stdout=log_fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,          # new session: immune to tool process-group kills
    close_fds=True,                  # no inherited fds from the shell tool
)

with open(pid_file, "w") as f:
    f.write(str(p.pid))
print(f"Step 7 worker launched detached. PID: {p.pid}")
print(f"Log: {log_file}")
PYEOF
