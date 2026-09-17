#!/bin/bash
# Launcher for the persistent code-rag HTTP server.
# Safe to call from several Claude Code sessions at once (SessionStart hook):
#   - a lock serializes concurrent starts
#   - a live server is never killed by `start`, even while it is still loading
#     models or busy indexing
#   - stale-port cleanup only ever targets a LISTENING http_server.py process,
#     never client connections (the old `lsof -ti :PORT` matched those too)
#
# Usage: ./code-rag-server.sh [start|stop|status|restart|logs]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$HOME/.code-rag"
PID_FILE="$SERVER_DIR/server.pid"
WATCHDOG_PID_FILE="$SERVER_DIR/watchdog.pid"
LOG_FILE="$SERVER_DIR/server.log"
LOCK_DIR="$SERVER_DIR/start.lock"
PYTHON="$SCRIPT_DIR/venv/bin/python"
PORT="${CODE_RAG_PORT:-7101}"
HEALTH_URL="http://127.0.0.1:$PORT/health"
START_TIMEOUT="${CODE_RAG_START_TIMEOUT:-180}"   # model load can take a while on first start
HEALTH_TIMEOUT=10
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=6                           # 3 minutes of consecutive failures
LOG_MAX_BYTES=$((20 * 1024 * 1024))
LOG_KEEP=3

mkdir -p "$SERVER_DIR"

log() { echo "[code-rag] $*" >&2; }
wlog() { echo "$(date '+%Y-%m-%d %H:%M:%S') [code-rag-watchdog] $*" >> "$LOG_FILE"; }

server_pid() { cat "$PID_FILE" 2>/dev/null || true; }

pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

pid_is_ours() {
    [ -n "${1:-}" ] && ps -o command= -p "$1" 2>/dev/null | grep -q "http_server.py"
}

is_alive() {
    local pid; pid=$(server_pid)
    pid_alive "$pid" && pid_is_ours "$pid"
}

is_healthy() { curl -sf --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1; }

listener_pids() { lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true; }

rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(stat -f %z "$LOG_FILE" 2>/dev/null || echo 0)" -ge "$LOG_MAX_BYTES" ]; then
        local i
        for (( i=LOG_KEEP-1; i>=1; i-- )); do
            [ -f "$LOG_FILE.$i" ] && mv -f "$LOG_FILE.$i" "$LOG_FILE.$((i+1))"
        done
        mv -f "$LOG_FILE" "$LOG_FILE.1"
    fi
}

acquire_lock() {
    local waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        # Stale lock (older than 5 minutes) from a killed start: remove it.
        if [ -d "$LOCK_DIR" ]; then
            local age=$(( $(date +%s) - $(stat -f %m "$LOCK_DIR" 2>/dev/null || date +%s) ))
            if [ "$age" -gt 300 ]; then rmdir "$LOCK_DIR" 2>/dev/null || true; continue; fi
        fi
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 60 ]; then log "Another start is in progress (lock $LOCK_DIR); giving up."; exit 1; fi
    done
    trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
}

kill_pid() {  # kill_pid PID GRACE_SECONDS
    local pid="$1" grace="${2:-30}" waited=0
    pid_alive "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    while pid_alive "$pid" && [ "$waited" -lt "$grace" ]; do sleep 1; waited=$((waited + 1)); done
    if pid_alive "$pid"; then kill -9 "$pid" 2>/dev/null || true; sleep 1; fi
}

clear_port() {
    # Only ever kill a listener that is one of our http_server.py processes.
    local pid
    for pid in $(listener_pids); do
        if pid_is_ours "$pid"; then
            log "Killing stale code-rag listener on port $PORT (PID $pid)"
            kill_pid "$pid" 10
        else
            log "Port $PORT is held by another process (PID $pid: $(ps -o command= -p "$pid" | head -c 80))."
            log "Set CODE_RAG_PORT to use a different port."
            return 1
        fi
    done
}

launch_server() {
    rotate_log
    export PYTHONPATH="$SCRIPT_DIR"
    nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" </dev/null >> "$LOG_FILE" 2>&1 &
    local spid=$!
    disown "$spid" 2>/dev/null || true
    echo "$spid"
}

wait_healthy() {  # wait_healthy PID TIMEOUT
    local pid="$1" timeout="$2" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        sleep 1
        waited=$((waited + 1))
        if ! pid_alive "$pid"; then return 1; fi
        if is_healthy; then return 0; fi
    done
    return 2
}

stop_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local wpid; wpid=$(cat "$WATCHDOG_PID_FILE")
        pid_alive "$wpid" && kill "$wpid" 2>/dev/null || true
        rm -f "$WATCHDOG_PID_FILE"
    fi
}

start_watchdog() {
    stop_watchdog
    # The subshell must not inherit our stdout/stderr: anything that reads this
    # script's output (a SessionStart hook, `start | tail`) would otherwise wait
    # for the watchdog to exit, i.e. forever.
    (
        set +e
        trap - EXIT
        failures=0
        while true; do
            sleep "$WATCHDOG_INTERVAL"
            pid=$(server_pid)
            if ! pid_alive "$pid"; then
                wlog "Server process $pid is gone; restarting."
                failures=$WATCHDOG_MAX_FAILURES
            elif is_healthy; then
                failures=0
            else
                failures=$((failures + 1))
                wlog "Health check failed ($failures/$WATCHDOG_MAX_FAILURES)"
            fi

            if [ "$failures" -ge "$WATCHDOG_MAX_FAILURES" ]; then
                wlog "Restarting server..."
                pid=$(server_pid)
                if pid_alive "$pid" && pid_is_ours "$pid"; then kill_pid "$pid" 30; fi
                rm -f "$PID_FILE"
                for lp in $(listener_pids); do pid_is_ours "$lp" && kill_pid "$lp" 10; done
                new_pid=$(launch_server)
                if wait_healthy "$new_pid" "$START_TIMEOUT"; then
                    wlog "Server restarted (PID $new_pid)"
                else
                    wlog "Server failed to become healthy after restart; will retry on next failure window"
                fi
                failures=0
            fi
        done
    ) </dev/null >/dev/null 2>&1 &
    local wpid=$!
    disown "$wpid" 2>/dev/null || true
    echo "$wpid" > "$WATCHDOG_PID_FILE"
    log "Watchdog started (PID $wpid)"
}

do_start() {
    acquire_lock
    if is_alive; then
        if is_healthy; then
            log "Already running (PID $(server_pid))"
        else
            log "Server PID $(server_pid) is alive but not healthy yet (starting or busy); leaving it alone."
            log "Use '$0 restart' to force a restart, or '$0 logs' to look."
        fi
        [ -f "$WATCHDOG_PID_FILE" ] && pid_alive "$(cat "$WATCHDOG_PID_FILE")" || start_watchdog
        exit 0
    fi

    rm -f "$PID_FILE"
    clear_port || exit 1

    log "Starting HTTP server on port $PORT..."
    local server_pid; server_pid=$(launch_server)
    wait_healthy "$server_pid" "$START_TIMEOUT"
    case $? in
        0) log "Server ready (PID $server_pid)"; start_watchdog; exit 0 ;;
        1) log "Server process died during startup. Last log lines:"; tail -20 "$LOG_FILE" >&2; exit 1 ;;
        *) log "Server not healthy after ${START_TIMEOUT}s (still running as PID $server_pid). Check $LOG_FILE"; exit 1 ;;
    esac
}

do_stop() {
    stop_watchdog
    local pid; pid=$(server_pid)
    if pid_alive "$pid"; then
        log "Stopping server (PID $pid)..."
        kill_pid "$pid" 30   # graceful: lets an in-flight index batch finish its write
        rm -f "$PID_FILE"
        log "Stopped."
    else
        rm -f "$PID_FILE"
        log "Not running."
    fi
}

do_status() {
    if is_alive; then
        if is_healthy; then
            log "Running (PID $(server_pid))"
            curl -s --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); [print(f"  {k}: {v}") for k,v in d.items() if k not in ("watchers","active_jobs","chunker")]; print("  chunker:", d.get("chunker",{}).get("active"), "|", d.get("chunker",{}).get("node")); [print(f"  watcher {k}: {v}") for k,v in d.get("watchers",{}).items()]; [print(f"  job: {j}") for j in d.get("active_jobs",[])]' 2>/dev/null || true
        else
            log "Running (PID $(server_pid)) but not answering /health"
            exit 2
        fi
    else
        log "Not running."
        exit 1
    fi
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart) do_stop; do_start ;;
    logs)    tail -n "${2:-50}" "$LOG_FILE" ;;
    *)       echo "Usage: $0 {start|stop|status|restart|logs [n]}" >&2; exit 1 ;;
esac
