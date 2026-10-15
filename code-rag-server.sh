#!/bin/bash
# Launcher for the persistent code-rag HTTP server.
# Safe to call from several Claude Code sessions at once (SessionStart hook):
#   - a lock serializes concurrent starts
#   - a live server is never killed by `start`, even while it is still loading
#     models or busy indexing
#   - stale-port cleanup only ever targets a LISTENING http_server.py process,
#     never client connections (the old `lsof -ti :PORT` matched those too)
#
# Linux: `install-systemd` installs a systemd --user service (code-rag.service)
# that starts at boot (with `loginctl enable-linger $USER`) and restarts on
# failure. Once it is installed, start/stop/status/restart/logs go through
# systemctl/journalctl and the watchdog below is not used. The service reads
# ~/.config/code-rag.env (e.g. CODE_RAG_BACKEND=bedrock) and always binds 127.0.0.1.
# Without the service (macOS, or Linux before install-systemd) the launcher runs
# the server itself as before, reading ~/.config/code-rag.env if it exists.
#
# Usage: ./code-rag-server.sh [start|stop|status|restart|logs [n]|install-systemd|uninstall-systemd]

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
ENV_FILE="${CODE_RAG_ENV_FILE:-$HOME/.config/code-rag.env}"
SYSTEMD_UNIT="code-rag.service"
SYSTEMD_UNIT_FILE="$HOME/.config/systemd/user/$SYSTEMD_UNIT"

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

listener_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true
    else
        ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
    fi
}

# BSD stat (macOS) and GNU stat (Linux) take different flags.
file_size()  { stat -f %z "$1" 2>/dev/null || stat -c %s "$1" 2>/dev/null || echo 0; }
file_mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null || date +%s; }

rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(file_size "$LOG_FILE")" -ge "$LOG_MAX_BYTES" ]; then
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
            local age=$(( $(date +%s) - $(file_mtime "$LOCK_DIR") ))
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

load_env_file() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
    fi
}

launch_server() {
    rotate_log
    load_env_file
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

# --- systemd --user (Linux) ---

systemd_installed() { [ "$(uname -s)" = "Linux" ] && [ -f "$SYSTEMD_UNIT_FILE" ]; }

print_health() {
    curl -s --max-time "$HEALTH_TIMEOUT" "$HEALTH_URL" | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); [print(f"  {k}: {v}") for k,v in d.items() if k not in ("watchers","active_jobs","chunker")]; print("  chunker:", d.get("chunker",{}).get("active"), "|", d.get("chunker",{}).get("node")); [print(f"  watcher {k}: {v}") for k,v in d.get("watchers",{}).items()]; [print(f"  job: {j}") for j in d.get("active_jobs",[])]' 2>/dev/null || true
}

systemd_wait_healthy() {
    local waited=0
    while [ "$waited" -lt "$START_TIMEOUT" ]; do
        if is_healthy; then log "Server ready (systemd $SYSTEMD_UNIT)"; return 0; fi
        local state; state=$(systemctl --user show -p ActiveState --value "$SYSTEMD_UNIT" 2>/dev/null)
        if [ "$state" != "active" ] && [ "$state" != "activating" ] && [ "$state" != "reloading" ]; then
            log "$SYSTEMD_UNIT is not running. Last log lines:"
            journalctl --user -u "$SYSTEMD_UNIT" -n 20 --no-pager >&2 || true
            return 1
        fi
        sleep 1; waited=$((waited + 1))
    done
    log "Not healthy after ${START_TIMEOUT}s; see: $0 logs"
    return 1
}

write_systemd_unit() {
    mkdir -p "$(dirname "$SYSTEMD_UNIT_FILE")"
    cat > "$SYSTEMD_UNIT_FILE" << UNITEOF
# Installed by $SCRIPT_DIR/code-rag-server.sh install-systemd
[Unit]
Description=code-rag MCP server (127.0.0.1:$PORT)
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
# Optional settings (CODE_RAG_BACKEND=bedrock, models, CODE_RAG_NODE, ...).
EnvironmentFile=-$ENV_FILE
Environment=PYTHONPATH=$SCRIPT_DIR
Environment=GRPC_VERBOSITY=ERROR
# Host and port are set on the command line so the env file cannot override them:
# the server must only ever listen on loopback.
ExecStart=/usr/bin/env CODE_RAG_HOST=127.0.0.1 CODE_RAG_PORT=$PORT $PYTHON -u $SCRIPT_DIR/http_server.py
Restart=on-failure
RestartSec=10
# Lets an in-flight index batch finish its write (uvicorn graceful shutdown is 30s).
TimeoutStopSec=45

[Install]
WantedBy=default.target
UNITEOF
}

do_install_systemd() {
    [ "$(uname -s)" = "Linux" ] || { log "install-systemd is for Linux; on macOS use start (hooks)"; exit 1; }
    do_stop   # a launcher-managed server and its watchdog would hold the port
    write_systemd_unit
    systemctl --user daemon-reload
    systemctl --user enable --now "$SYSTEMD_UNIT"
    log "Installed $SYSTEMD_UNIT_FILE (enabled)."
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
        log "Note: run 'loginctl enable-linger $USER' so the service starts at boot without a login."
    fi
    systemd_wait_healthy
}

do_uninstall_systemd() {
    systemctl --user disable --now "$SYSTEMD_UNIT" 2>/dev/null || true
    rm -f "$SYSTEMD_UNIT_FILE"
    systemctl --user daemon-reload
    log "Removed $SYSTEMD_UNIT."
}

if systemd_installed; then
    case "${1:-start}" in
        start)
            if systemctl --user is-active --quiet "$SYSTEMD_UNIT" && is_healthy; then
                log "Already running (systemd $SYSTEMD_UNIT, PID $(systemctl --user show -p MainPID --value "$SYSTEMD_UNIT"))"
                exit 0
            fi
            systemctl --user start "$SYSTEMD_UNIT"; systemd_wait_healthy ;;
        stop)    systemctl --user stop "$SYSTEMD_UNIT"; log "Stopped (systemd $SYSTEMD_UNIT)." ;;
        restart) systemctl --user restart "$SYSTEMD_UNIT"; systemd_wait_healthy ;;
        status)
            systemctl --user --no-pager status "$SYSTEMD_UNIT" | head -n 5 >&2 || true
            if is_healthy; then print_health; else log "Not answering /health"; exit 2; fi ;;
        logs)    journalctl --user -u "$SYSTEMD_UNIT" -n "${2:-50}" --no-pager ;;
        install-systemd)   do_install_systemd ;;
        uninstall-systemd) do_uninstall_systemd ;;
        *) echo "Usage: $0 {start|stop|status|restart|logs [n]|install-systemd|uninstall-systemd}" >&2; exit 1 ;;
    esac
    exit $?
fi

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart) do_stop; do_start ;;
    logs)    tail -n "${2:-50}" "$LOG_FILE" ;;
    install-systemd)   do_install_systemd ;;
    uninstall-systemd) do_uninstall_systemd ;;
    *)       echo "Usage: $0 {start|stop|status|restart|logs [n]|install-systemd|uninstall-systemd}" >&2; exit 1 ;;
esac
