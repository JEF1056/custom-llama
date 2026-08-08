#!/usr/bin/env bash
#
# check-ml2-logs.sh — Inspect MLX server health on remote Macs
#
# SSHes into a fleet host, checks log file presence/content, verifies
# the server process is running, inspects the LaunchAgent, and reports
# the last 50 lines of each log file.
#
# Usage:
#   bash mac/check-ml2-logs.sh                  # check ml-2 (default)
#   bash mac/check-ml2-logs.sh ml-2             # check specific host
#   bash mac/check-ml2-logs.sh ml-2 ml-3 ml-4   # check multiple hosts
#
# Prerequisites:
#   - SSH configured for each host (passwordless or key-based)
#   - python3 or python available on the remote host
#
# Verification:
#   # Test with one host (hydrated from the local machine):
#     bash mac/check-ml2-logs.sh ml-2
#
#   # Test with fleet:
#     bash mac/check-ml2-logs.sh ml-2 ml-3 ml-4
#
#   # Verify expected output contains:
#   #   1) SSH connectivity confirmation
#   #   2) Log file existence status (EXISTS/MISSING)
#   #   3) Last 50 lines of each log (truncated if missing)
#   #   4) Check whether the MLX server is accessible
#   #   5) LaunchAgent status
#   #   6) Health summary block
#

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
HOSTS=()
MLX_PORT=8080
LOG_DIR="$HOME/Library/Logs"
OUT_LOG="qwen36-mlx.out.log"
ERR_LOG="qwen36-mlx.err.log"
SUPERVISOR_LOG="qwen36-mlx-supervisor.log"
PLIST_NAME="com.custom-llama.qwen36-mlx.plist"

# ---- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|--) MLX_PORT="${2:-$MLX_PORT}"; shift 2 ;;
        --help)
            sed -n '4,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) HOSTS+=("$1"); shift ;;
    esac
done

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    HOSTS=(ml-2)
fi

# ---- Color helpers -----------------------------------------------------------
green()  { printf '\033[1;32m%s\033[0m' "$*"; }
red()    { printf '\033[1;31m%s\033[0m' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m' "$*"; }
cyan()   { printf '\033[1;36m%s\033[0m' "$*"; }
dim()    { printf '\033[0;90m%s\033[0m' "$*"; }

# ---- SSH inline command helper ----------------------------------------------
# Runs a command remotely via SSH and captures output.
# Usage: remote_run <host> <command>
remote_run() {
    local host="$1"; shift
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "$*" 2>/dev/null
}

# ---- Check a single host ----------------------------------------------------
check_host() {
    local host="$1"
    local separator
    separator="================================================================="
    echo ""
    echo "$separator"
    cyan "  Host: $host"
    echo "$separator"

    # 1) SSH connectivity
    if ! remote_run "$host" "echo 'SSH connected'" &>/dev/null; then
        red "  [!] SSH unreachable — $host"
        echo "  Check SSH configuration: ssh -o ConnectTimeout=5 $host 'echo ok'"
        return 1
    fi
    green "  [OK] SSH connectivity confirmed"

    # 2) Log file presence check
    local log_count=0
    local log_total=3
    local logs_status=()

    for log_name in "$OUT_LOG" "$ERR_LOG" "$SUPERVISOR_LOG"; do
        local full_path="${LOG_DIR}/${log_name}"
        if remote_run "$host" "test -f '${full_path}'" 2>/dev/null; then
            green "  [EXISTS] $log_name"
            log_count=$((log_count + 1))
            logs_status+=("EXISTS")
        else
            yellow "  [MISSING] $log_name"
            logs_status+=("MISSING")
        fi
    done

    # 3) Last 50 lines of each log (only if it exists)
    for log_name in "$OUT_LOG" "$ERR_LOG" "$SUPERVISOR_LOG"; do
        local full_path="${LOG_DIR}/${log_name}"
        if [[ " ${logs_status[*]:-} " =~ " EXISTS " ]]; then
            if remote_run "$host" "test -f '${full_path}'" 2>/dev/null; then
                echo ""
                yellow "  --- Last 50 lines of $log_name ---"
                remote_run "$host" "tail -50 '${full_path}'" 2>/dev/null | \
                    sed "s/^/    /" || yellow "    (unable to retrieve)"
                echo "  --- end of $log_name ---"
                echo ""
            fi
        fi
    done

    # 4) Check if MLX server process is running and healthy
    echo "  -------------------------------------------"
    echo "  Server Health Probe"
    echo "  -------------------------------------------"

    # Check if the python3 server process is running
    local server_running=false
    if remote_run "$host" "pgrep -f '[m]lx_vlm.server'" >/dev/null 2>&1; then
        green "  [OK] Server process detected via pgrep"
        server_running=true

        # Check HTTP health endpoint
        local http_code
        http_code=$(remote_run "$host" \
            "curl -s -o /dev/null -w '%{http_code}' --connect-timeout 10 \
             'http://localhost:${MLX_PORT}/health'" 2>/dev/null || echo "000")

        if [[ "$http_code" == "200" ]]; then
            green "  [OK] HTTP health endpoint healthy (port $MLX_PORT)"
        elif [[ "$http_code" != "000" ]]; then
            yellow "  [WARN] HTTP health returned HTTP $http_code"
        else
            red "  [FAIL] No response from health endpoint on port $MLX_PORT"
        fi
    else
        red "  [FAIL] No server process found (mlx_vlm.server not running)"
    fi

    # 5) LaunchAgent status
    echo ""
    echo "  -------------------------------------------"
    echo "  LaunchAgent Status"
    echo "  -------------------------------------------"

    local plist_path="/Users/jfan/Library/LaunchAgents/${PLIST_NAME}"
    local agent_state="unknown"

    # Check if the plist exists
    if remote_run "$host" "test -f '${plist_path}'" 2>/dev/null; then
        green "  [EXISTS] Plist: $PLIST_NAME"

        # Check if it's loaded (launched)
        # launchctl list outputs "0\t<identifier>" when loaded, nothing when unloaded
        local load_state
        load_state=$(remote_run "$host" \
            "launchctl list '${PLIST_NAME%.*}' 2>/dev/null" 2>/dev/null || echo "")

        if [[ -n "$load_state" ]]; then
            green "  [LOADED] LaunchAgent is active"
            agent_state="loaded"
        else
            yellow "  [UNLOADED] Plist exists but agent may not be loaded"
            agent_state="unloaded"
        fi
    else
        red "  [MISSING] $PLIST_NAME not found at $plist_path"
        agent_state="not installed"
    fi

    # 6) Check for recent errors in stderr log
    echo ""
    echo "  -------------------------------------------"
    echo "  Recent Errors (from stderr log)"
    echo "  -------------------------------------------"

    if remote_run "$host" "test -f '${LOG_DIR}/${ERR_LOG}'" 2>/dev/null; then
        local error_count
        error_count=$(remote_run "$host" \
            "grep -ciE '(error|fatal|exception|traceback)' '${LOG_DIR}/${ERR_LOG}' 2>/dev/null || echo 0" 2>/dev/null)

        if [[ "$error_count" -gt 0 ]] 2>/dev/null; then
            red "  Found $error_count error/fatal lines in $ERR_LOG"
            echo "  -------------------------------------------"
            remote_run "$host" \
                "grep -iE '(error|fatal|exception|traceback)' '${LOG_DIR}/${ERR_LOG}' | tail -10" 2>/dev/null | \
                sed "s/^/    /" || echo "    (no match)"
        else
            green "  No errors detected in $ERR_LOG"
        fi
    else
        dim "    (stderr log not available)"
    fi

    # ---- Summary block ---------------------------------------------------
    echo ""
    green "  ===== Health Summary for $host ====="
    echo "    Server process:    $([ "$server_running" = true ] && echo 'RUNNING' || echo 'NOT RUNNING')"
    echo "    Logs available:    $log_count / $log_total"
    echo "    LaunchAgent:       $agent_state"
    echo "    Health status:     $([ "$server_running" = true ] && echo 'UP' || echo 'DOWN')"
    green "  ======================================"

    echo ""
    return 0
}

# ---- Main -------------------------------------------------------------------
cyan "========================================================"
cyan "  MLX Server Health Check"
cyan "========================================================"

EXIT_CODE=0
for host in "${HOSTS[@]}"; do
    if ! check_host "$host"; then
        EXIT_CODE=1
    fi
done

if [[ ${#HOSTS[@]} -gt 1 ]]; then
    echo ""
    yellow "Fleet Summary:"
    for host in "${HOSTS[@]}"; do
        if remote_run "$host" "pgrep -f '[m]lx_vlm.server'" >/dev/null 2>&1; then
            green "  $host: UP"
        else
            red "  $host: DOWN"
        fi
    done
    echo "========================================================"
fi

exit $EXIT_CODE
