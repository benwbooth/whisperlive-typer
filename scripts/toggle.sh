#!/usr/bin/env bash
# Toggle WhisperLive Typer on/off
# Designed to be triggered by a KDE hotkey

set -euo pipefail

# PROJECT_DIR can be set by the flake wrapper, or auto-detected
if [[ -z "${PROJECT_DIR:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

PID_FILE="$HOME/.cache/whisper-typer/pid"
LOG_FILE="$HOME/.cache/whisper-typer/log"

# Ensure cache directory exists
mkdir -p "$(dirname "$PID_FILE")"

# Send desktop notification
notify() {
    local title="$1"
    local message="$2"
    local icon="${3:-audio-input-microphone}"

    if command -v notify-send &>/dev/null; then
        notify-send -i "$icon" "$title" "$message" -t 2000
    fi
}

# Check if running
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        # Stale PID file
        rm -f "$PID_FILE"
    fi
    return 1
}

# Start the typer
start_typer() {
    # Check if server is running
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q whisperlive-rocm; then
        notify "Whisper Typer" "Server not running! Starting..." "dialog-warning"

        if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
            cd "$PROJECT_DIR"
            if command -v "docker compose" &>/dev/null; then
                docker compose up -d
            else
                docker-compose up -d
            fi
            # Wait a moment for server to start
            sleep 3
        else
            notify "Whisper Typer" "docker-compose.yml not found" "dialog-error"
            echo "Error: docker-compose.yml not found at $PROJECT_DIR"
            exit 1
        fi
    fi

    # Run via nix develop to get portaudio in LD_LIBRARY_PATH
    # Config is loaded from ~/.config/whisper-typer/config.yaml
    cd "$PROJECT_DIR"
    nix develop --command ./whisper_typer.py >> "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    echo "Started whisper-typer (PID: $pid)"
}

# Stop the typer
stop_typer() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            # Wait briefly for graceful shutdown
            sleep 0.5
            # Force kill if still running
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi

    echo "Stopped whisper-typer"
}

# Main toggle logic
main() {
    if is_running; then
        stop_typer
    else
        start_typer
    fi
}

main "$@"
