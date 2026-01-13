#!/usr/bin/env bash
# Toggle WhisperLive Typer on/off
# Designed to be triggered by a hotkey
# Supports both Linux (systemd) and macOS (launchctl/direct process)

set -euo pipefail

# Ensure PATH includes common locations (needed when run from Karabiner/launchd)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

SERVICE="whisper-typer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PGREP_PATTERN="whisper_typer.py"

# Detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="macos"
else
    PLATFORM="linux"
fi

# ============ Linux (systemd) ============

linux_is_running() {
    systemctl --user is-active --quiet "$SERVICE" 2>/dev/null
}

linux_close_notification() {
    local notification_id_file="$HOME/.cache/whisper-typer/notification_id"
    if [[ -f "$notification_id_file" ]]; then
        local nid
        nid=$(cat "$notification_id_file")
        gdbus call --session \
            --dest org.freedesktop.Notifications \
            --object-path /org/freedesktop/Notifications \
            --method org.freedesktop.Notifications.CloseNotification \
            "$nid" 2>/dev/null || true
        rm -f "$notification_id_file"
    fi
}

linux_install_service() {
    local service_dir="$HOME/.config/systemd/user"
    local service_file="$PROJECT_DIR/whisper-typer.service"

    if [[ ! -f "$service_dir/$SERVICE.service" ]] || \
       [[ "$service_file" -nt "$service_dir/$SERVICE.service" ]]; then
        mkdir -p "$service_dir"
        cp "$service_file" "$service_dir/"
        systemctl --user daemon-reload
    fi
}

linux_toggle() {
    linux_install_service

    if linux_is_running; then
        systemctl --user stop "$SERVICE"
        linux_close_notification
        echo "Stopped $SERVICE"
    else
        systemctl --user start "$SERVICE"
        echo "Started $SERVICE"
    fi
}

# ============ macOS (launchctl) ============

MACOS_SERVICE_LABEL="com.whispertyper.client"

macos_is_running() {
    # Check if launchctl service has a PID (first column is not "-")
    local pid
    pid=$(launchctl list | grep "$MACOS_SERVICE_LABEL" | awk '{print $1}')
    [[ -n "$pid" && "$pid" != "-" ]]
}

macos_toggle() {
    if macos_is_running; then
        launchctl kill SIGTERM "gui/$UID/$MACOS_SERVICE_LABEL"
        echo "Stopped whisper-typer"
    else
        launchctl kickstart "gui/$UID/$MACOS_SERVICE_LABEL"
        echo "Started whisper-typer"
    fi
}

# ============ Main ============

main() {
    if [[ "$PLATFORM" == "macos" ]]; then
        macos_toggle
    else
        linux_toggle
    fi
}

main "$@"
