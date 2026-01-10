#!/usr/bin/env bash
# Toggle WhisperLive Typer on/off via systemd user service
# Designed to be triggered by a hotkey

set -euo pipefail

SERVICE="whisper-typer"
NOTIFICATION_ID_FILE="$HOME/.cache/whisper-typer/notification_id"

# Check if running
is_running() {
    systemctl --user is-active --quiet "$SERVICE" 2>/dev/null
}

# Close notification using saved ID (backup cleanup)
close_notification() {
    if [[ -f "$NOTIFICATION_ID_FILE" ]]; then
        local nid
        nid=$(cat "$NOTIFICATION_ID_FILE")
        gdbus call --session \
            --dest org.freedesktop.Notifications \
            --object-path /org/freedesktop/Notifications \
            --method org.freedesktop.Notifications.CloseNotification \
            "$nid" 2>/dev/null || true
        rm -f "$NOTIFICATION_ID_FILE"
    fi
}

# Install service if needed
install_service() {
    local service_dir="$HOME/.config/systemd/user"
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_dir="$(dirname "$script_dir")"
    local service_file="$project_dir/whisper-typer.service"

    if [[ ! -f "$service_dir/$SERVICE.service" ]] || \
       [[ "$service_file" -nt "$service_dir/$SERVICE.service" ]]; then
        mkdir -p "$service_dir"
        cp "$service_file" "$service_dir/"
        systemctl --user daemon-reload
    fi
}

# Main toggle logic
main() {
    install_service

    if is_running; then
        systemctl --user stop "$SERVICE"
        # Backup: close notification in case atexit didn't run
        close_notification
        echo "Stopped $SERVICE"
    else
        systemctl --user start "$SERVICE"
        echo "Started $SERVICE"
    fi
}

main "$@"
