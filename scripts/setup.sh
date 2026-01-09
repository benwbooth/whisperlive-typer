#!/usr/bin/env bash
# Setup script for WhisperLive with ydotool on NixOS/KDE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== WhisperLive Setup ==="
echo ""

# Check dependencies
check_deps() {
    local missing=()

    command -v docker &>/dev/null || missing+=("docker")
    command -v docker-compose &>/dev/null || command -v "docker compose" &>/dev/null || missing+=("docker-compose")
    command -v ydotool &>/dev/null || missing+=("ydotool")
    command -v uv &>/dev/null || missing+=("uv")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Missing dependencies: ${missing[*]}"
        echo ""
        echo "Install missing packages:"
        echo "  docker, docker-compose - for the server"
        echo "  ydotool - for keyboard simulation"
        echo "  uv - Python package manager (https://docs.astral.sh/uv/)"
        exit 1
    fi

    echo "✓ All dependencies found"
}

# Check if ydotoold is running
check_ydotool() {
    if ! pgrep -x ydotoold &>/dev/null; then
        echo ""
        echo "⚠ ydotoold is not running!"
        echo ""
        echo "Add to your NixOS configuration:"
        echo "  programs.ydotool.enable = true;"
        echo ""
        echo "Or start manually: ydotoold &"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    else
        echo "✓ ydotoold is running"
    fi
}

# Build Docker image
build_docker() {
    echo ""
    echo "Building Docker image (this may take 30+ minutes on first run)..."
    echo ""

    cd "$PROJECT_DIR"

    if command -v "docker compose" &>/dev/null; then
        docker compose build
    else
        docker-compose build
    fi

    echo ""
    echo "✓ Docker image built"
}

# Start the server
start_server() {
    echo ""
    echo "Starting WhisperLive server..."

    cd "$PROJECT_DIR"

    if command -v "docker compose" &>/dev/null; then
        docker compose up -d
    else
        docker-compose up -d
    fi

    echo "✓ Server started"
    echo ""
    echo "Check logs with: docker compose logs -f"
}

# Set up KDE hotkey automatically
setup_kde_hotkey_auto() {
    local toggle_script="$1"
    local kwriteconfig

    if command -v kwriteconfig6 &>/dev/null; then
        kwriteconfig=kwriteconfig6
    else
        kwriteconfig=kwriteconfig5
    fi

    # Create custom shortcut via khotkeys
    local khotkeys_file="$HOME/.config/khotkeysrc"

    # Add our shortcut group if it doesn't exist
    if ! grep -q "WhisperTyper" "$khotkeys_file" 2>/dev/null; then
        cat >> "$khotkeys_file" << EOF

[Data_3]
Comment=Whisper Typer Toggle
Enabled=true
Name=Whisper Typer
Type=SIMPLE_ACTION_DATA

[Data_3Conditions]

[Data_3Actions]
ActionsCount=1

[Data_3Actions0]
CommandURL=$toggle_script
Type=COMMAND_URL

[Data_3Triggers]
TriggersCount=1

[Data_3Triggers0]
Key=Meta+H
Type=SHORTCUT
Uuid={$(uuidgen)}
EOF
        # Reload khotkeys
        if command -v qdbus6 &>/dev/null; then
            qdbus6 org.kde.kglobalaccel /kglobalaccel reloadConfig 2>/dev/null || true
        elif command -v qdbus &>/dev/null; then
            qdbus org.kde.kglobalaccel /kglobalaccel reloadConfig 2>/dev/null || true
        fi
        return 0
    else
        echo "WhisperTyper hotkey already configured"
        return 0
    fi
}

# Configure KDE hotkey
setup_kde_hotkey() {
    echo ""
    echo "=== KDE Hotkey Setup ==="
    echo ""

    local toggle_script="$PROJECT_DIR/scripts/toggle.sh"
    local desktop_file="$HOME/.local/share/applications/whisper-typer-toggle.desktop"
    local khotkeys_dir="$HOME/.config"

    # Create .desktop file for KDE
    mkdir -p "$(dirname "$desktop_file")"
    cat > "$desktop_file" << EOF
[Desktop Entry]
Name=Whisper Typer Toggle
Comment=Toggle speech-to-text typing
Exec=$toggle_script
Icon=audio-input-microphone
Type=Application
Categories=Utility;
NoDisplay=true
EOF

    echo "✓ Created desktop entry: $desktop_file"
    echo ""
    echo "To set up the hotkey in KDE:"
    echo ""
    echo "1. Open System Settings → Shortcuts → Custom Shortcuts"
    echo "2. Click 'Edit → New → Global Shortcut → Command/URL'"
    echo "3. Name it 'Whisper Typer Toggle'"
    echo "4. Set the command to: $toggle_script"
    echo "5. Click the 'Trigger' tab and set: Super+H"
    echo ""

    # Try to set up the hotkey automatically using kwriteconfig
    if command -v kwriteconfig6 &>/dev/null || command -v kwriteconfig5 &>/dev/null; then
        echo "Attempting to configure hotkey automatically..."
        setup_kde_hotkey_auto "$toggle_script" && echo "✓ Hotkey Super+H configured!" || echo "Manual setup required (see above)"
    fi
}

# Select microphone
select_microphone() {
    echo ""
    echo "=== Select Microphone ==="
    echo ""

    # Get device names into array
    local -a device_names=()
    local -a device_indices=()

    while IFS= read -r line; do
        # Extract index and name from lines like:   [4] "Logitech Webcam C930e"
        if [[ $line =~ \[([0-9]+)\]\ \"(.+)\" ]]; then
            device_indices+=("${BASH_REMATCH[1]}")
            device_names+=("${BASH_REMATCH[2]}")
        fi
    done < <("$PROJECT_DIR/whisper_typer.py" --list-devices 2>/dev/null)

    if [[ ${#device_names[@]} -eq 0 ]]; then
        echo "No microphones found!"
        SELECTED_DEVICE=""
        return
    fi

    # Display numbered list
    local i
    for i in "${!device_names[@]}"; do
        echo "  $((i+1))) ${device_names[$i]}"
    done
    echo ""
    echo "  0) System default"
    echo ""
    read -p "Select microphone [1-${#device_names[@]}, or 0 for default]: " choice

    if [[ -z "$choice" || "$choice" == "0" ]]; then
        SELECTED_DEVICE=""
        echo "Using system default microphone"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#device_names[@]} )); then
        SELECTED_DEVICE="${device_names[$((choice-1))]}"
        echo "Selected: $SELECTED_DEVICE"
    else
        echo "Invalid selection, using system default"
        SELECTED_DEVICE=""
    fi
}

# Create config file
create_config() {
    local config_dir="$HOME/.config/whisper-typer"
    local config_file="$config_dir/config.yaml"
    local device_value="${SELECTED_DEVICE:-}"

    mkdir -p "$config_dir"

    if [[ -f "$config_file" ]]; then
        echo ""
        echo "Config file already exists: $config_file"
        read -p "Overwrite? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || return
    fi

    cat > "$config_file" << EOF
# WhisperLive Typer Configuration

server:
  host: localhost
  port: 9090

whisper:
  language: en
  model: small

audio:
  # Microphone device - index number or name substring (e.g., "C930e" or "4")
  # Leave empty for system default
  device: "$device_value"

vad:
  # Voice Activity Detection thresholds (0.0 - 1.0)
  # Run: scripts/vad_calibrate.py --auto
  # Lower = more sensitive (may trigger on noise)
  # Higher = less sensitive (may miss quiet speech)
  onset: 0.3
  offset: 0.2

# Voice commands - map phrases to actions
commands:
  # Keyboard shortcuts
  keys:
    "scratch that": "ctrl+z"
    "undo": "ctrl+z"
    "redo": "ctrl+shift+z"
    "new line": "enter"
    "new paragraph": "enter enter"
    "go back": "backspace"
    "delete word": "ctrl+backspace"
    "select all": "ctrl+a"
    "copy": "ctrl+c"
    "paste": "ctrl+v"
    "cut": "ctrl+x"
    "save": "ctrl+s"

  # Literal text strings (uncomment to use)
  literals: {}
    # "smiley face": ":)"
    # "my email": "you@example.com"
EOF

    echo ""
    echo "✓ Created config file: $config_file"
}

# Main
main() {
    check_deps
    check_ydotool

    echo ""
    read -p "Build Docker image now? (takes ~30 min first time) [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        build_docker
        start_server
    fi

    select_microphone
    create_config
    setup_kde_hotkey

    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Press Super+H to toggle speech-to-text!"
    echo ""
    echo "Config:    ~/.config/whisper-typer/config.yaml"
    echo "Test:      $PROJECT_DIR/scripts/toggle.sh"
    echo "Calibrate: $PROJECT_DIR/scripts/vad_calibrate.py --auto"
    echo ""
}

main "$@"
