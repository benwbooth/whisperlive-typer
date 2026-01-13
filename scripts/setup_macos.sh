#!/bin/bash
# Setup script for WhisperLive on macOS with MLX
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MACOS_DIR="$PROJECT_DIR/macos"
PLIST_NAME="com.whispertyper.server.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== WhisperLive macOS Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (arm64)"
    echo "Current architecture: $(uname -m)"
    echo ""
fi

# Check for uv
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# Install dependencies with uv
echo ""
echo "Installing Python dependencies..."
cd "$MACOS_DIR"
uv sync

# Verify MLX installation
echo ""
echo "Verifying MLX installation..."
uv run python -c "
import mlx.core as mx
import mlx_whisper
print(f'MLX device: {mx.default_device()}')
print('MLX Whisper: OK')
"

# Pre-download model
MODEL=${MLX_WHISPER_MODEL:-mlx-community/whisper-large-v3-turbo}
echo ""
echo "Pre-downloading model: $MODEL"
echo "(This may take a few minutes on first run)"
uv run python -c "
import mlx_whisper
import numpy as np
# Transcribe silence to trigger model download
silence = np.zeros(16000, dtype=np.float32)
try:
    mlx_whisper.transcribe(silence, path_or_hf_repo='$MODEL')
    print('Model downloaded successfully')
except Exception as e:
    print(f'Model download triggered (this is normal): {e}')
"
cd "$PROJECT_DIR"

# Update plist with correct paths
echo ""
echo "Configuring launchctl service..."
PLIST_SRC="$PROJECT_DIR/$PLIST_NAME"
PLIST_DST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

# Create LaunchAgents directory if needed
mkdir -p "$LAUNCH_AGENTS_DIR"

# Update paths in plist
UV_PATH="$(which uv)"
sed -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__UV_PATH__|$UV_PATH|g" \
    "$PLIST_SRC" > "$PLIST_DST"
echo "Installed plist to: $PLIST_DST"

# Unload if already loaded
if launchctl list | grep -q "com.whispertyper.server"; then
    echo "Unloading existing service..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Load service
echo "Loading service..."
launchctl load "$PLIST_DST"

# Check status
sleep 1
if launchctl list | grep -q "com.whispertyper.server"; then
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Service status:"
    launchctl list | grep whispertyper || echo "  (starting up...)"
    echo ""
    echo "View logs:"
    echo "  tail -f /tmp/whispertyper-server.log"
    echo "  tail -f /tmp/whispertyper-server.err"
    echo ""
    echo "Control service:"
    echo "  launchctl stop com.whispertyper.server"
    echo "  launchctl start com.whispertyper.server"
    echo "  launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
else
    echo ""
    echo "Warning: Service may not have started correctly"
    echo "Check logs: tail -f /tmp/whispertyper-server.err"
fi
