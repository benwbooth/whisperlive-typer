#!/bin/bash
# WhisperLive server startup script

set -e

# Configuration via environment variables
MODEL=${WHISPER_MODEL:-small}
LANGUAGE=${WHISPER_LANGUAGE:-en}
PORT=${WHISPER_PORT:-9090}
MAX_CLIENTS=${WHISPER_MAX_CLIENTS:-4}
MAX_CONNECTION_TIME=${WHISPER_MAX_CONNECTION_TIME:-600}

echo "=== WhisperLive Server with ROCm ==="
echo "Model: $MODEL"
echo "Language: $LANGUAGE"
echo "Port: $PORT"
echo "Max clients: $MAX_CLIENTS"
echo "GPU Architecture: $PYTORCH_ROCM_ARCH"
echo "HSA Version Override: $HSA_OVERRIDE_GFX_VERSION"

# Check GPU availability
echo ""
echo "Checking GPU..."
python3 -c "
import torch
import ctranslate2

print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}')
"

echo ""
echo "Starting WhisperLive server..."

cd /app/WhisperLive

exec python3 run_server.py \
    --port "$PORT" \
    --backend faster_whisper \
    --max_clients "$MAX_CLIENTS" \
    --max_connection_time "$MAX_CONNECTION_TIME"
