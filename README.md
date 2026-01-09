# WhisperLive with ydotool for NixOS

Real-time speech-to-text that types directly using keyboard simulation.
Uses [WhisperLive](https://github.com/collabora/WhisperLive) with
[CTranslate2-ROCm](https://github.com/arlo-phoenix/CTranslate2-rocm) for
AMD GPU acceleration.

## Architecture

```
Microphone → Client (NixOS) → WebSocket → Server (Docker/ROCm) → GPU
                ↓
           ydotool → Keyboard Input → Any Application
```

**Key feature**: When the transcription updates (e.g., correcting a word),
the client automatically sends backspace keypresses to fix the text.

## Prerequisites

- Docker or Podman with compose
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- AMD GPU (RX 7000 series tested)
- ydotool daemon running
- portaudio (for microphone access)

## Quick Start

```bash
# Run the setup script (builds Docker, configures hotkey)
./scripts/setup.sh

# Edit config to set your microphone
nano ~/.config/whisper-typer/config

# Press Super+H to toggle speech-to-text!
```

## Detailed Setup

### 1. Determine your GPU architecture

```bash
# Check your GPU
rocminfo | grep "gfx"
```

| GPU | Architecture | HSA_OVERRIDE_GFX_VERSION |
|-----|--------------|--------------------------|
| RX 7900 XTX/XT | gfx1100 | 11.0.0 |
| RX 7800 XT, 7700 XT | gfx1101 | 11.0.0 |
| RX 7600 | gfx1102 | 11.0.0 |
| RX 6900/6800/6700 XT | gfx1030 | 10.3.0 |
| RX 6600 XT | gfx1032 | 10.3.0 |

### 2. Configure docker-compose.yml

Edit `docker-compose.yml` and set your GPU architecture:

```yaml
args:
  PYTORCH_ROCM_ARCH: gfx1100  # Change to match your GPU
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"
```

### 3. Build and start the server

```bash
# Enter the development shell
nix develop

# Build and start the server (first build takes a while - ~30 min)
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

### 4. Start ydotool daemon

```bash
# The daemon needs to run as your user
# Add to your NixOS configuration:
programs.ydotool.enable = true;

# Or run manually:
ydotoold &
```

### 5. Run the client

```bash
# Using the flake
nix run . -- --host localhost --port 9090

# Or in dev shell
python whisper_typer.py --host localhost --port 9090

# Dry run (logs what would be typed)
python whisper_typer.py --dry-run --verbose
```

## Hotkey Usage

**Super+H** toggles speech-to-text:

1. Press `Super+H` → notification "Listening..." appears
2. Speak → words are typed at cursor position
3. Press `Super+H` again → notification "Stopped" appears

The toggle script is at `./scripts/toggle.sh`. Config file: `~/.config/whisper-typer/config`

## CLI Usage

```
./whisper_typer.py [OPTIONS]

Options:
  -H, --host HOST       Server host (default: localhost)
  -p, --port PORT       Server port (default: 9090)
  -l, --language LANG   Language code (default: en)
  -m, --model MODEL     Whisper model size (default: small)
  -d, --device DEVICE   Microphone (index or name, e.g., "C930e")
  -L, --list-devices    List available microphones
  -s, --socket PATH     ydotool socket path
  -n, --dry-run         Don't type, just log
  -v, --verbose         Enable debug logging
```

The script uses `uv` to automatically manage Python dependencies.

## Server Configuration

Environment variables for the Docker container:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | small | Model size: tiny, base, small, medium, large |
| `WHISPER_LANGUAGE` | en | Language code |
| `WHISPER_PORT` | 9090 | WebSocket port |
| `WHISPER_MAX_CLIENTS` | 4 | Maximum concurrent clients |
| `WHISPER_MAX_CONNECTION_TIME` | 600 | Max session duration (seconds) |

## Troubleshooting

### GPU not detected

```bash
# Check ROCm can see your GPU
docker-compose exec whisper-server rocminfo

# Check PyTorch CUDA (ROCm) availability
docker-compose exec whisper-server python -c "import torch; print(torch.cuda.is_available())"
```

### ydotool not typing

```bash
# Check the daemon is running
pgrep ydotoold

# Check socket exists
ls -la /run/user/$(id -u)/.ydotool_socket

# Test ydotool directly
ydotool type "hello world"
```

### Build fails

The CTranslate2-ROCm build requires ~100GB disk space and 30+ minutes.
Ensure you have enough resources.

## How It Works

1. **Server**: WhisperLive runs in a Docker container with ROCm-enabled
   CTranslate2 for fast GPU inference using faster-whisper.

2. **Client**: Captures microphone audio, streams it via WebSocket to the
   server, receives transcription segments.

3. **Typing**: The client tracks what has been typed. When a segment updates
   (common in streaming transcription), it calculates the diff and uses
   backspace to correct, then types the new text.

4. **Finalization**: When a segment is marked "completed" (speech pause
   detected), a space is added and the client moves to the next segment.

## References

- [WhisperLive](https://github.com/collabora/WhisperLive)
- [CTranslate2-ROCm](https://github.com/arlo-phoenix/CTranslate2-rocm)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [wyoming-faster-whisper-rocm](https://github.com/Donkey545/wyoming-faster-whisper-rocm)
- [AMD ROCm Whisper Blog](https://rocm.blogs.amd.com/artificial-intelligence/whisper/README.html)
