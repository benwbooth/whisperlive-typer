# WhisperLive Typer

Real-time speech-to-text that types directly into any application using keyboard simulation.
Uses [WhisperLive](https://github.com/collabora/WhisperLive) server with AMD ROCm GPU acceleration.

## Features

- **Real-time typing**: Speaks → types at cursor position in any app
- **Auto-correction**: Backspaces and fixes text as transcription updates
- **Voice commands**: "press enter", "scratch that", "new line", etc.
- **Configurable**: YAML config for commands, VAD thresholds, server settings
- **VAD calibration**: Tool to optimize voice detection for your environment
- **KDE hotkey**: Super+H to toggle on/off

## Architecture

```
Microphone → Client (whisper_typer.py) → WebSocket → Server (Docker/ROCm) → GPU
                      ↓
                 ydotool → Keyboard Input → Any Application
```

## Prerequisites

- Docker or Podman with compose
- AMD GPU with ROCm support (RX 6000/7000 series)
- [Nix](https://nixos.org/) with flakes enabled
- ydotool daemon running

## Quick Start

```bash
# Enter development shell
nix develop

# Run setup (builds Docker image, configures KDE hotkey)
./scripts/setup.sh

# Calibrate VAD for your microphone/environment
./scripts/vad_calibrate.py --auto

# Press Super+H to toggle speech-to-text!
```

## Configuration

Config file: `~/.config/whisper-typer/config.yaml`

```yaml
server:
  host: localhost
  port: 9090

whisper:
  language: en
  model: small

audio:
  device: ""  # or "C930e" or device index

vad:
  onset: 0.3   # speech detection threshold
  offset: 0.2  # speech end threshold

commands:
  keys:
    "scratch that": "ctrl+z"
    "new line": "enter"
    # add your own...
  literals:
    # "smiley face": ":)"
```

## Voice Commands

**Modifier combos** - say modifier + key:
- `"control c"` → Ctrl+C
- `"alt f4"` → Alt+F4
- `"control shift t"` → Ctrl+Shift+T

**Action keys** - say the key name:
- `"enter"`, `"tab"`, `"escape"`, `"backspace"`, `"delete"`
- `"home"`, `"end"`, `"page up"`, `"page down"`
- `"up"`, `"down"`, `"left"`, `"right"`
- `"f1"` through `"f12"`

**Punctuation** - say the name to type the symbol:
- `"semicolon"` → `;`, `"colon"` → `:`
- `"hyphen"` → `-`, `"underscore"` → `_`
- `"apostrophe"` → `'`, `"double quote"` → `"`
- `"at sign"` → `@`, `"hash sign"` → `#`
- `"left paren"` → `(`, `"right paren"` → `)`
- See config.yaml.example for full list

**Escape hatch** - type literally:
- `"say enter"` → types "enter" (the word)
- `"say control c"` → types "control c"

**Configured shortcuts**:
- `"scratch that"` / `"undo"` → Ctrl+Z
- `"new line"` → Enter
- `"new paragraph"` → Enter Enter
- Add your own in config.yaml

## VAD Calibration

The VAD (Voice Activity Detection) determines when you're speaking vs silence.
Calibrate it for your environment:

```bash
./scripts/vad_calibrate.py --auto
```

This measures your silence/speech levels and saves optimal thresholds to config.

## CLI Usage

```bash
./whisper_typer.py [OPTIONS]

Options:
  -c, --config PATH     Config file (default: ~/.config/whisper-typer/config.yaml)
  -H, --host HOST       Server host
  -p, --port PORT       Server port
  -l, --language LANG   Language code
  -m, --model MODEL     Whisper model size
  -d, --device DEVICE   Microphone (index or name substring)
  -L, --list-devices    List available microphones
  --vad-onset FLOAT     VAD onset threshold (0.0-1.0)
  --vad-offset FLOAT    VAD offset threshold (0.0-1.0)
  -n, --dry-run         Don't type, just log
  -v, --verbose         Debug logging
```

## GPU Setup

### 1. Check your GPU architecture

```bash
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

```yaml
args:
  PYTORCH_ROCM_ARCH: gfx1100  # Change to match your GPU
  HSA_OVERRIDE_GFX_VERSION: "11.0.0"
```

### 3. Build and start server

```bash
docker compose up -d --build  # First build takes ~30 min
docker compose logs -f
```

## ydotool Setup

ydotool is required for keyboard simulation (works on Wayland):

```bash
# NixOS: add to configuration.nix
programs.ydotool.enable = true;

# Or run daemon manually
sudo ydotoold &

# Socket should be at /run/ydotoold/socket
```

## Troubleshooting

### GPU not detected

```bash
docker compose exec whisper-server rocminfo
docker compose exec whisper-server python -c "import torch; print(torch.cuda.is_available())"
```

### ydotool not typing

```bash
# Check daemon is running
pgrep ydotoold

# Check socket exists
ls -la /run/ydotoold/socket

# Test directly
ydotool type "hello"
```

### VAD too sensitive / not sensitive enough

Run the calibrator: `./scripts/vad_calibrate.py --auto`

Or manually adjust `vad.onset` in config:
- Lower (0.1-0.2) = more sensitive, may trigger on noise
- Higher (0.4-0.5) = less sensitive, may miss quiet speech

### Whisper hallucinations

If Whisper types random phrases ("Bye", "Thank you") during silence:
- Increase `vad.onset` threshold
- Use a directional microphone
- Reduce background noise

## References

- [WhisperLive](https://github.com/collabora/WhisperLive)
- [CTranslate2-ROCm](https://github.com/arlo-phoenix/CTranslate2-rocm)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
