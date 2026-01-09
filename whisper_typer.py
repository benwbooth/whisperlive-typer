#!/usr/bin/env -S uv run
"""
WhisperLive client that types transcriptions using ydotool.

Connects to a WhisperLive server, streams microphone audio,
and types the transcribed text in real-time with backspace
corrections when the transcription is updated.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keyboard scan codes for ydotool
# Reference: Linux input-event-codes.h
KEY_CODES = {
    # Special keys
    "escape": 1, "esc": 1,
    "backspace": 14, "back": 14,
    "tab": 15,
    "enter": 28, "return": 28,
    "space": 57,
    "delete": 111, "del": 111,
    # Modifiers
    "ctrl": 29, "control": 29, "leftctrl": 29,
    "shift": 42, "leftshift": 42,
    "alt": 56, "leftalt": 56,
    "super": 125, "meta": 125, "windows": 125, "win": 125,
    "rightctrl": 97, "rightshift": 54, "rightalt": 100,
    # Navigation
    "home": 102, "end": 107,
    "pageup": 104, "pgup": 104,
    "pagedown": 109, "pgdn": 109,
    "up": 103, "down": 108, "left": 105, "right": 106,
    "insert": 110, "ins": 110,
    # Function keys
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63, "f6": 64,
    "f7": 65, "f8": 66, "f9": 67, "f10": 68, "f11": 87, "f12": 88,
    # Numbers (top row)
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
    # Letters
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
    "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
    "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
    "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    # Punctuation
    "minus": 12, "equal": 13, "equals": 13,
    "leftbracket": 26, "rightbracket": 27,
    "semicolon": 39, "apostrophe": 40, "quote": 40,
    "grave": 41, "backslash": 43,
    "comma": 51, "period": 52, "dot": 52, "slash": 53,
}

# Default voice commands
DEFAULT_COMMANDS = {
    "keys": {
        "scratch that": "ctrl+z",
        "undo": "ctrl+z",
        "undo that": "ctrl+z",
        "redo": "ctrl+shift+z",
        "redo that": "ctrl+shift+z",
        "new line": "enter",
        "newline": "enter",
        "new paragraph": "enter enter",
        "go back": "backspace",
        "delete word": "ctrl+backspace",
        "delete line": "ctrl+shift+k",
        "select all": "ctrl+a",
        "copy": "ctrl+c",
        "copy that": "ctrl+c",
        "paste": "ctrl+v",
        "paste that": "ctrl+v",
        "cut": "ctrl+x",
        "cut that": "ctrl+x",
        "save": "ctrl+s",
        "save that": "ctrl+s",
    },
    "literals": {
        # Example literal strings - user can customize
    },
}


@dataclass
class CommandResult:
    """Result of processing a voice command."""
    action: str  # "keys", "literal", "type_escape", "none"
    payload: str  # key sequence, literal text, or original text


CONFIG_PATH = Path.home() / ".config" / "whisper-typer" / "config.yaml"


@dataclass
class Config:
    """Unified configuration loaded from YAML."""
    host: str = "localhost"
    port: int = 9090
    language: str = "en"
    model: str = "small"
    device: str = ""
    vad_onset: float = 0.3
    vad_offset: float = 0.2
    commands: dict = None

    def __post_init__(self):
        if self.commands is None:
            self.commands = DEFAULT_COMMANDS.copy()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load config from YAML file, falling back to defaults."""
        path = path or CONFIG_PATH
        config = cls()

        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}

                # Server settings
                server = data.get("server", {})
                config.host = server.get("host", config.host)
                config.port = server.get("port", config.port)

                # Whisper settings
                whisper = data.get("whisper", {})
                config.language = whisper.get("language", config.language)
                config.model = whisper.get("model", config.model)

                # Audio settings
                audio = data.get("audio", {})
                config.device = audio.get("device", config.device)

                # VAD settings
                vad = data.get("vad", {})
                config.vad_onset = vad.get("onset", config.vad_onset)
                config.vad_offset = vad.get("offset", config.vad_offset)

                # Commands
                commands_data = data.get("commands", {})
                config.commands = {
                    "keys": DEFAULT_COMMANDS["keys"].copy(),
                    "literals": DEFAULT_COMMANDS["literals"].copy(),
                }
                if "keys" in commands_data:
                    config.commands["keys"].update(commands_data["keys"])
                if "literals" in commands_data:
                    config.commands["literals"].update(commands_data["literals"])

                logger.info(f"Loaded config from {path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return config

    def save_vad(self, onset: float, offset: float, path: Optional[Path] = None):
        """Update VAD settings in config file."""
        path = path or CONFIG_PATH

        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        if "vad" not in data:
            data["vad"] = {}
        data["vad"]["onset"] = onset
        data["vad"]["offset"] = offset

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved VAD settings to {path}")


class CommandProcessor:
    """Processes voice commands from transcribed text."""

    def __init__(self, commands: Optional[dict] = None):
        self.commands = commands or DEFAULT_COMMANDS

    def process(self, text: str) -> CommandResult:
        """
        Process text to detect voice commands.

        Args:
            text: Transcribed text (start of segment)

        Returns:
            CommandResult with action type and payload
        """
        text_lower = text.lower().strip()

        # 1. Check for "type X" escape (literal typing)
        if text_lower.startswith("type "):
            literal = text[5:]  # Preserve original case
            return CommandResult("type_escape", literal)

        # 2. Check for "press X" key command
        if text_lower.startswith("press "):
            keys = text_lower[6:].strip()
            return CommandResult("keys", self._normalize_keys(keys))

        # 3. Check for configured key commands
        for cmd, keys in self.commands["keys"].items():
            if text_lower == cmd or text_lower.startswith(cmd + " "):
                return CommandResult("keys", keys)

        # 4. Check for configured literal commands
        for cmd, literal in self.commands["literals"].items():
            if text_lower == cmd or text_lower.startswith(cmd + " "):
                return CommandResult("literal", literal)

        # 5. Not a command - normal dictation
        return CommandResult("none", text)

    def _normalize_keys(self, keys: str) -> str:
        """Normalize key names to standard format (e.g., 'control c' -> 'ctrl+c')."""
        # Strip trailing punctuation from the whole string first
        keys = keys.rstrip('.,!?')
        parts = keys.split()
        normalized = []
        for part in parts:
            # Normalize common variations and strip punctuation
            part = part.lower().rstrip('.,!?')
            if part == "control":
                part = "ctrl"
            elif part in ("windows", "meta", "win"):
                part = "super"
            if part:  # Skip empty parts
                normalized.append(part)
        return "+".join(normalized)


class Notifier:
    """Desktop notification that can be shown and closed."""

    def __init__(self):
        self.notification_id = None

    def show(self, message: str, persistent: bool = False):
        """Show notification. If persistent, stays until closed."""
        try:
            cmd = ['notify-send', '--app-name=Whisper Typer', '--print-id']
            if persistent:
                cmd.extend(['--expire-time=0'])  # Never auto-close
            if self.notification_id:
                cmd.extend(['--replace-id', str(self.notification_id)])
            cmd.append(message)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout.strip():
                self.notification_id = int(result.stdout.strip())
        except (FileNotFoundError, ValueError):
            pass

    def close(self):
        """Close the current notification."""
        if self.notification_id:
            try:
                subprocess.run([
                    'gdbus', 'call', '--session',
                    '--dest', 'org.freedesktop.Notifications',
                    '--object-path', '/org/freedesktop/Notifications',
                    '--method', 'org.freedesktop.Notifications.CloseNotification',
                    str(self.notification_id)
                ], capture_output=True)
                self.notification_id = None
            except FileNotFoundError:
                pass


@dataclass
class TyperState:
    """Tracks what has been typed to enable corrections."""
    confirmed_text: str = ""  # Text from completed segments (won't change)
    pending_text: str = ""     # Text from current in-progress segment
    completed_count: int = 0   # Number of completed segments processed
    last_command_text: str = ""  # Last command text we executed (to prevent re-execution)


class YdotoolTyper:
    """Handles typing via ydotool with backspace correction support."""

    def __init__(self, socket_path: Optional[str] = None, dry_run: bool = False):
        self.socket_path = socket_path or os.environ.get(
            'YDOTOOL_SOCKET',
            '/run/ydotoold/socket'
        )
        self.dry_run = dry_run
        self.state = TyperState()
        self._lock = threading.Lock()

    def _run_ydotool(self, *args):
        """Execute ydotool command."""
        env = os.environ.copy()
        env['YDOTOOL_SOCKET'] = self.socket_path

        cmd = ['ydotool'] + list(args)

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return

        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ydotool error: {e.stderr.decode()}")
        except FileNotFoundError:
            logger.error("ydotool not found. Is it installed?")

    def type_text(self, text: str):
        """Type text using ydotool."""
        if not text:
            return
        logger.info(f"Typing: {text!r}")
        # ydotool type command with minimal delays for speed
        self._run_ydotool('type', '--key-delay=2', '--key-hold=2', '--clearmodifiers', '--', text)

    def backspace(self, count: int):
        """Send backspace key presses."""
        if count <= 0:
            return
        # Key code 14 is backspace - send all at once for speed
        # Format: keycode:1 (down) keycode:0 (up)
        keys = ' '.join(['14:1', '14:0'] * count)
        self._run_ydotool('key', '--key-delay=2', *keys.split())

    def send_keys(self, keys_str: str):
        """
        Send a key combination.

        Args:
            keys_str: Key combo like "ctrl+c", "ctrl+shift+t", "enter", "enter enter"
        """
        # Handle multiple key sequences separated by space (e.g., "enter enter")
        sequences = keys_str.strip().split()
        if not sequences:
            return

        # Group into combos (consecutive keys with + are one combo)
        combos = []
        current_combo = []
        for seq in sequences:
            if '+' in seq:
                # This is a combo like ctrl+c
                if current_combo:
                    combos.extend(current_combo)
                    current_combo = []
                combos.append(seq)
            else:
                # Single key, might be part of "enter enter" sequence
                combos.append(seq)

        for combo in combos:
            self._send_key_combo(combo)

    def _send_key_combo(self, combo: str):
        """Send a single key combination like 'ctrl+c' or 'enter'."""
        # Strip punctuation that Whisper might add
        combo = combo.lower().rstrip('.,!?')
        parts = combo.split('+')
        codes = []

        for part in parts:
            part = part.strip().rstrip('.,!?')
            if part in KEY_CODES:
                codes.append(KEY_CODES[part])
            else:
                logger.warning(f"Unknown key: {part}")
                return

        if not codes:
            return

        logger.info(f"Sending keys: {combo}")

        # Build key sequence: press all modifiers, press key, release all in reverse
        key_args = []
        # Press all keys down
        for code in codes:
            key_args.append(f"{code}:1")
        # Release all keys up (reverse order)
        for code in reversed(codes):
            key_args.append(f"{code}:0")

        self._run_ydotool('key', '--key-delay=2', *key_args)

    def update_pending(self, new_text: str):
        """
        Update the pending (in-progress) text, using backspaces to correct.

        Args:
            new_text: The new transcription text for the current segment
        """
        with self._lock:
            old_text = self.state.pending_text

            if old_text == new_text:
                # No change
                return

            if not old_text:
                # Nothing pending yet, just type the new text
                if new_text:
                    self.type_text(new_text)
                    self.state.pending_text = new_text
                    logger.debug(f"Typed new: {new_text!r}")
                return

            # Find common prefix length
            common_len = 0
            min_len = min(len(old_text), len(new_text))
            for i in range(min_len):
                if old_text[i] == new_text[i]:
                    common_len = i + 1
                else:
                    break

            # Calculate what needs to be deleted and added
            chars_to_delete = len(old_text) - common_len
            chars_to_add = new_text[common_len:]

            if chars_to_delete > 0:
                logger.debug(f"Backspacing {chars_to_delete} chars")
                self.backspace(chars_to_delete)

            if chars_to_add:
                logger.debug(f"Typing: {chars_to_add!r}")
                self.type_text(chars_to_add)

            self.state.pending_text = new_text

    def finalize_pending(self):
        """Finalize the current pending text (segment completed)."""
        with self._lock:
            if self.state.pending_text:
                # Add space after completed segment
                self.type_text(" ")
                self.state.confirmed_text += self.state.pending_text + " "
                logger.info(f"Finalized: {self.state.pending_text!r}")
                self.state.pending_text = ""
            self.state.completed_count += 1

    def clear_pending(self):
        """Clear pending text without finalizing (e.g., segment was empty)."""
        with self._lock:
            if self.state.pending_text:
                # Backspace to remove unfinalied text
                self.backspace(len(self.state.pending_text))
                self.state.pending_text = ""


class MicrophoneStream:
    """Streams audio from the microphone using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.stream = None
        self._running = False
        self._queue: queue.Queue = queue.Queue()

    @classmethod
    def list_devices(cls) -> list[dict]:
        """List available input devices."""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': int(dev['default_samplerate']),
                })
        return devices

    @classmethod
    def find_device(cls, query: str) -> Optional[int]:
        """
        Find a device by index or name substring.

        Args:
            query: Device index (as string) or name substring (case-insensitive)

        Returns:
            Device index if found, None otherwise
        """
        devices = cls.list_devices()

        # Try as numeric index first
        try:
            index = int(query)
            if any(d['index'] == index for d in devices):
                return index
        except ValueError:
            pass

        # Search by name substring (case-insensitive)
        query_lower = query.lower()
        matches = [d for d in devices if query_lower in d['name'].lower()]

        if len(matches) == 1:
            return matches[0]['index']
        elif len(matches) > 1:
            print(f"Multiple devices match '{query}':")
            for d in matches:
                print(f"  [{d['index']}] {d['name']}")
            print("Please be more specific or use the index number.")
            return None
        else:
            print(f"No device found matching '{query}'")
            print("Use --list-devices to see available devices.")
            return None

    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        # Send float32 audio directly (WhisperLive expects float32 in -1 to 1 range)
        self._queue.put(indata.astype(np.float32).tobytes())

    def start(self):
        """Start the audio stream."""
        if self.device_index is not None:
            dev_info = sd.query_devices(self.device_index)
            logger.info(f"Using device: {dev_info['name']}")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device_index,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()
        self._running = True
        logger.info("Microphone stream started")

    def stop(self):
        """Stop the audio stream."""
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("Microphone stream stopped")

    def read_chunk(self) -> bytes:
        """Read a chunk of audio data."""
        if not self._running:
            return b''
        try:
            return self._queue.get(timeout=0.1)
        except queue.Empty:
            return b''

    @property
    def running(self) -> bool:
        return self._running


class WhisperTyperClient:
    """
    Main client that connects WhisperLive to ydotool.
    """

    def __init__(
        self,
        config: Config,
        typer: Optional[YdotoolTyper] = None,
        dry_run: bool = False,
        device_index: Optional[int] = None,
    ):
        self.config = config
        self.host = config.host
        self.port = config.port
        self.language = config.language
        self.model = config.model
        self.vad_onset = config.vad_onset
        self.vad_offset = config.vad_offset
        self.typer = typer or YdotoolTyper(dry_run=dry_run)
        self.mic = MicrophoneStream(device_index=device_index)
        self.notifier = Notifier()
        self.commands = CommandProcessor(config.commands)
        self._running = False
        self._websocket = None

    async def connect(self):
        """Connect to the WhisperLive server."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to {uri}...")

        self._websocket = await websockets.connect(uri)

        # Send initial configuration
        # VAD parameters: lower onset = more sensitive to speech
        # Silero VAD thresholds are probability-based (0-1)
        config = {
            "uid": os.getpid(),
            "language": self.language,
            "task": "transcribe",
            "model": self.model,
            "use_vad": True,
            "vad_parameters": {
                "onset": self.vad_onset,
                "offset": self.vad_offset,
                "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 100,
            },
        }
        await self._websocket.send(json.dumps(config))
        logger.info("Connected and configured")

    async def run(self):
        """Main run loop."""
        self._running = True
        self.notifier.show("🎤 Listening...", persistent=True)

        try:
            await self.connect()
            self.mic.start()

            # Start tasks for sending audio and receiving transcriptions
            send_task = asyncio.create_task(self._send_audio())
            recv_task = asyncio.create_task(self._receive_transcriptions())

            # Wait for either task to complete (usually due to disconnection)
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.stop()

    async def _send_audio(self):
        """Send audio chunks to the server."""
        while self._running and self.mic.running:
            chunk = self.mic.read_chunk()
            if chunk and self._websocket:
                try:
                    await self._websocket.send(chunk)
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    break
            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

    async def _receive_transcriptions(self):
        """Receive and process transcriptions."""
        while self._running and self._websocket:
            try:
                message = await self._websocket.recv()
                logger.info(f"Received: {message[:500]}")  # Debug: log received messages
                data = json.loads(message)
                self._handle_transcription(data)
            except websockets.exceptions.ConnectionClosed:
                break
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON")
            except Exception as e:
                logger.error(f"Error receiving: {e}")
                break

    def _handle_transcription(self, data: dict):
        """Process a transcription message from the server."""
        logger.debug(f"Handling data: {data}")
        if "segments" not in data:
            logger.debug("No segments in data")
            return

        segments = data.get("segments", [])
        if not segments:
            logger.debug("Empty segments list")
            return

        # Count completed segments
        completed_segments = [s for s in segments if s.get("completed", False)]
        pending_segments = [s for s in segments if not s.get("completed", False)]

        num_completed = len(completed_segments)
        prev_completed = self.typer.state.completed_count

        # Process newly completed segments
        if num_completed > prev_completed:
            # Check if pending text is a command before finalizing
            pending_text = self.typer.state.pending_text.strip()
            if pending_text:
                cmd_result = self.commands.process(pending_text)
                if cmd_result.action != "none":
                    # It's a command - clear pending and execute
                    self.typer.clear_pending()
                    self._execute_command(cmd_result)
                else:
                    # Normal text - finalize it
                    self.typer.finalize_pending()
            # Update our count and reset command tracking for next segment
            self.typer.state.completed_count = num_completed
            self.typer.state.last_command_text = ""
            logger.debug(f"Completed segments: {prev_completed} -> {num_completed}")

        # Handle the current in-progress segment
        if pending_segments:
            current = pending_segments[-1]
            text = current.get("text", "").strip()
            logger.info(f"Segment: text={text!r}, completed=False")

            # Skip if we already executed a command for this text
            # (prevents re-execution as segment updates with same/similar content)
            # Normalize by lowercasing and stripping punctuation for comparison
            def normalize_cmd(s: str) -> str:
                return s.lower().rstrip('.,!?')

            if self.typer.state.last_command_text and normalize_cmd(text) == normalize_cmd(self.typer.state.last_command_text):
                logger.debug(f"Skipping already-executed command: {text!r}")
                return

            # Check if text is a command (for immediate execution of some commands)
            cmd_result = self.commands.process(text)
            if cmd_result.action == "keys" or cmd_result.action == "type_escape":
                # Execute key commands and type escapes immediately
                # Clear any existing pending text first
                if self.typer.state.pending_text:
                    self.typer.clear_pending()
                self._execute_command(cmd_result)
                # Track that we executed this command to prevent re-execution
                self.typer.state.last_command_text = text
            elif cmd_result.action == "literal":
                # Literals also execute immediately
                if self.typer.state.pending_text:
                    self.typer.clear_pending()
                self._execute_command(cmd_result)
                self.typer.state.last_command_text = text
            else:
                # Normal dictation - update pending
                self.typer.update_pending(text)

    def _execute_command(self, cmd_result: CommandResult):
        """Execute a voice command."""
        logger.info(f"Executing command: {cmd_result.action} -> {cmd_result.payload!r}")

        if cmd_result.action == "keys":
            self.typer.send_keys(cmd_result.payload)
        elif cmd_result.action == "type_escape":
            self.typer.type_text(cmd_result.payload)
        elif cmd_result.action == "literal":
            self.typer.type_text(cmd_result.payload)

    def stop(self):
        """Stop the client."""
        self._running = False
        self.mic.stop()
        if self._websocket:
            asyncio.create_task(self._websocket.close())
        self.notifier.close()
        logger.info("Client stopped")


def main():
    parser = argparse.ArgumentParser(
        description="WhisperLive client that types transcriptions via ydotool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Config file: {CONFIG_PATH}
See config.yaml.example for format.

CLI arguments override config file settings.
"""
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help=f"Config file path (default: {CONFIG_PATH})"
    )
    parser.add_argument(
        "--host", "-H",
        default=None,
        help="WhisperLive server host"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="WhisperLive server port"
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Whisper model size"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="Audio input device: index number or name substring (e.g., 'C930e' or '4')"
    )
    parser.add_argument(
        "--list-devices", "-L",
        action="store_true",
        help="List available audio input devices and exit"
    )
    parser.add_argument(
        "--socket", "-s",
        default=None,
        help="ydotool socket path (default: $YDOTOOL_SOCKET or /run/ydotoold/socket)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't actually type, just log what would be typed"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=None,
        help="VAD onset threshold (0.0-1.0). Lower = more sensitive."
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=None,
        help="VAD offset threshold (0.0-1.0). Lower = longer speech segments."
    )

    args = parser.parse_args()

    # Handle --list-devices first (doesn't need config)
    if args.list_devices:
        print("Available audio input devices:")
        print("-" * 60)
        devices = MicrophoneStream.list_devices()
        for dev in devices:
            print(f"  [{dev['index']}] \"{dev['name']}\"")
            print(f"      Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']} Hz")
        print("-" * 60)
        print("Use --device \"<name>\" or --device <index>")
        sys.exit(0)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config from file
    config = Config.load(args.config)

    # CLI args override config
    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.language is not None:
        config.language = args.language
    if args.model is not None:
        config.model = args.model
    if args.vad_onset is not None:
        config.vad_onset = args.vad_onset
    if args.vad_offset is not None:
        config.vad_offset = args.vad_offset

    # Resolve device - CLI arg or config
    device_str = args.device if args.device is not None else config.device
    device_index = None
    if device_str:
        device_index = MicrophoneStream.find_device(device_str)
        if device_index is None:
            sys.exit(1)

    typer = YdotoolTyper(socket_path=args.socket, dry_run=args.dry_run)

    client = WhisperTyperClient(
        config=config,
        typer=typer,
        dry_run=args.dry_run,
        device_index=device_index,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        client.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 50)
    print("WhisperLive Typer")
    print("=" * 50)
    print(f"Server: {config.host}:{config.port}")
    print(f"Language: {config.language}")
    print(f"Model: {config.model}")
    print(f"VAD: onset={config.vad_onset}, offset={config.vad_offset}")
    if device_index is not None:
        devices = MicrophoneStream.list_devices()
        dev_name = next((d['name'] for d in devices if d['index'] == device_index), f"Device {device_index}")
        print(f"Microphone: [{device_index}] {dev_name}")
    else:
        print("Microphone: System default")
    if args.dry_run:
        print("Mode: DRY RUN (not typing)")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    asyncio.run(client.run())


if __name__ == "__main__":
    main()
