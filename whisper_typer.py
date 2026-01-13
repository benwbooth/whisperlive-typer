#!/usr/bin/env -S uv run
"""
WhisperLive client that types transcriptions using ydotool.

Connects to a WhisperLive server, streams microphone audio,
and types the transcribed text in real-time with backspace
corrections when the transcription is updated.
"""

import argparse
import asyncio
import atexit
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets
import yaml

# Try to import grapheme for proper Unicode grapheme cluster counting
# Falls back to len() if not available (pip install grapheme)
try:
    import grapheme
    def grapheme_len(s: str) -> int:
        """Count grapheme clusters (user-perceived characters) in a string."""
        return grapheme.length(s)

    def grapheme_slice(s: str, start: int, end: int = None) -> str:
        """Slice string by grapheme clusters."""
        if end is None:
            return "".join(grapheme.graphemes(s)[start:])
        return "".join(list(grapheme.graphemes(s))[start:end])

    def grapheme_iter(s: str):
        """Iterate over grapheme clusters in a string."""
        return grapheme.graphemes(s)
except ImportError:
    def grapheme_len(s: str) -> int:
        """Fallback: count codepoints (may be wrong for complex emoji/combining chars)."""
        return len(s)

    def grapheme_slice(s: str, start: int, end: int = None) -> str:
        """Fallback: slice by codepoints."""
        if end is None:
            return s[start:]
        return s[start:end]

    def grapheme_iter(s: str):
        """Fallback: iterate over codepoints."""
        return iter(s)

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

# Modifier keys that indicate a key combo
MODIFIER_KEYS = {"ctrl", "control", "alt", "shift", "super", "meta", "windows", "win"}

# Punctuation/symbol names to characters (for typing, not key codes)
# Use distinct phrases to avoid accidental triggers during normal speech
PUNCTUATION_CHARS = {
    "hyphen": "-",
    "underscore": "_",
    "plus sign": "+",
    "equals sign": "=",
    "semicolon": ";",
    "colon": ":",
    "comma": ",",
    "period": ".",
    "forward slash": "/",
    "backslash": "\\",
    "pipe": "|",
    "apostrophe": "'", "single quote": "'",
    "double quote": '"', "quotation mark": '"',
    "backtick": "`",
    "tilde": "~",
    "at sign": "@",
    "hash sign": "#", "hashtag": "#",
    "dollar sign": "$",
    "percent sign": "%",
    "caret": "^",
    "ampersand": "&",
    "asterisk": "*",
    "left paren": "(", "open paren": "(",
    "right paren": ")", "close paren": ")",
    "left bracket": "[", "open bracket": "[",
    "right bracket": "]", "close bracket": "]",
    "left brace": "{", "open brace": "{",
    "right brace": "}", "close brace": "}",
    "less than": "<",
    "greater than": ">",
    "question mark": "?",
    "exclamation point": "!",
}

# Single letters (typed immediately when spoken alone)
SINGLE_LETTERS = {c: c for c in "abcdefghijklmnopqrstuvwxyz"}

# Digit words -> numbers
DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Action keys (these send key codes, not characters)
ACTION_KEYS = {
    "enter", "return", "tab", "escape", "esc", "backspace", "back",
    "delete", "del", "home", "end", "page up", "pageup", "page down", "pagedown",
    "up", "down", "left", "right", "insert", "ins",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
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
    action: str  # "keys", "literal", "none"
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
    no_speech_thresh: float = 0.45  # Segments with no_speech_prob above this are filtered (hallucinations)
    min_avg_logprob: float = -0.8  # Segments with avg_logprob below this are filtered (low confidence)
    pending_debounce_ms: int = 200  # Debounce pending updates to avoid excessive retyping
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
                config.no_speech_thresh = vad.get("no_speech_thresh", config.no_speech_thresh)
                config.min_avg_logprob = vad.get("min_avg_logprob", config.min_avg_logprob)

                # Typer settings
                typer = data.get("typer", {})
                config.pending_debounce_ms = typer.get("pending_debounce_ms", config.pending_debounce_ms)

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

        Command priority:
        1. "say X" -> type X literally (escape hatch)
        2. Configured key commands (e.g., "scratch that" -> ctrl+z)
        3. Configured literals
        4. Modifier combos (e.g., "control c" -> ctrl+c)
        5. "press X" (e.g., "press enter", "press tab")
        6. Action keys (e.g., "enter", "backspace")
        7. Punctuation names (e.g., "semicolon" -> ;)
        8. Spelled-out letters/digits (e.g., "A-B-C" or "A B C" -> abc)
        9. Single letters (e.g., "a" -> a, "b" -> b)
        10. Digit words (e.g., "one" -> 1, "two" -> 2)
        11. Normal dictation
        """
        text_lower = text.lower().strip().rstrip('.,!?')
        text_clean = text.strip().rstrip('.,!?')

        # 1. "say X" escape - type literally
        if text_lower.startswith("say "):
            literal = text_clean[4:]  # Preserve original case
            return CommandResult("literal", literal)

        # 2. Configured key commands (e.g., "scratch that")
        for cmd, keys in self.commands["keys"].items():
            if text_lower == cmd:
                return CommandResult("keys", keys)

        # 3. Configured literals
        for cmd, literal in self.commands["literals"].items():
            if text_lower == cmd:
                return CommandResult("literal", literal)

        # 4. Check for modifier combo (e.g., "control c", "alt f4")
        words = text_lower.split()
        if words and words[0] in MODIFIER_KEYS:
            return CommandResult("keys", self._normalize_keys(text_lower))

        # 5. "press X" prefix for action keys (e.g., "press enter", "press tab")
        if text_lower.startswith("press "):
            key_name = text_lower[6:].strip()
            if key_name in ACTION_KEYS or key_name in KEY_CODES:
                return CommandResult("keys", self._normalize_keys(key_name))

        # 6. Action keys (enter, backspace, tab, etc.)
        if text_lower in ACTION_KEYS:
            return CommandResult("keys", self._normalize_keys(text_lower))

        # 7. Punctuation names (semicolon -> ;)
        if text_lower in PUNCTUATION_CHARS:
            return CommandResult("literal", PUNCTUATION_CHARS[text_lower])

        # 8. Spelled-out letters (A-B-C or A B C -> abc)
        # Returns "none" so it goes through pending text system, not immediate execution
        spelled = self._parse_spelled_letters(text_lower)
        if spelled:
            return CommandResult("none", spelled)

        # 9. Single letters (a, b, c, ...)
        if text_lower in SINGLE_LETTERS:
            return CommandResult("literal", SINGLE_LETTERS[text_lower])

        # 10. Digit words (one -> 1, two -> 2, ...)
        if text_lower in DIGIT_WORDS:
            return CommandResult("literal", DIGIT_WORDS[text_lower])

        # 11. Normal dictation
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

    def _parse_spelled_letters(self, text: str) -> Optional[str]:
        """
        Parse spelled-out letters/digits like 'A-B-C' or 'A, B, C' into 'abc'.

        Returns the parsed string if it's a spelling pattern, None otherwise.
        """
        # Split on hyphens, commas, or spaces
        parts = re.split(r'[-,\s]+', text.strip())

        # Filter out empty parts and strip punctuation
        parts = [p.strip('.,!?') for p in parts if p.strip('.,!?')]

        # Need at least 2 parts to be considered spelling
        if len(parts) < 2:
            return None

        result = []
        for part in parts:
            part = part.lower()
            if part in SINGLE_LETTERS:
                result.append(part)
            elif part in DIGIT_WORDS:
                result.append(DIGIT_WORDS[part])
            elif part.isdigit() and len(part) == 1:
                result.append(part)
            else:
                # Not a valid spelling pattern
                return None

        return ''.join(result)


NOTIFICATION_ID_FILE = Path.home() / ".cache/whisper-typer/notification_id"


def close_notification_by_id(notification_id: int):
    """Close a notification by its ID."""
    try:
        subprocess.run([
            'gdbus', 'call', '--session',
            '--dest', 'org.freedesktop.Notifications',
            '--object-path', '/org/freedesktop/Notifications',
            '--method', 'org.freedesktop.Notifications.CloseNotification',
            str(notification_id)
        ], capture_output=True, timeout=5)
    except Exception:
        pass


def cleanup_notification():
    """atexit handler to close notification from saved ID."""
    try:
        if NOTIFICATION_ID_FILE.exists():
            notification_id = int(NOTIFICATION_ID_FILE.read_text().strip())
            close_notification_by_id(notification_id)
            NOTIFICATION_ID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


atexit.register(cleanup_notification)


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
                # Save to file for cleanup if process is killed
                NOTIFICATION_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
                NOTIFICATION_ID_FILE.write_text(str(self.notification_id))
        except (FileNotFoundError, ValueError):
            pass

    def close(self):
        """Close the current notification."""
        if self.notification_id:
            close_notification_by_id(self.notification_id)
            self.notification_id = None
        # Clean up the saved ID file
        NOTIFICATION_ID_FILE.unlink(missing_ok=True)


@dataclass
class TyperState:
    """Tracks what has been typed to enable corrections.

    Simple protocol:
    - finalized_length: characters that are permanent (never delete before this)
    - pending_text: current pending text (can be updated via diff)
    """
    finalized_length: int = 0  # Characters typed that are permanent
    pending_text: str = ""     # Current pending text (can change)
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
        self._run_ydotool('type', '--key-delay=5', '--key-hold=2', '--clearmodifiers', '--', text)

    def backspace(self, count: int):
        """Send backspace key presses."""
        if count <= 0:
            return
        # Key code 14 is backspace - send all at once for speed
        # Format: keycode:1 (down) keycode:0 (up)
        keys = ' '.join(['14:1', '14:0'] * count)
        self._run_ydotool('key', '--key-delay=5', *keys.split())

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

        self._run_ydotool('key', '--key-delay=5', *key_args)

    def handle_finalize(self, text: str):
        """
        Handle finalized text from server - type it and mark as permanent.

        Args:
            text: Text that is now finalized (will never be deleted)
        """
        with self._lock:
            pending = self.state.pending_text

            if not text:
                # Nothing to finalize, just clear pending
                if pending:
                    self.backspace(grapheme_len(pending))
                    self.state.pending_text = ""
                return

            # Optimize: if finalized text starts with pending, just type the rest
            if pending and text.startswith(pending):
                # Pending is already on screen, just add the remainder
                remainder = text[len(pending):]  # Safe: pending is exact prefix
                if remainder:
                    logger.info(f"Finalizing (appending): {remainder!r}")
                    self.type_text(remainder)
                else:
                    logger.info(f"Finalizing (already typed): {text!r}")
                self.state.pending_text = ""
                self.state.finalized_length += grapheme_len(text)
                return

            # Optimize: if pending starts with finalized, backspace the extra
            if pending and pending.startswith(text):
                extra = grapheme_len(pending) - grapheme_len(text)
                if extra > 0:
                    logger.info(f"Finalizing (trimming {extra} graphemes): {text!r}")
                    self.backspace(extra)
                else:
                    logger.info(f"Finalizing (exact match): {text!r}")
                self.state.pending_text = ""
                self.state.finalized_length += grapheme_len(text)
                return

            # General case: clear pending and type finalized
            if pending:
                logger.info(f"Clearing pending before finalize: {pending!r}")
                self.backspace(grapheme_len(pending))
                self.state.pending_text = ""

            logger.info(f"Finalizing: {text!r}")
            self.type_text(text)
            self.state.finalized_length += grapheme_len(text)

    def update_pending(self, new_text: str):
        """
        Update pending text using diff/backspace.

        Args:
            new_text: The new pending text from server
        """
        with self._lock:
            old = self.state.pending_text

            if old == new_text:
                # No change
                return

            logger.info(f"update_pending: {old!r} -> {new_text!r}")

            if not old:
                # No previous pending, just type
                if new_text:
                    self.type_text(new_text)
                    self.state.pending_text = new_text
                return

            if not new_text:
                # Clear pending
                self.backspace(grapheme_len(old))
                self.state.pending_text = ""
                return

            # Find common prefix (grapheme-aware)
            old_graphemes = list(grapheme_iter(old))
            new_graphemes = list(grapheme_iter(new_text))
            common_len = 0
            min_len = min(len(old_graphemes), len(new_graphemes))
            for i in range(min_len):
                if old_graphemes[i] == new_graphemes[i]:
                    common_len = i + 1
                else:
                    break

            to_delete = len(old_graphemes) - common_len
            to_add = "".join(new_graphemes[common_len:])

            logger.info(f"Pending diff: common={common_len}, delete={to_delete}, add={grapheme_len(to_add)}")

            if to_delete > 0:
                self.backspace(to_delete)

            if to_add:
                self.type_text(to_add)

            self.state.pending_text = new_text

    def clear_pending(self):
        """Clear pending text without finalizing."""
        with self._lock:
            if self.state.pending_text:
                self.backspace(grapheme_len(self.state.pending_text))
                self.state.pending_text = ""


class MicrophoneStream:
    """Streams audio from the microphone using sounddevice."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.stream = None
        self._running = False
        self._queue: queue.Queue = queue.Queue()
        self._device_sample_rate: Optional[int] = None

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

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        # Calculate the number of output samples
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        # Use numpy interpolation for resampling
        orig_indices = np.arange(len(audio))
        target_indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(target_indices, orig_indices, audio.flatten()).astype(np.float32)

    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        # Resample if needed and send float32 audio
        audio = indata.flatten().astype(np.float32)
        if self._device_sample_rate and self._device_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, self._device_sample_rate, self.target_sample_rate)
        self._queue.put(audio.tobytes())

    def start(self):
        """Start the audio stream."""
        # Get device info
        dev_name = "default"
        if self.device_index is not None:
            dev_info = sd.query_devices(self.device_index)
            dev_name = dev_info['name']
            logger.info(f"Using device: {dev_name}")

        # For pipewire/pulseaudio/default, request target rate directly (they handle conversion)
        # For hardware devices (hw:), use native rate and resample ourselves
        use_native_rate = dev_name.startswith('hw:') or '(hw:' in dev_name

        if use_native_rate and self.device_index is not None:
            dev_info = sd.query_devices(self.device_index)
            self._device_sample_rate = int(dev_info['default_samplerate'])
            device_chunk_size = int(self.chunk_size * self._device_sample_rate / self.target_sample_rate)
            logger.info(f"Hardware device at {self._device_sample_rate} Hz, will resample to {self.target_sample_rate} Hz")
        else:
            self._device_sample_rate = self.target_sample_rate
            device_chunk_size = self.chunk_size
            logger.info(f"Using {self.target_sample_rate} Hz directly")

        self.stream = sd.InputStream(
            samplerate=self._device_sample_rate,
            channels=self.channels,
            device=self.device_index,
            blocksize=device_chunk_size,
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
        self.no_speech_thresh = config.no_speech_thresh
        self.min_avg_logprob = config.min_avg_logprob
        self.typer = typer or YdotoolTyper(dry_run=dry_run)
        self.mic = MicrophoneStream(device_index=device_index)
        self.notifier = Notifier()
        self.commands = CommandProcessor(config.commands)
        self._running = False
        self._websocket = None
        # Debounce for pending updates (avoid retyping every intermediate update)
        self._pending_debounce_ms = config.pending_debounce_ms
        self._debounce_timer: Optional[threading.Timer] = None
        self._debounced_pending: Optional[str] = None
        self._debounce_lock = threading.Lock()

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
            "no_speech_thresh": self.no_speech_thresh,
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

        # Set up signal handlers in the event loop
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def handle_signal():
            logger.info("Signal received, stopping...")
            self._running = False
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        send_task = None
        recv_task = None

        try:
            await self.connect()
            self.mic.start()

            # Start tasks for sending audio and receiving transcriptions
            send_task = asyncio.create_task(self._send_audio())
            recv_task = asyncio.create_task(self._receive_transcriptions())
            stop_task = asyncio.create_task(stop_event.wait())

            # Wait for either task to complete (usually due to disconnection or signal)
            done, pending = await asyncio.wait(
                [send_task, recv_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Close websocket first to unblock recv()
            if self._websocket:
                await self._websocket.close()

            # Cancel and await pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            # Stop mic (this is synchronous)
            self.mic.stop()
            self.notifier.close()
            logger.info("Client stopped")

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

    @staticmethod
    def _parse_segment_time(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_cmd_text(text: str) -> str:
        return text.lower().rstrip('.,!?')

    def _segment_passes_filters(self, segment: dict) -> bool:
        no_speech_prob = float(segment.get("no_speech_prob", 0))
        avg_logprob = float(segment.get("avg_logprob", 0))
        return no_speech_prob <= self.no_speech_thresh and avg_logprob >= self.min_avg_logprob

    def _handle_transcription(self, data: dict):
        """Process a transcription message from the server.

        Simple protocol:
        - finalize: text that is now final (type and never delete)
        - text: current pending text (update via diff)
        """
        logger.debug(f"Handling data: {data}")

        # Handle finalized text first (if any)
        if "finalize" in data:
            # Flush any debounced pending before finalizing
            self._flush_debounced_pending()

            finalized = data["finalize"]

            # Check for commands in finalized text
            finalized_stripped = finalized.strip()
            if finalized_stripped:
                cmd_result = self.commands.process(finalized_stripped)
                if cmd_result.action in ("keys", "literal"):
                    if (
                        not self.typer.state.last_command_text
                        or self._normalize_cmd_text(finalized_stripped)
                        != self._normalize_cmd_text(self.typer.state.last_command_text)
                    ):
                        self.typer.clear_pending()
                        self._execute_command(cmd_result)
                        self.typer.state.last_command_text = finalized_stripped
                    return

            # Normal finalization
            self.typer.handle_finalize(finalized)

        # Handle pending text update (if any)
        if "text" in data:
            pending = data["text"]

            # Check for commands in pending text
            pending_stripped = pending.strip()
            if pending_stripped:
                cmd_result = self.commands.process(pending_stripped)
                if cmd_result.action in ("keys", "literal"):
                    if (
                        not self.typer.state.last_command_text
                        or self._normalize_cmd_text(pending_stripped)
                        != self._normalize_cmd_text(self.typer.state.last_command_text)
                    ):
                        self.typer.clear_pending()
                        self._execute_command(cmd_result)
                        self.typer.state.last_command_text = pending_stripped
                    return

            # Normal pending update (debounced)
            self._schedule_pending_update(pending)

    def _flush_debounced_pending(self):
        """Immediately apply any debounced pending update."""
        with self._debounce_lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None
            if self._debounced_pending is not None:
                pending = self._debounced_pending
                self._debounced_pending = None
                self.typer.update_pending(pending)

    def _schedule_pending_update(self, pending: str):
        """Schedule a debounced pending update."""
        with self._debounce_lock:
            # Cancel existing timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Store the latest pending text
            self._debounced_pending = pending

            # If debounce is disabled (0), apply immediately
            if self._pending_debounce_ms <= 0:
                self._debounced_pending = None
                self.typer.update_pending(pending)
                return

            # Schedule delayed update
            def apply_pending():
                with self._debounce_lock:
                    if self._debounced_pending is not None:
                        p = self._debounced_pending
                        self._debounced_pending = None
                        self._debounce_timer = None
                        self.typer.update_pending(p)

            self._debounce_timer = threading.Timer(
                self._pending_debounce_ms / 1000.0,
                apply_pending
            )
            self._debounce_timer.start()

    def _execute_command(self, cmd_result: CommandResult):
        """Execute a voice command."""
        logger.info(f"Executing command: {cmd_result.action} -> {cmd_result.payload!r}")

        if cmd_result.action == "keys":
            self.typer.send_keys(cmd_result.payload)
        elif cmd_result.action == "literal":
            self.typer.type_text(cmd_result.payload)


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

    print("=" * 50)
    print("WhisperLive Typer")
    print("=" * 50)
    print(f"Server: {config.host}:{config.port}")
    print(f"Language: {config.language}")
    print(f"Model: {config.model}")
    print(f"VAD: onset={config.vad_onset}, offset={config.vad_offset}")
    print(f"Filters: no_speech_thresh={config.no_speech_thresh}, min_avg_logprob={config.min_avg_logprob}")
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
