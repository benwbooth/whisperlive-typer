"""
Linux platform implementation using ydotool and notify-send.
"""

import atexit
import logging
import os
import subprocess
from pathlib import Path

from .base import Notifier, Typer

logger = logging.getLogger(__name__)

# Linux keyboard scan codes for ydotool
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

NOTIFICATION_ID_FILE = Path.home() / ".cache/whisper-typer/notification_id"


def _close_notification_by_id(notification_id: int):
    """Close a notification by its ID using D-Bus."""
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


def _cleanup_notification():
    """atexit handler to close notification from saved ID."""
    try:
        if NOTIFICATION_ID_FILE.exists():
            notification_id = int(NOTIFICATION_ID_FILE.read_text().strip())
            _close_notification_by_id(notification_id)
            NOTIFICATION_ID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# Register cleanup on exit
atexit.register(_cleanup_notification)


class LinuxNotifier(Notifier):
    """Desktop notifications using notify-send (freedesktop)."""

    def __init__(self):
        self.notification_id = None

    def show(self, message: str, persistent: bool = False):
        """Show notification using notify-send."""
        try:
            cmd = ['notify-send', '--app-name=Whisper Typer', '--print-id']
            if persistent:
                cmd.extend(['--expire-time=0'])
            if self.notification_id:
                cmd.extend(['--replace-id', str(self.notification_id)])
            cmd.append(message)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout.strip():
                self.notification_id = int(result.stdout.strip())
                NOTIFICATION_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
                NOTIFICATION_ID_FILE.write_text(str(self.notification_id))
        except (FileNotFoundError, ValueError):
            pass

    def close(self):
        """Close the current notification."""
        if self.notification_id:
            _close_notification_by_id(self.notification_id)
            self.notification_id = None
        NOTIFICATION_ID_FILE.unlink(missing_ok=True)


class YdotoolTyper(Typer):
    """Keyboard input using ydotool (Linux/Wayland)."""

    def __init__(
        self,
        socket_path: str | None = None,
        dry_run: bool = False,
        key_delay: int = 2,
        key_hold: int = 1,
    ):
        self.socket_path = socket_path or os.environ.get(
            'YDOTOOL_SOCKET',
            '/run/ydotoold/socket'
        )
        self.dry_run = dry_run
        self.key_delay = key_delay
        self.key_hold = key_hold

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
        self._run_ydotool('type', f'--key-delay={self.key_delay}', f'--key-hold={self.key_hold}', '--clearmodifiers', '--', text)

    def backspace(self, count: int):
        """Send backspace key presses."""
        if count <= 0:
            return
        remaining = count
        batch_size = 32
        while remaining > 0:
            batch = min(batch_size, remaining)
            key_events = []
            for _ in range(batch):
                key_events.extend(['14:1', '14:0'])
            self._run_ydotool(
                'key',
                f'--key-delay={self.key_delay}',
                f'--key-hold={self.key_hold}',
                '--clearmodifiers',
                *key_events,
            )
            remaining -= batch

    def send_keys(self, keys_str: str):
        """Send key combination(s)."""
        sequences = keys_str.strip().split()
        if not sequences:
            return

        combos = []
        for seq in sequences:
            if '+' in seq:
                combos.append(seq)
            else:
                combos.append(seq)

        for combo in combos:
            self._send_key_combo(combo)

    def _send_key_combo(self, combo: str):
        """Send a single key combination like 'ctrl+c' or 'enter'."""
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

        key_args = []
        for code in codes:
            key_args.append(f"{code}:1")
        for code in reversed(codes):
            key_args.append(f"{code}:0")

        self._run_ydotool(
            'key',
            f'--key-delay={self.key_delay}',
            f'--key-hold={self.key_hold}',
            '--clearmodifiers',
            *key_args,
        )
