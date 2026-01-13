"""
macOS platform implementation using pynput and osascript.
"""

import logging
import subprocess
import sys

from .base import Notifier, Typer

logger = logging.getLogger(__name__)

# macOS key mappings for pynput
# pynput uses Key enum for special keys
KEY_MAP = {
    # Special keys
    "escape": "esc", "esc": "esc",
    "backspace": "backspace", "back": "backspace",
    "tab": "tab",
    "enter": "enter", "return": "enter",
    "space": "space",
    "delete": "delete", "del": "delete",
    # Modifiers
    "ctrl": "ctrl", "control": "ctrl",
    "shift": "shift",
    "alt": "alt", "option": "alt",
    "super": "cmd", "meta": "cmd", "command": "cmd", "cmd": "cmd",
    # Navigation
    "home": "home", "end": "end",
    "pageup": "page_up", "pgup": "page_up",
    "pagedown": "page_down", "pgdn": "page_down",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "insert": "insert", "ins": "insert",
    # Function keys
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4", "f5": "f5", "f6": "f6",
    "f7": "f7", "f8": "f8", "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
}


def check_accessibility_permissions() -> bool:
    """
    Check if the app has accessibility permissions on macOS.

    Returns:
        True if permissions are granted, False otherwise.
    """
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import AXIsProcessTrusted
        return AXIsProcessTrusted()
    except ImportError:
        # PyObjC not available, try alternative check
        pass

    # Alternative: try to use pynput and see if it works
    try:
        from pynput.keyboard import Controller
        kb = Controller()
        # Try a no-op to see if we have permissions
        return True
    except Exception:
        return False


def request_accessibility_permissions():
    """
    Prompt user to grant accessibility permissions.

    Opens System Preferences to the correct pane.
    """
    print("\n" + "=" * 60)
    print("ACCESSIBILITY PERMISSIONS REQUIRED")
    print("=" * 60)
    print()
    print("Whisper Typer needs accessibility permissions to type text.")
    print()
    print("Please follow these steps:")
    print("1. System Preferences will open to Privacy & Security")
    print("2. Click on 'Accessibility' in the left sidebar")
    print("3. Click the '+' button and add Terminal (or your IDE)")
    print("4. Restart Whisper Typer after granting permissions")
    print()
    print("=" * 60 + "\n")

    # Open System Preferences to Accessibility pane
    subprocess.run([
        'open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'
    ])


def ensure_accessibility_permissions():
    """
    Check for accessibility permissions and prompt if missing.

    Exits the program if permissions are not granted (only when running interactively).
    """
    if not check_accessibility_permissions():
        # Check if running interactively (has a tty)
        if sys.stdin.isatty():
            request_accessibility_permissions()
            print("Please grant accessibility permissions and restart the application.")
            sys.exit(1)
        else:
            # Running under launchctl or similar - just log a warning
            logger.warning("Accessibility permissions may not be granted. Typing may not work.")
            logger.warning("Add the Python interpreter to System Settings > Privacy & Security > Accessibility")


class MacOSNotifier(Notifier):
    """
    Desktop notifications for macOS.

    Uses terminal-notifier if available (brew install terminal-notifier),
    otherwise falls back to osascript (appears under "Script Editor" in
    System Settings > Notifications).
    """

    # Common locations for terminal-notifier
    TERMINAL_NOTIFIER_PATHS = [
        '/opt/homebrew/bin/terminal-notifier',  # Apple Silicon Homebrew
        '/usr/local/bin/terminal-notifier',      # Intel Homebrew
    ]

    def __init__(self):
        self._active = False
        # Find terminal-notifier path
        self._terminal_notifier_path = self._find_terminal_notifier()

    def _find_terminal_notifier(self) -> str | None:
        """Find terminal-notifier executable."""
        import os
        for path in self.TERMINAL_NOTIFIER_PATHS:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Found terminal-notifier at: {path}")
                return path
        logger.info("terminal-notifier not found")
        return None

    def show(self, message: str, persistent: bool = False):
        """Show notification."""
        if self._terminal_notifier_path:
            try:
                logger.info(f"Sending notification: {message}")
                result = subprocess.run(
                    [
                        self._terminal_notifier_path,
                        '-title', 'Whisper Typer',
                        '-message', message,
                        '-group', 'whisper-typer',
                    ],
                    capture_output=True,
                    timeout=5
                )
                logger.info(f"terminal-notifier returned: {result.returncode}, stderr: {result.stderr.decode() if result.stderr else ''}")
                self._active = True
                return
            except Exception as e:
                logger.warning(f"terminal-notifier failed: {e}")

        # Fallback to osascript
        try:
            escaped_message = message.replace('"', '\\"').replace("'", "\\'")
            script = f'display notification "{escaped_message}" with title "Whisper Typer"'
            subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                timeout=5
            )
            self._active = True
        except Exception as e:
            logger.warning(f"Failed to show notification: {e}")

    def close(self):
        """Remove notifications (only works with terminal-notifier)."""
        self._active = False
        if self._terminal_notifier_path:
            try:
                subprocess.run(
                    [self._terminal_notifier_path, '-remove', 'whisper-typer'],
                    capture_output=True,
                    timeout=5
                )
            except Exception:
                pass


class PynputTyper(Typer):
    """Keyboard input using pynput (cross-platform)."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._controller = None
        self._Key = None

        if not dry_run:
            try:
                from pynput.keyboard import Controller, Key
                self._controller = Controller()
                self._Key = Key
            except ImportError:
                logger.error("pynput not installed. Run: pip install pynput")
                raise

    def _get_key(self, key_name: str):
        """Convert key name to pynput Key object."""
        key_name = key_name.lower().rstrip('.,!?')

        # Map to pynput key name
        mapped = KEY_MAP.get(key_name, key_name)

        # Try to get from Key enum
        if hasattr(self._Key, mapped):
            return getattr(self._Key, mapped)

        # Single character
        if len(key_name) == 1:
            return key_name

        return None

    def type_text(self, text: str):
        """Type text using pynput."""
        if not text:
            return

        logger.info(f"Typing: {text!r}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would type: {text}")
            return

        self._controller.type(text)

    def backspace(self, count: int):
        """Send backspace key presses."""
        if count <= 0:
            return

        if self.dry_run:
            logger.info(f"[DRY RUN] Would backspace {count} times")
            return

        for _ in range(count):
            self._controller.press(self._Key.backspace)
            self._controller.release(self._Key.backspace)

    def send_keys(self, keys_str: str):
        """Send key combination(s)."""
        sequences = keys_str.strip().split()
        if not sequences:
            return

        for seq in sequences:
            self._send_key_combo(seq)

    def _send_key_combo(self, combo: str):
        """Send a single key combination like 'ctrl+c' or 'enter'."""
        combo = combo.lower().rstrip('.,!?')
        parts = combo.split('+')

        keys = []
        for part in parts:
            part = part.strip().rstrip('.,!?')
            key = self._get_key(part)
            if key is None:
                logger.warning(f"Unknown key: {part}")
                return
            keys.append(key)

        if not keys:
            return

        logger.info(f"Sending keys: {combo}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would send: {combo}")
            return

        # Press all keys (modifiers first)
        for key in keys:
            self._controller.press(key)

        # Release all keys in reverse order
        for key in reversed(keys):
            self._controller.release(key)
