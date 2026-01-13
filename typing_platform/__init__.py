"""
Platform-specific implementations for notifications and keyboard input.

Auto-selects the appropriate implementation based on the current OS.
"""

import sys

from .base import Notifier, Typer

__all__ = ['Notifier', 'Typer', 'get_notifier', 'get_typer', 'ensure_permissions']


def get_notifier() -> Notifier:
    """Get the platform-appropriate Notifier implementation."""
    if sys.platform == 'darwin':
        from .macos import MacOSNotifier
        return MacOSNotifier()
    elif sys.platform == 'linux':
        from .linux import LinuxNotifier
        return LinuxNotifier()
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")


def get_typer(dry_run: bool = False, **kwargs) -> Typer:
    """
    Get the platform-appropriate Typer implementation.

    Args:
        dry_run: If True, don't actually type (just log)
        **kwargs: Platform-specific arguments (e.g., socket_path for Linux)
    """
    if sys.platform == 'darwin':
        from .macos import PynputTyper
        return PynputTyper(dry_run=dry_run)
    elif sys.platform == 'linux':
        from .linux import YdotoolTyper
        return YdotoolTyper(dry_run=dry_run, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")


def ensure_permissions():
    """
    Ensure platform-specific permissions are granted.

    On macOS, checks for accessibility permissions and prompts if missing.
    On Linux, this is a no-op (ydotool doesn't need special permissions if
    the daemon is running).
    """
    if sys.platform == 'darwin':
        from .macos import ensure_accessibility_permissions
        ensure_accessibility_permissions()
    # Linux: no special permissions needed (ydotool daemon handles it)
