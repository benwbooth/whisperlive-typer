"""
Abstract base classes for platform-specific functionality.
"""

from abc import ABC, abstractmethod


class Notifier(ABC):
    """Abstract base class for desktop notifications."""

    @abstractmethod
    def show(self, message: str, persistent: bool = False):
        """
        Show a notification.

        Args:
            message: The notification message
            persistent: If True, notification stays until manually closed
        """
        pass

    @abstractmethod
    def close(self):
        """Close the current notification."""
        pass


class Typer(ABC):
    """Abstract base class for keyboard input simulation."""

    @abstractmethod
    def type_text(self, text: str):
        """
        Type text as if from keyboard.

        Args:
            text: The text to type
        """
        pass

    @abstractmethod
    def backspace(self, count: int):
        """
        Send backspace key presses.

        Args:
            count: Number of backspaces to send
        """
        pass

    @abstractmethod
    def send_keys(self, keys_str: str):
        """
        Send a key combination.

        Args:
            keys_str: Key combo like "ctrl+c", "ctrl+shift+t", "enter", "enter enter"
        """
        pass
