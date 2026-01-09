#!/usr/bin/env -S uv run
"""
Test ydotool key sending to verify it works correctly.
"""

import subprocess
import os
import time
import sys

# Key codes for testing
KEY_CODES = {
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
    "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
    "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
    "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "enter": 28, "backspace": 14, "space": 57,
}


def get_socket():
    """Find ydotool socket."""
    for path in [
        os.environ.get('YDOTOOL_SOCKET'),
        '/run/ydotoold/socket',
        f'/run/user/{os.getuid()}/.ydotool_socket',
    ]:
        if path and os.path.exists(path):
            return path
    return None


def send_key(key: str, socket: str, dry_run: bool = False):
    """Send a single key press."""
    code = KEY_CODES.get(key.lower())
    if not code:
        print(f"Unknown key: {key}")
        return False

    # Key down, key up
    args = ['ydotool', 'key', '--key-delay=10', f'{code}:1', f'{code}:0']
    env = os.environ.copy()
    env['YDOTOOL_SOCKET'] = socket

    print(f"Sending: {key} (code {code})")
    print(f"Command: {' '.join(args)}")

    if dry_run:
        print("[DRY RUN]")
        return True

    result = subprocess.run(args, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def main():
    dry_run = '--dry-run' in sys.argv

    socket = get_socket()
    if not socket:
        print("ERROR: ydotool socket not found!")
        print("Make sure ydotoold is running.")
        sys.exit(1)

    print(f"Using socket: {socket}")
    print()

    if dry_run:
        print("=== DRY RUN MODE ===")
        print()

    print("Testing individual keys...")
    print("Focus a text editor window!")
    print()

    time.sleep(2)

    # Test each letter
    for key in ['h', 'e', 'l', 'l', 'o']:
        send_key(key, socket, dry_run)
        time.sleep(0.1)

    print()
    print("Should have typed: hello")
    print()

    # Test x specifically
    print("Now testing 'x' specifically...")
    time.sleep(1)
    send_key('x', socket, dry_run)

    print()
    print("Done. Check if the output is correct.")


if __name__ == '__main__':
    main()
