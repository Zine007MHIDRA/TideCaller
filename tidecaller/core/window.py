"""Locating the Roblox window. Read-only: we never touch the process itself."""
from __future__ import annotations

import sys

from tidecaller.core.regions import Rect

ROBLOX_TITLE = "Roblox"


def find_roblox_client_rect() -> Rect | None:
    """Client-area rect of the Roblox window in screen coordinates, or None."""
    if sys.platform != "win32":
        return None
    import win32gui

    hwnd = win32gui.FindWindow(None, ROBLOX_TITLE)
    if not hwnd or win32gui.IsIconic(hwnd):
        return None
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
    return Rect(sx, sy, right - left, bottom - top)


def roblox_is_focused() -> bool:
    if sys.platform != "win32":
        return True
    import win32gui

    return win32gui.GetWindowText(win32gui.GetForegroundWindow()) == ROBLOX_TITLE


def primary_screen_rect() -> Rect:
    if sys.platform == "win32":
        import ctypes

        u = ctypes.windll.user32
        return Rect(0, 0, u.GetSystemMetrics(0), u.GetSystemMetrics(1))
    return Rect(0, 0, 1920, 1080)
