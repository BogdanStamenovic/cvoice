# -*- coding: utf-8 -*-
"""A small multi-select list for the terminal.

Deliberately dependency-free: cvoice's client half is meant to install on any
laptop without pulling a TUI toolkit in behind it. Raw-mode keys on POSIX,
msvcrt on Windows, and a numbered prompt when stdin is not a terminal.
"""
from __future__ import annotations

import sys

B, D, G, Y, R, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
CLR = "\033[2K"
UP = "\033[A"


def _read_key_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":                       # escape sequence: arrows etc.
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        return {b"H": "up", b"P": "down", b"M": "right", b"K": "left"}.get(
            msvcrt.getch(), "esc")
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch == b" ":
        return "space"
    if ch == b"\x03":
        raise KeyboardInterrupt
    return ch.decode("utf-8", "ignore").lower()


def read_key() -> str:
    return _read_key_windows() if sys.platform == "win32" else _read_key_posix()


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def multiselect(items, title="", labeller=str, preselected=None):
    """Return the chosen items, or None if cancelled.

    items      : the objects to choose from
    labeller   : object -> one display line
    """
    if not items:
        return []
    if not interactive():
        # Non-interactive: make the caller pass names explicitly rather than
        # guessing, so a script never silently shares the wrong thing.
        return None

    chosen = set(preselected or ())
    cursor = 0
    n = len(items)
    drawn = 0

    def draw(first):
        nonlocal drawn
        if not first:
            sys.stdout.write(UP * drawn)
        lines = []
        if title:
            lines.append(f"{B}{title}{X}")
        for i, it in enumerate(items):
            mark = f"{G}●{X}" if i in chosen else f"{D}○{X}"
            pointer = f"{B}❯{X}" if i == cursor else " "
            label = labeller(it)
            if i == cursor:
                label = f"{B}{label}{X}"
            lines.append(f" {pointer} {mark} {label}")
        lines.append(f"{D}   ↑↓ kreći se · razmak označi · a sve · enter potvrdi · q otkaži{X}")
        sys.stdout.write("".join(CLR + line + "\n" for line in lines))
        sys.stdout.flush()
        drawn = len(lines)

    draw(True)
    try:
        while True:
            key = read_key()
            if key in ("up", "k"):
                cursor = (cursor - 1) % n
            elif key in ("down", "j"):
                cursor = (cursor + 1) % n
            elif key == "space":
                chosen.symmetric_difference_update({cursor})
            elif key == "a":
                chosen = set() if len(chosen) == n else set(range(n))
            elif key == "enter":
                break
            elif key in ("q", "esc"):
                draw(False)
                return None
            draw(False)
    except KeyboardInterrupt:
        print()
        return None
    return [items[i] for i in sorted(chosen)]
