"""The app's notifications: kept in a list you can look back at, and pushed to the window."""
import json
import threading
import time
import uuid

from . import config

_PATH = config.DATA_DIR / "notifications.json"
_lock = threading.Lock()
listeners = []  # callbacks(notification dict); the window shows it, Windows pops it up


def _read():
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _write(items):
    _PATH.write_text(json.dumps(items[-100:], indent=1), encoding="utf-8")


def add(kind, title, body="", job=None):
    """kind: done | error | info"""
    n = {"id": uuid.uuid4().hex[:8], "kind": kind, "title": title, "body": body, "job": job,
         "time": time.time(), "read": False}
    with _lock:
        items = _read()
        items.append(n)
        _write(items)
    for cb in listeners:
        cb(n)
    return n


def items():
    with _lock:
        return list(reversed(_read()))


def unread():
    return sum(1 for n in items() if not n["read"])


def mark_all_read():
    with _lock:
        all_ = _read()
        for n in all_:
            n["read"] = True
        _write(all_)


def clear():
    with _lock:
        _write([])
