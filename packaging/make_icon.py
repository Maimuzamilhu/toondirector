"""Writes packaging/icon.ico from the icon the app draws. Run: python packaging/make_icon.py"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from studio.widgets import app_icon  # noqa: E402

app = QGuiApplication(sys.argv)
parts = []
for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    app_icon(size).save(buf, "PNG")
    parts.append((size, bytes(buf.data())))
head = struct.pack("<HHH", 0, 1, len(parts))
offset = 6 + 16 * len(parts)
table = b""
for size, data in parts:
    table += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
(Path(__file__).parent / "icon.ico").write_bytes(head + table + b"".join(d for _, d in parts))
