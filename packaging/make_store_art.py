"""Writes the Store logos and poster to store/logos, from the icon the app draws."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter  # noqa: E402

from studio import theme  # noqa: E402
from studio.widgets import app_icon  # noqa: E402

app = QGuiApplication(sys.argv)
OUT = Path(__file__).parent.parent / "store" / "logos"
OUT.mkdir(exist_ok=True)


def art(w, h, icon, title=None):
    img = QImage(w, h, QImage.Format_RGB32)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    sky = QLinearGradient(0, 0, 0, h)
    sky.setColorAt(0, QColor("#23264F"))
    sky.setColorAt(0.65, QColor("#6B3A86"))
    sky.setColorAt(1, QColor("#E2694F"))
    p.fillRect(img.rect(), sky)
    top = (h - icon) // 2 - (h // 10 if title else 0)
    p.drawImage((w - icon) // 2, top, app_icon(icon))
    if title:
        f = QFont(theme.HEAD, 10)
        f.setPixelSize(w // 9)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(0, top + icon + h // 24, w, h // 7), Qt.AlignHCenter | Qt.AlignTop, title)
    p.end()
    return img


art(720, 1080, 430, "ToonDirector").save(str(OUT / "poster-720x1080.png"))
art(1080, 1080, 640).save(str(OUT / "box-art-1080x1080.png"))
for size in (300, 150, 71):
    app_icon(size).save(str(OUT / f"tile-{size}x{size}.png"))
print("ok")
