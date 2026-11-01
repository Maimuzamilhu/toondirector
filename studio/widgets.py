"""Small building blocks shared by every page."""
import ctypes
import time
import urllib.parse
from pathlib import Path

from PySide6.QtCore import QByteArray, QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QIcon, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPixmap, QRadialGradient)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QComboBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from app import config
from app.jobs import FRAME

from . import theme


def label(text="", role=None, wrap=False):
    w = QLabel(text)
    if role:
        w.setProperty("role", role)
    w.setWordWrap(wrap)
    return w


def button(text, kind=None, on_click=None, tip=None):
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if on_click:
        b.clicked.connect(lambda *_: on_click())
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


def combo(items, value=None):
    """items: list of (value, text)."""
    c = QComboBox()
    for v, t in items:
        c.addItem(t, v)
    if value is not None:
        c.setCurrentIndex(max(c.findData(value), 0))
    return c


def field(text, widget, hint=None):
    box = QVBoxLayout()
    box.setSpacing(5)
    box.addWidget(label(text, "field"))
    box.addWidget(widget)
    if hint:
        box.addWidget(label(hint, "small", True))
    return box


def hbox(*items, spacing=10, stretch_end=False):
    h = QHBoxLayout()
    h.setSpacing(spacing)
    h.setContentsMargins(0, 0, 0, 0)
    for i in items:
        if i is None:
            h.addStretch(1)
        elif isinstance(i, QWidget):
            h.addWidget(i)
        else:
            h.addLayout(i)
    if stretch_end:
        h.addStretch(1)
    return h


def clear_layout(lay):
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w:
            w.setParent(None)
            w.deleteLater()
        elif it.layout():
            clear_layout(it.layout())


class Page(QScrollArea):
    """A scrolling page with even margins. Add content to self.body."""

    def __init__(self, max_width=1180):
        super().__init__()
        self.setObjectName("page")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget()
        self.body = QVBoxLayout(holder)
        self.body.setContentsMargins(40, 32, 40, 32)
        self.body.setSpacing(18)
        self.setWidget(holder)


class StageStrip(QWidget):
    """The job as a film strip: one frame per step, filling up as the work moves along."""

    def __init__(self):
        super().__init__()
        self.stages, self.index, self.fraction, self.state = [], 0, 0.0, "running"
        self.shown = 0.0   # eases towards the real position, so progress glides instead of jumping
        self.setFixedHeight(66)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.glide)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAccessibleName("Progress")

    def set_job(self, job):
        self.stages = [x[1] for x in job.stages]
        self.index, self.fraction = job.stage_position()
        self.state = job.status
        self.setAccessibleDescription(f"{job.progress:.0f} percent, {job.message}")
        if abs(self.index + self.fraction - self.shown) > 1 or reduced_motion():
            self.shown = self.index + self.fraction
        if not self.timer.isActive():
            self.timer.start()
        self.update()

    def glide(self):
        goal = self.index + self.fraction
        self.shown += (goal - self.shown) * 0.12
        if abs(goal - self.shown) < 0.001:
            self.shown = goal
            self.timer.stop()
        self.update()

    def paintEvent(self, _):
        if not self.stages:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, n = self.width(), self.height(), len(self.stages)
        strip_h = h - 22
        gap = 6
        fw = (w - gap * (n - 1)) / n
        accent = QColor(theme.CORAL if self.state == "error" else theme.MARIGOLD)
        p.setBrush(QColor(theme.SLATE))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, w, strip_h), 7, 7)
        for i, name in enumerate(self.stages):
            x = i * (fw + gap)
            r = QRectF(x + 3, 9, fw - 6, strip_h - 18)
            p.setBrush(QColor(theme.NIGHT))
            p.drawRoundedRect(r, 3, 3)
            fill = min(max(self.shown - i, 0.0), 1.0)
            if self.state == "done":
                fill = 1.0
            if fill > 0:
                p.setBrush(QColor(theme.TEAL) if self.state == "done" else accent)
                p.drawRoundedRect(QRectF(r.x(), r.y(), r.width() * fill, r.height()), 3, 3)
            current = i == self.index and self.state != "done"
            p.setPen(QColor(theme.TEXT if current or fill else theme.MUTED))
            f = QFont(theme.BODY, 9)
            f.setBold(current)
            p.setFont(f)
            p.drawText(QRectF(x, strip_h + 4, fw, 18), Qt.AlignCenter, name)
            p.setPen(Qt.NoPen)
        # sprocket holes along both edges
        p.setBrush(QColor(theme.NIGHT))
        step = 16
        for k in range(int(w / step)):
            for y in (2.5, strip_h - 6.5):
                p.drawRoundedRect(QRectF(5 + k * step, y, 7, 4), 1, 1)


class Thumb(QWidget):
    """A 16:9 picture, or a calm placeholder while a video is not finished."""

    def __init__(self):
        super().__init__()
        self.pix = None
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_image(self, path):
        self.pix = QPixmap(str(path)) if path else None
        self.update()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return int(w * 9 / 16)

    def resizeEvent(self, e):
        self.setFixedHeight(self.heightForWidth(self.width()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        if r.isEmpty():
            return
        clip = QPainterPath()
        clip.addRoundedRect(r, 10, 10)
        p.setClipPath(clip)
        if self.pix and not self.pix.isNull():
            s = self.pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            p.drawPixmap((self.width() - s.width()) // 2, (self.height() - s.height()) // 2, s)
        else:
            g = QLinearGradient(0, 0, 0, r.height())
            g.setColorAt(0, QColor("#2B2E52"))
            g.setColorAt(1, QColor("#4A3A66"))
            p.fillRect(r, g)


class Stat(QWidget):
    """One live number with its name underneath."""

    def __init__(self, name):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.value = label("", "stat")
        v.addWidget(self.value)
        v.addWidget(label(name, "small"))

    def set(self, text):
        self.value.setText(text)
        self.setAccessibleDescription(text)


class LivePanel(QFrame):
    """A video while it is being made: the newest drawn frame, the percent, what is happening
    this second, and the measured speed and time left."""
    opened = Signal(str)

    def __init__(self, compact=False):
        super().__init__()
        self.compact = compact
        self.job_id = None
        self.pic_key = None
        self.setProperty("panel", True)
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 16)
        v.setSpacing(14)
        top = QHBoxLayout()
        top.setSpacing(22)
        self.picture = Thumb()
        width = 300 if compact else 400
        self.picture.setFixedSize(width, width * 9 // 16)
        self.picture.setMinimumHeight(0)
        top.addWidget(self.picture)
        side = QVBoxLayout()
        side.setSpacing(4)
        self.title = label("", "h2", True)
        self.percent = label("", "display")
        self.step = label("", "muted", True)
        side.addWidget(self.title)
        side.addWidget(self.percent)
        side.addWidget(self.step)
        side.addStretch(1)
        self.frames, self.speed, self.left, self.elapsed = Stat("Frames"), Stat("Speed"), Stat("Time left"), Stat("Running for")
        row = QHBoxLayout()
        row.setSpacing(24)
        for s in (self.frames, self.speed, self.left, self.elapsed):
            row.addWidget(s)
        row.addStretch(1)
        side.addLayout(row)
        top.addLayout(side, 1)
        v.addLayout(top)
        self.strip = StageStrip()
        v.addWidget(self.strip)
        self.title.setVisible(compact)
        if compact:
            self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self.compact and e.button() == Qt.LeftButton and self.job_id:
            self.opened.emit(self.job_id)

    def show_job(self, job):
        self.job_id = job.id
        self.title.setText(job.title)
        self.percent.setText(f"{job.progress:.0f}%")
        self.strip.set_job(job)
        stopped = job.status == "error"
        drawing = FRAME.search(job.message or "")
        self.step.setText(job.error if stopped else "Waiting for the video before it to finish"
                          if job.status == "queued" else "Drawing the video" if drawing else job.message)
        live = job.live()
        self.frames.set(f"{live['frame']:,} of {live['total']:,}" if live["frame"] is not None else "Not started")
        self.speed.set(f"{live['fps']:.0f} frames a second" if live["fps"] else "Measuring" if live["frame"] is not None else "Not started")
        self.left.set(clock(live["left"]) if live["left"] is not None else "Stopped" if stopped else "Working it out")
        self.elapsed.set(clock(live["elapsed"]) if live["elapsed"] is not None else "Stopped" if stopped else "Not started")
        path = job.preview
        key = (str(path), path.stat().st_mtime_ns) if path and path.exists() else None
        if key != self.pic_key:
            self.pic_key = key
            self.picture.set_image(path)


class RailJob(QFrame):
    """A small live progress card in the side rail, visible from every page."""
    opened = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("railJob")
        self.setCursor(Qt.PointingHandCursor)
        self.job_id = None
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        self.text = label("", "small", True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        v.addWidget(self.text)
        v.addWidget(self.bar)

    def show_job(self, job):
        self.job_id = job.id
        left = job.live()["left"]
        self.text.setText(f"Making a video, {job.progress:.0f}%" + (f"\n{clock(left)} left" if left is not None else ""))
        self.bar.setValue(int(job.progress * 10))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.job_id:
            self.opened.emit(self.job_id)


class VideoCard(QFrame):
    """One video in the library."""
    opened = Signal(str)

    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
        self.setObjectName("videoCard")
        self.setProperty("panel", True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 12)
        v.setSpacing(6)
        self.thumb = Thumb()
        self.title = label("", "h3", True)
        self.title.setMaximumHeight(44)
        self.status = label("", "small")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        v.addWidget(self.thumb)
        v.addWidget(self.title)
        v.addWidget(self.bar)
        v.addWidget(self.status)

    def show_job(self, job, thumb):
        self.title.setText(job.title)
        self.setAccessibleName(job.title)
        self.bar.setVisible(job.active)
        self.bar.setValue(int(job.progress))
        self.thumb.set_image(thumb)
        when = time.strftime("%d %b %Y", time.localtime(job.created or 0))
        text = {"queued": "Waiting to start", "running": f"Making it now, {job.progress:.0f}%", "done": f"Finished {when}"}
        self.status.setProperty("role", "error" if job.status == "error" else "small")
        self.status.setText(text.get(job.status, "Stopped, can be resumed"))
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.opened.emit(self.job_id)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.opened.emit(self.job_id)
        else:
            super().keyPressEvent(e)


class Toast(QFrame):
    """A short message in the corner that fades away. Click it to open what it talks about."""
    clicked = Signal(object)

    def __init__(self, parent, note):
        super().__init__(parent)
        self.note = note
        self.setObjectName("toast")
        self.setProperty("kind", note["kind"])
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(340)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 11, 14, 11)
        v.setSpacing(2)
        role = {"done": "good", "error": "error"}.get(note["kind"], "h3")
        title = label(note["title"], role, True)
        v.addWidget(title)
        if note.get("body"):
            v.addWidget(label(note["body"][:120], "small", True))
        self.fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.fx)
        self.anim = QPropertyAnimation(self.fx, b"opacity", self)
        self.anim.setDuration(220)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()
        QTimer.singleShot(6000, self.fade)

    def fade(self):
        self.anim.setDirection(QPropertyAnimation.Backward)
        self.anim.finished.connect(self.deleteLater)
        self.anim.start()

    def mouseReleaseEvent(self, _):
        self.clicked.emit(self.note)
        self.deleteLater()


def report_problem(job=None):
    """Opens an email to the developer. Microsoft requires a way to report bad AI output."""
    subject = "Report: inappropriate AI content or a problem"
    body = f"What went wrong?\n\n\nApp version: {config.VERSION}\n" + (f"Video: {job.title}\n" if job else "")
    query = urllib.parse.urlencode({"subject": subject, "body": body}, quote_via=urllib.parse.quote)
    QDesktopServices.openUrl(QUrl(f"mailto:{config.SUPPORT_EMAIL}?{query}"))


def reduced_motion():
    """True when Windows is set to show fewer animations."""
    on = ctypes.c_int(1)
    ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(on), 0)
    return not on.value


def _svg(name, color, size):
    data = (Path(__file__).parent / "assets" / f"{name}.svg").read_text(encoding="utf-8").replace("COLOR", color)
    pix = QPixmap(size * 3, size * 3)
    pix.setDevicePixelRatio(3)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    QSvgRenderer(QByteArray(data.encode())).render(p, QRectF(0, 0, size, size))
    return pix, p


def line_icon(name, size=20):
    """A rail icon: quiet normally, marigold on the page you are on."""
    icon = QIcon()
    for color, state in ((theme.MUTED, QIcon.Off), (theme.MARIGOLD, QIcon.On)):
        pix, p = _svg(name, color, size)
        p.end()
        icon.addPixmap(pix, QIcon.Normal, state)
    return icon


def bell_icon(unread, size=20):
    pix, p = _svg("bell", theme.MUTED, size)
    if unread:
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.MARIGOLD))
        p.drawEllipse(QPointF(size * .76, size * .24), size * .2, size * .2)
    p.end()
    return QIcon(pix)


def app_icon(size=256):
    """A character standing in front of the sun at dusk, the look of the videos themselves.
    Drawn in code, so it stays sharp at every size."""
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 100
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(3 * s, 3 * s, 94 * s, 94 * s), 22 * s, 22 * s)
    p.setClipPath(tile)
    sky = QLinearGradient(0, 3 * s, 0, 97 * s)
    sky.setColorAt(0, QColor("#23264F"))
    sky.setColorAt(0.6, QColor("#6B3A86"))
    sky.setColorAt(1, QColor("#E2694F"))
    p.fillRect(img.rect(), sky)
    glow = QRadialGradient(QPointF(50 * s, 52 * s), 46 * s)
    glow.setColorAt(0, QColor(244, 185, 46, 150))
    glow.setColorAt(1, QColor(244, 185, 46, 0))
    p.fillRect(img.rect(), glow)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.MARIGOLD))
    p.drawEllipse(QPointF(50 * s, 52 * s), 31 * s, 31 * s)
    ink = QColor("#12121A")
    p.setBrush(ink)
    p.drawRoundedRect(QRectF(34 * s, 69 * s, 32 * s, 50 * s), 16 * s, 16 * s)
    p.drawEllipse(QPointF(50 * s, 47 * s), 20 * s, 20 * s)
    p.setBrush(QColor("white"))
    p.drawEllipse(QPointF(42.5 * s, 46 * s), 4.6 * s, 4.6 * s)
    p.drawEllipse(QPointF(57.5 * s, 46 * s), 4.6 * s, 4.6 * s)
    p.end()
    return img


def dot_icon(on):
    pix = QPixmap(30, 30)
    pix.setDevicePixelRatio(3)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.TEAL if on else theme.EDGE))
    p.drawEllipse(QPointF(5, 5), 4, 4)
    p.end()
    return QIcon(pix)


def app_qicon():
    icon = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(app_icon(size)))
    return icon


def clock(seconds):
    """A running time, down to the second."""
    seconds = int(max(seconds, 0))
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    if h:
        return f"{h} h {m} min"
    return f"{m} min {s} sec" if m else f"{s} sec"


def hms(seconds):
    seconds = int(max(seconds, 0))
    h, m = seconds // 3600, seconds % 3600 // 60
    if h:
        return f"{h} h {m} min"
    return f"{m} min" if m else "under a minute"
