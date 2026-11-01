"""The main window: side rail, pages, notifications, tray, one running copy only."""
import ctypes
import sys
import time

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QProcess, QPropertyAnimation, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QMainWindow,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QSystemTrayIcon, QVBoxLayout,
                               QWidget)

from app import addons, config, jobs, notify, providers

from . import theme
from .create import Convert, NewVideo
from .detail import Detail
from .library import Library, current_job
from .onboarding import Onboarding
from .settings import Addons, Settings
from .taskbar import ERROR, NONE, NORMAL, Taskbar
from .widgets import (RailJob, Toast, app_qicon, bell_icon, button, clear_layout, hbox, label, line_icon,
                      reduced_motion)

APP_ID = "ToonDirector.Desktop"


class Bridge(QObject):
    """Engine threads talk to the window only through these signals."""
    job_changed = Signal(str)
    note = Signal(dict)
    addon = Signal(str)


class Bell(QFrame):
    """The list of past notifications."""
    open_job = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("popover")
        self.setFixedSize(380, 480)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.addLayout(hbox(label("Notifications", "h2"), None, button("Clear all", "quiet", self.clear)))
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        self.list = QVBoxLayout(host)
        self.list.setAlignment(Qt.AlignTop)
        self.list.setSpacing(12)
        area.setWidget(host)
        v.addWidget(area)
        self.hide()

    def clear(self):
        notify.clear()
        self.fill()

    def fill(self):
        clear_layout(self.list)
        items = notify.items()
        if not items:
            self.list.addWidget(label("Nothing yet. You will see finished videos and problems here.", "muted", True))
        for n in items:
            role = {"done": "good", "error": "error"}.get(n["kind"], "h3")
            b = QVBoxLayout()
            b.setSpacing(1)
            b.addWidget(label(n["title"], role, True))
            if n.get("body"):
                b.addWidget(label(n["body"], "small", True))
            b.addWidget(label(time.strftime("%d %b, %H:%M", time.localtime(n["time"])), "small"))
            self.list.addLayout(b)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ToonDirector")
        self.setMinimumSize(1040, 680)
        self.resize(1320, 840)
        self.really_quit = False
        self.dirty = set()
        self.bridge = Bridge()
        self.bridge.job_changed.connect(lambda j: self.dirty.add(j))
        self.bridge.note.connect(self.on_note)
        self.bridge.addon.connect(self.on_addon)
        jobs.listeners.append(self.bridge.job_changed.emit)
        notify.listeners.append(self.bridge.note.emit)
        addons.on_change.append(self.bridge.addon.emit)
        providers.on_switch.append(lambda a, b, why: notify.add("info", f"Switched to {b}", f"{a} {why}."))

        root = QWidget()
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.setCentralWidget(root)
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(244)
        r = QVBoxLayout(rail)
        r.setContentsMargins(14, 20, 14, 14)
        r.setSpacing(4)
        mark = label()
        mark.setPixmap(app_qicon().pixmap(28, 28))
        word = label("ToonDirector")
        word.setObjectName("wordmark")
        r.addLayout(hbox(mark, word, None, spacing=9))
        r.addSpacing(18)

        self.pages = QStackedWidget()
        self.library = Library()
        self.new = NewVideo()
        self.convert = Convert()
        self.addons_page = Addons()
        self.settings = Settings()
        self.detail = Detail()
        self.group = QButtonGroup(self)
        self.nav = {}
        for name, ic, page in (("Your videos", "videos", self.library), ("New video", "new", self.new),
                               ("Cartoon from a video", "cartoon", self.convert), ("Add-ons", "addons", self.addons_page),
                               ("Settings", "settings", self.settings)):
            b = QPushButton(" " + name)
            b.setProperty("nav", True)
            b.setCheckable(True)
            b.setIcon(line_icon(ic))
            b.setIconSize(QSize(20, 20))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda *_, p=page: self.show_page(p))
            self.group.addButton(b)
            r.addWidget(b)
            self.nav[page] = b
            self.pages.addWidget(page)
        self.pages.addWidget(self.detail)
        r.addStretch(1)
        self.rail_job = RailJob()
        self.rail_job.hide()
        self.rail_job.opened.connect(self.open_job)
        r.addWidget(self.rail_job)
        r.addSpacing(6)
        self.bell_btn = QPushButton(" Notifications")
        self.bell_btn.setProperty("nav", True)
        self.bell_btn.setIconSize(QSize(20, 20))
        self.bell_btn.setCursor(Qt.PointingHandCursor)
        self.bell_btn.clicked.connect(self.toggle_bell)
        r.addWidget(self.bell_btn)
        h.addWidget(rail)
        h.addWidget(self.pages, 1)

        self.bell = Bell(root)
        self.toasts = []
        self.library.open_job.connect(self.open_job)
        self.library.new_video.connect(lambda: self.show_page(self.new))
        for page in (self.new, self.convert):
            page.started.connect(self.open_job)
            page.get_addons.connect(lambda: self.show_page(self.addons_page))
        self.new.open_ai.connect(lambda: self.show_page(self.settings))
        self.detail.back.connect(lambda: self.show_page(self.library))
        self.settings.relaunch.connect(self.relaunch)
        QShortcut(QKeySequence("Ctrl+N"), self, lambda: self.show_page(self.new))

        self.tray = QSystemTrayIcon(app_qicon(), self)
        self.tray.setToolTip("ToonDirector")
        menu = QMenu()
        menu.addAction(QAction("Open ToonDirector", menu, triggered=self.bring_up))
        menu.addAction(QAction("Quit", menu, triggered=self.quit_now))
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r_: r_ == QSystemTrayIcon.Trigger and self.bring_up())
        self.tray.messageClicked.connect(self.bring_up)
        self.tray.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(500)
        self.taskbar = None
        self.update_badges()
        self.show_page(self.library)

        if not config.ONBOARDED:
            self.onboarding = Onboarding()
            self.onboarding.finished.connect(self.done_onboarding)
            self.pages.addWidget(self.onboarding)
            self.pages.setCurrentWidget(self.onboarding)
            rail.hide()
            self.rail = rail
        else:
            self.onboarding = None
            self.rail = rail

    def done_onboarding(self, restart):
        if restart:
            self.relaunch()
            return
        self.rail.show()
        self.pages.removeWidget(self.onboarding)
        self.onboarding.deleteLater()
        self.onboarding = None
        self.show_page(self.library)

    def fade_in(self, page):
        """A quick fade answers the click; the video page skips it because of its player."""
        if page is self.detail or reduced_motion():
            return
        fx = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(fx)
        self.fade = QPropertyAnimation(fx, b"opacity", self)
        self.fade.setDuration(130)
        self.fade.setStartValue(0.0)
        self.fade.setEndValue(1.0)
        self.fade.setEasingCurve(QEasingCurve.OutCubic)
        self.fade.finished.connect(lambda: page.setGraphicsEffect(None))
        self.fade.start()

    def show_page(self, page):
        self.bell.hide()
        if self.pages.currentWidget() is not page:
            self.fade_in(page)
        self.pages.setCurrentWidget(page)
        if page in self.nav:
            self.nav[page].setChecked(True)
        if page is self.library:
            self.library.refresh()
        if page is self.convert:
            self.convert.refresh()

    def open_job(self, jid):
        self.detail.show_job(jid)
        self.pages.setCurrentWidget(self.detail)
        self.nav[self.library].setChecked(True)
        self.bell.hide()

    def tick(self):
        """Screen updates are grouped, so a fast job never makes the window stutter.
        While a video is being made the clocks move even between engine updates."""
        current = current_job()
        changed, self.dirty = self.dirty, set()
        if current or changed:
            if self.pages.currentWidget() is self.library:
                self.library.refresh()
            if self.detail.job and self.pages.currentWidget() is self.detail and (
                    self.detail.job.active or self.detail.job.id in changed):
                self.detail.refresh()
        self.rail_job.setVisible(current is not None)
        if current:
            self.rail_job.show_job(current)
        self.show_taskbar(current, changed)

    def show_taskbar(self, current, changed):
        if self.taskbar is None:
            self.taskbar = Taskbar(int(self.winId()))
        if current and current.status == "running":
            self.taskbar.set(NORMAL, current.progress)
            self.setWindowTitle(f"{current.progress:.0f}% - ToonDirector")
            self.tray.setToolTip(f"Making a video, {current.progress:.0f}%")
        elif not current:
            failed = any(jobs.JOBS[j].status == "error" for j in changed if j in jobs.JOBS)
            self.taskbar.set(ERROR if failed else NONE, 100)
            self.setWindowTitle("ToonDirector")
            self.tray.setToolTip("ToonDirector")

    def on_note(self, n):
        self.update_badges()
        toast = Toast(self.centralWidget(), n)
        toast.clicked.connect(lambda note: note.get("job") in jobs.JOBS and self.open_job(note["job"]))
        toast.destroyed.connect(lambda *_: self.place_toasts())
        self.toasts.append(toast)
        self.toasts = self.toasts[-4:]
        toast.show()
        self.place_toasts()
        if config.NOTIFY_DESKTOP != "0" and not self.isActiveWindow():
            icon = QSystemTrayIcon.Critical if n["kind"] == "error" else QSystemTrayIcon.Information
            self.tray.showMessage(n["title"], n.get("body", ""), icon, 8000)

    def place_toasts(self):
        y = self.centralWidget().height() - 18
        alive = []
        for t in reversed(self.toasts):
            try:
                t.adjustSize()
                y -= t.height()
                t.move(self.centralWidget().width() - t.width() - 28, y)
                y -= 10
                alive.append(t)
            except RuntimeError:
                pass
        self.toasts = alive[::-1]

    def on_addon(self, aid):
        self.addons_page.refresh(aid)
        self.convert.refresh()
        if self.onboarding:
            self.onboarding.refresh_addon(aid)

    def update_badges(self):
        n = notify.unread()
        self.bell_btn.setText(f" Notifications ({n})" if n else " Notifications")
        self.bell_btn.setIcon(bell_icon(n))

    def toggle_bell(self):
        if self.bell.isVisible():
            self.bell.hide()
            return
        notify.mark_all_read()
        self.update_badges()
        self.bell.fill()
        self.bell.setFixedHeight(min(480, 120 + 92 * max(len(notify.items()), 1)))
        self.bell.move(254, max(self.centralWidget().height() - self.bell.height() - 14, 10))
        self.bell.raise_()
        self.bell.show()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.place_toasts()

    def bring_up(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def quit_now(self):
        self.really_quit = True
        self.close()

    def relaunch(self):
        self.really_quit = True
        QProcess.startDetached(sys.executable, sys.argv[1:] if config.FROZEN else ["-m", "studio"])
        self.close()

    def closeEvent(self, e):
        """Closing while a video is being made asks first, because that video would stop."""
        if not self.really_quit and jobs.active_count():
            box = QMessageBox(self)
            box.setWindowTitle("A video is being made")
            box.setText("Keep making it in the background?")
            box.setInformativeText("If you quit, it stops and can be resumed later.")
            keep = box.addButton("Keep making it", QMessageBox.AcceptRole)
            quit_ = box.addButton("Quit", QMessageBox.DestructiveRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is keep:
                self.hide()
                self.tray.showMessage("Still working", "ToonDirector keeps going here in the tray.", QSystemTrayIcon.Information, 4000)
                e.ignore()
                return
            if box.clickedButton() is not quit_:
                e.ignore()
                return
        self.tray.hide()
        e.accept()
        QApplication.quit()


def dark_title_bar(win):
    """Keeps the Windows title bar in the same dark colour as the app."""
    try:
        v = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(int(win.winId()), 20, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


class DarkDialogs(QObject):
    """Every dialog gets the same dark title bar as the main window."""

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Show and isinstance(obj, QDialog):
            dark_title_bar(obj)
        return False


def main():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    app = QApplication(sys.argv)
    app.setApplicationName("ToonDirector")
    app.setQuitOnLastWindowClosed(False)
    sock = QLocalSocket()
    sock.connectToServer(APP_ID)
    if sock.waitForConnected(200):  # already open: bring that window forward instead
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(200)
        return 0
    QLocalServer.removeServer(APP_ID)
    server = QLocalServer()
    server.listen(APP_ID)

    app.setStyle("Fusion")
    pal = app.palette()
    for role, c in ((QPalette.Window, theme.NIGHT), (QPalette.Base, theme.SLATE), (QPalette.Text, theme.TEXT),
                    (QPalette.WindowText, theme.TEXT), (QPalette.ButtonText, theme.TEXT), (QPalette.Button, theme.SLATE),
                    (QPalette.PlaceholderText, theme.MUTED), (QPalette.Highlight, theme.MARIGOLD)):
        pal.setColor(role, QColor(c))
    app.setPalette(pal)
    app.setStyleSheet(theme.QSS)
    app.setWindowIcon(app_qicon())
    dialogs = DarkDialogs(app)
    app.installEventFilter(dialogs)
    providers.load()
    jobs.start()
    win = Window()
    dark_title_bar(win)
    win.show()
    server.newConnection.connect(lambda: (server.nextPendingConnection(), win.bring_up()))
    return app.exec()
