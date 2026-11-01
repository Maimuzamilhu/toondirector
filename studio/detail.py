"""One video: watch it, see how it is going, save it, or change it."""
import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QBoxLayout, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QMessageBox, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from app import jobs

from .create import segments
from .widgets import LivePanel, Page, button, clear_layout, hbox, label, report_problem


class Player(QFrame):
    def __init__(self):
        super().__init__()
        self.setProperty("panel", True)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 10)
        v.setSpacing(8)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(360)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.video, 1)
        self.audio = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.play = button("Play", on_click=self.toggle)
        self.play.setFixedWidth(84)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setAccessibleName("Position in the video")
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.clock = label("", "small")
        v.addLayout(self._pad(hbox(self.play, self.seek, self.clock)))
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(lambda d: self.seek.setRange(0, d))
        self.player.mediaStatusChanged.connect(self.first_frame)
        self.player.playbackStateChanged.connect(
            lambda s: self.play.setText("Pause" if s == QMediaPlayer.PlayingState else "Play"))

    @staticmethod
    def _pad(lay):
        lay.setContentsMargins(12, 0, 12, 0)
        return lay

    def first_frame(self, status):
        """Shows the opening picture instead of an empty box until Play is pressed."""
        if status == QMediaPlayer.LoadedMedia:
            self.player.pause()

    def toggle(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_position(self, ms):
        self.seek.setValue(ms)
        total = self.player.duration()
        t = lambda x: f"{x // 60000}:{x // 1000 % 60:02d}"
        self.clock.setText(f"{t(ms)} / {t(total)}")

    def open(self, path):
        self.player.setSource(QUrl.fromLocalFile(str(path)))

    def release(self):
        """Lets go of the file so a new version can replace it."""
        self.player.stop()
        self.player.setSource(QUrl())


class Detail(Page):
    back = Signal()

    def __init__(self):
        super().__init__()
        self.job = None
        self.gallery_key = None
        self.loaded = None
        self.back_btn = button("Your videos", "quiet", self.back.emit)
        self.body.addLayout(hbox(self.back_btn, None))
        self.title = label("", "h1", True)
        self.body.addWidget(self.title)
        self.live = LivePanel()
        self.body.addWidget(self.live)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        self.body.addLayout(cols)
        self.player = Player()
        cols.addWidget(self.player, 3)
        side = QBoxLayout(QBoxLayout.TopToBottom)
        side.setSpacing(8)
        cols.addLayout(side, 1)
        self.b_save = button("Save video", "primary", self.save)
        self.side_box = side
        self.b_folder = button("Open folder", on_click=lambda: os.startfile(self.job.dir))
        self.b_resume = button("Resume", "primary", lambda: self.run(jobs.resume))
        self.b_stop = button("Stop", on_click=lambda: jobs.cancel(self.job.id))
        self.b_more = button("Add more scenes", on_click=self.more)
        self.b_again = button("Draw it again", on_click=self.again)
        self.b_delete = button("Delete video", "danger", self.remove)
        self.b_report = button("Report a problem", "quiet", lambda: report_problem(self.job),
                               tip="Tell the developer about inappropriate or wrong AI content")
        for b in (self.b_save, self.b_resume, self.b_stop, self.b_more, self.b_again, self.b_folder, self.b_delete, self.b_report):
            side.addWidget(b)
        side.addStretch(1)

        tabs, _ = segments(["Characters and places", "Description", "Activity"], lambda i: self.pages.setCurrentIndex(i))
        self.body.addLayout(tabs)
        self.pages = QStackedWidget()
        g = QWidget()
        self.gallery = QVBoxLayout(g)
        self.gallery.setContentsMargins(0, 0, 0, 0)
        self.gallery.setSpacing(12)
        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.description.setMinimumHeight(240)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("log", True)
        self.log.setMinimumHeight(240)
        for w in (g, self.description, self.log):
            self.pages.addWidget(w)
        self.body.addWidget(self.pages)
        self.body.addStretch(1)

    def show_job(self, jid):
        self.job = jobs.JOBS[jid]
        self.gallery_key = self.loaded = None
        self.player.release()
        self.refresh()

    def refresh(self):
        j = self.job
        if j is None or j.id not in jobs.JOBS:
            return
        self.title.setText(j.title)
        self.live.setVisible(j.status != "done")
        if j.status != "done":
            self.live.show_job(j)
        playable = j.status == "done" and j.output.exists()
        self.player.setVisible(playable)
        # beside the player when there is one, otherwise a tidy row
        self.side_box.setDirection(QBoxLayout.TopToBottom if playable else QBoxLayout.LeftToRight)
        if playable and self.loaded != str(j.output):
            self.loaded = str(j.output)
            self.player.open(j.output)
        elif j.active and self.loaded:
            self.loaded = None
            self.player.release()
        self.b_save.setVisible(playable)
        self.b_resume.setVisible(j.status == "error")
        self.b_stop.setVisible(j.active)
        self.b_more.setVisible(not j.active and j.is_story and j.output.exists())
        self.b_again.setVisible(not j.active)
        self.b_delete.setVisible(not j.active)
        text = "\n".join(j.logs)
        if self.log.toPlainText() != text:
            self.log.setPlainText(text)
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        desc = Path(j.dir) / "youtube.txt"
        self.description.setPlainText(desc.read_text(encoding="utf-8") if desc.exists() else "Appears when the video is finished.")
        self.fill_gallery()

    def fill_gallery(self):
        ddir = Path(self.job.dir) / "design"
        files = sorted(list(ddir.glob("character_*.png")) + list(ddir.glob("setpreview_*.png"))) if ddir.exists() else []
        key = tuple(f.name for f in files)
        if key == self.gallery_key:
            return
        self.gallery_key = key
        clear_layout(self.gallery)
        if not files:
            self.gallery.addWidget(label("Characters and places appear here as they are drawn.", "muted"))
            return
        people = [f for f in files if f.name.startswith("character_")]
        places = [f for f in files if f not in people]
        for group, size, per_row in ((people, (112, 150), 7), (places, (232, 130), 4)):
            for i in range(0, len(group), per_row):
                row = []
                for f in group[i:i + per_row]:
                    w = label()
                    w.setPixmap(QPixmap(str(f)).scaled(*size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    row.append(w)
                self.gallery.addLayout(hbox(*row, None, spacing=12))

    def run(self, fn, *args):
        self.player.release()
        self.loaded = None
        fn(self.job.id, *args)

    def save(self):
        name = "".join(c for c in self.job.title if c not in '<>:"/\\|?*')[:80] or "video"
        path, _ = QFileDialog.getSaveFileName(self, "Save video", str(Path.home() / "Videos" / f"{name}.mp4"), "Video (*.mp4)")
        if path:
            shutil.copyfile(self.job.output, path)

    def more(self):
        now = float(self.job.options.get("minutes") or 2)
        m, ok = QInputDialog.getDouble(self, "Add more scenes", "How long should the video be, in minutes?\n"
                                       "New scenes are written and the whole video is drawn again.", now + 3, 1, 60, 1)
        if ok:
            self.run(jobs.extend, m)

    def again(self):
        d, ok = QInputDialog.getText(self, "Draw it again", "Any direction for the artist? (optional)")
        if ok:
            self.run(jobs.restyle, d)

    def remove(self):
        box = QMessageBox(self)
        box.setWindowTitle("Delete video")
        box.setText(f"Delete \"{self.job.title}\"?")
        box.setInformativeText("The video and everything made for it are removed from this PC. This cannot be undone.")
        yes = box.addButton("Delete video", QMessageBox.DestructiveRole)
        box.addButton("Keep it", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self.player.release()
            jobs.delete(self.job.id)
            self.job = None
            self.back.emit()
