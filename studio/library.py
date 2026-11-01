"""The home page: what is being made right now, and every finished video."""
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLineEdit, QVBoxLayout, QWidget

from app import jobs

from .widgets import LivePanel, Page, VideoCard, button, hbox, label


def current_job():
    """The video being made now, or the next one waiting."""
    all_jobs = sorted(jobs.JOBS.values(), key=lambda j: j.created or 0)
    return next((j for j in all_jobs if j.status == "running"), None) or next((j for j in all_jobs if j.status == "queued"), None)


class Library(Page):
    open_job = Signal(str)
    new_video = Signal()
    thumb_ready = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.cards = {}
        self.thumbs = {}
        self.thumb_ready.connect(self.got_thumb)
        self.columns = 0
        self.order = None
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a video")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(240)
        self.search.setAccessibleName("Find a video")
        self.search.textChanged.connect(self.refresh)
        self.body.addLayout(hbox(label("Your videos", "h1"), None, self.search,
                                 button("New video", "primary", self.new_video.emit)))
        self.now = LivePanel(compact=True)
        self.now.opened.connect(self.open_job)
        self.body.addWidget(self.now)
        self.empty = QWidget()
        e = QVBoxLayout(self.empty)
        e.setAlignment(Qt.AlignCenter)
        e.setSpacing(12)
        msg = label("Your first video starts with a topic", "h2")
        msg.setAlignment(Qt.AlignCenter)
        sub = label("Type what the video should be about. The app writes it, draws it and voices it.", "muted")
        sub.setAlignment(Qt.AlignCenter)
        e.addWidget(msg)
        e.addWidget(sub)
        e.addWidget(button("Make a video", "primary", self.new_video.emit), alignment=Qt.AlignCenter)
        self.body.addWidget(self.empty)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(16)
        self.body.addWidget(self.grid_host)
        self.body.addStretch(1)
        self.refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        cols = max(2, (self.width() - 80) // 300)
        if cols != self.columns:
            self.columns = cols
            self.order = None
            self.refresh()

    def got_thumb(self, jid, path):
        """Pictures are made in the background, so the window never waits for them."""
        self.thumbs[jid] = path
        if jid in self.cards and jid in jobs.JOBS:
            self.cards[jid].thumb.set_image(path)

    def refresh(self):
        current = current_job()
        self.now.setVisible(current is not None)
        if current:
            self.now.show_job(current)
        all_jobs = sorted(jobs.JOBS.values(), key=lambda j: j.created or 0, reverse=True)
        word = self.search.text().strip().lower()
        shown = [j for j in all_jobs if j is not current and word in j.title.lower()]
        self.empty.setVisible(not all_jobs)
        for jid in list(self.cards):
            if jid not in jobs.JOBS:
                self.cards.pop(jid).deleteLater()
                self.order = None
        order = [j.id for j in shown]
        if order != self.order:  # rebuilt only when the list changes, so scrolling stays smooth
            self.order = order
            for card in self.cards.values():
                self.grid.removeWidget(card)
                card.hide()
            for i, j in enumerate(shown):
                card = self.cards.get(j.id)
                if card is None:
                    card = self.cards[j.id] = VideoCard(j.id)
                    card.opened.connect(self.open_job)
                self.grid.addWidget(card, i // max(self.columns, 1), i % max(self.columns, 1))
                card.show()
            for c in range(self.columns or 1):
                self.grid.setColumnStretch(c, 1)
        for j in shown:
            if j.id not in self.thumbs and j.status == "done":
                self.thumbs[j.id] = None
                threading.Thread(target=lambda job=j: self.thumb_ready.emit(job.id, jobs.thumbnail(job)), daemon=True).start()
            self.cards[j.id].show_job(j, self.thumbs.get(j.id) or j.preview)
