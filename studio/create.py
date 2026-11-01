"""The two ways to start: a new story, or a cartoon made from an existing video."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLineEdit,
                               QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from app import addons, config, jobs, providers
from app.story import LANGS

from .widgets import Page, button, combo, field, hbox, hms, label

LENGTHS = [(0.5, "30 seconds"), (1.0, "1 minute"), (2.0, "2 minutes"), (3.0, "3 minutes"), (5.0, "5 minutes"), (8.0, "8 minutes"),
           (10.0, "10 minutes"), (15.0, "15 minutes"), (20.0, "20 minutes"), (30.0, "30 minutes"), (45.0, "45 minutes"),
           (60.0, "1 hour")]
MUSIC = [(0.0, "No music"), (0.3, "Quiet"), (0.5, "Normal"), (0.75, "Loud")]


def segments(names, on_change):
    """A row of tabs. Returns (layout, group)."""
    row = QHBoxLayout()
    row.setSpacing(4)
    group = QButtonGroup(row)
    for i, n in enumerate(names):
        b = QPushButton(n)
        b.setProperty("seg", True)
        b.setCheckable(True)
        b.setChecked(i == 0)
        b.setCursor(Qt.PointingHandCursor)
        group.addButton(b, i)
        row.addWidget(b)
    row.addStretch(1)
    group.idClicked.connect(on_change)
    return row, group


class NewVideo(Page):
    started = Signal(str)
    get_addons = Signal()
    open_ai = Signal()

    def __init__(self):
        super().__init__()
        self.voice_path = None
        self.body.addWidget(label("New video", "h1"))
        cols = QHBoxLayout()
        cols.setSpacing(28)
        self.body.addLayout(cols)

        left = QVBoxLayout()
        left.setSpacing(14)
        cols.addLayout(left, 3)
        tabs, self.src = segments(["From a topic", "From my script", "From a YouTube link"], self.pick_source)
        left.addLayout(tabs)
        self.stack = QStackedWidget()
        self.topic = QPlainTextEdit()
        self.topic.setProperty("big", True)
        self.topic.setPlaceholderText("What is the video about?\n\nWhy do cats purr? The history of the pyramids. Five money habits.")
        self.script = QPlainTextEdit()
        self.script.setProperty("big", True)
        self.script.setPlaceholderText("Paste your script. It is split into scenes and voiced as written.")
        yt = QWidget()
        yv = QVBoxLayout(yt)
        yv.setContentsMargins(0, 0, 0, 0)
        self.yt = QLineEdit()
        self.yt.setPlaceholderText("https://www.youtube.com/watch?v=...")
        yv.addWidget(self.yt)
        yv.addWidget(label("The video is only used as research. The app writes a new, original script with its own "
                           "wording, characters and scenes.", "muted", True))
        self.transcript = QPlainTextEdit()
        self.transcript.setPlaceholderText("Paste the video's transcript here (optional). On YouTube, open the "
                                           "description and choose Show transcript.")
        self.transcript.setVisible(config.STORE)
        yv.addWidget(self.transcript, 1)
        if config.STORE:
            yv.addWidget(label("Nothing is downloaded from YouTube. The app reads the video's public title, then "
                               "researches the topic.", "small", True))
        else:
            yv.addStretch(1)
        for w in (self.topic, self.script, yt):
            self.stack.addWidget(w)
        self.stack.setMinimumHeight(220)
        left.addWidget(self.stack, 1)
        self.direction = QPlainTextEdit()
        self.direction.setFixedHeight(84)
        self.direction.setPlaceholderText("Funny tone, for kids, set it in Lahore")
        left.addLayout(field("Direction (optional)", self.direction))

        right = QVBoxLayout()
        right.setSpacing(12)
        cols.addLayout(right, 2)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.lang = combo(list(LANGS.items()), "en")
        self.voice = combo([("male", "Male"), ("female", "Female")])
        self.minutes = combo(LENGTHS, 2.0)
        self.minutes.currentIndexChanged.connect(self.update_estimate)
        self.aspect = combo([("16:9", "YouTube, wide"), ("9:16", "Shorts, tall")])
        self.height_ = combo([(1080, "1080p"), (720, "720p, faster")])
        self.height_.currentIndexChanged.connect(self.update_estimate)
        self.music = combo(MUSIC, 0.5)
        for i, (name, w) in enumerate((("Language", self.lang), ("Voice", self.voice), ("Length", self.minutes),
                                       ("Shape", self.aspect), ("Quality", self.height_), ("Music", self.music))):
            grid.addLayout(field(name, w), i // 2, i % 2)
        right.addLayout(grid)
        self.captions = QCheckBox("Word-by-word captions")
        self.captions.setChecked(True)
        right.addWidget(self.captions)

        self.my_voice_btn = button("Use my voice", on_click=self.pick_voice,
                                   tip="10 to 30 seconds of you speaking clearly in a quiet room")
        self.voice_name = label("", "small")
        self.consent = QCheckBox("This is my voice, or I have the speaker's permission")
        self.consent.hide()
        right.addLayout(hbox(self.my_voice_btn, self.voice_name, None))
        right.addWidget(self.consent)
        right.addStretch(1)
        self.estimate = label("", "muted", True)
        right.addWidget(self.estimate)
        self.error = label("", "error", True)
        right.addWidget(self.error)
        self.go = button("Make this video", "primary", self.submit)
        right.addWidget(self.go)
        self.update_estimate()

    def pick_source(self, i):
        self.stack.setCurrentIndex(i)
        self.minutes.setEnabled(i != 1)

    def update_estimate(self, *_):
        m = self.minutes.currentData()
        fps = 6 if self.height_.currentData() == 1080 else 14
        self.estimate.setText(f"Takes about {hms(300 + m * 60 * 30 / fps)} on this PC. "
                              "You can keep using the app while it works.")

    def pick_voice(self):
        if addons.status("voice")["status"] != "done":
            self.get_addons.emit()
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a recording of the voice", "",
                                              "Audio or video (*.mp3 *.wav *.m4a *.ogg *.flac *.mp4 *.mov *.webm)")
        if path:
            self.voice_path = path
            self.voice_name.setText(path.replace("\\", "/").rsplit("/", 1)[-1])
            self.consent.show()

    def submit(self):
        self.error.setText("")
        if not providers.configured():
            self.error.setText("Add an AI provider first, in Settings.")
            self.open_ai.emit()
            return
        if addons.status("ffmpeg")["status"] != "done":
            self.error.setText("Download Video tools first.")
            self.get_addons.emit()
            return
        source = ("topic", "script", "youtube")[self.src.checkedId()]
        text = {"topic": self.topic.toPlainText(), "script": self.script.toPlainText(), "youtube": self.yt.text()}[source]
        if not text.strip():
            self.error.setText({"topic": "Type a topic.", "script": "Paste your script.", "youtube": "Paste a YouTube link."}[source])
            return
        if self.voice_path and not self.consent.isChecked():
            self.error.setText("Confirm the recording is your voice, or that you have permission.")
            return
        job = jobs.new_story(source, topic=text if source == "topic" else "", script_text=text if source == "script" else "",
                             youtube_url=text if source == "youtube" else "", language=self.lang.currentData(),
                             voice=self.voice.currentData(), minutes=self.minutes.currentData(), aspect=self.aspect.currentData(),
                             height=self.height_.currentData(), description=self.direction.toPlainText(),
                             music_level=self.music.currentData(), captions=self.captions.isChecked(), voice_file=self.voice_path,
                             transcript=self.transcript.toPlainText())
        self.topic.clear()
        self.script.clear()
        self.yt.clear()
        self.transcript.clear()
        self.direction.clear()
        self.voice_path = None
        self.voice_name.setText("")
        self.consent.setChecked(False)
        self.consent.hide()
        self.started.emit(job.id)


class Convert(Page):
    started = Signal(str)
    get_addons = Signal()

    NEEDS = ("ffmpeg", "whisper", "tracking", "lipsync")

    def __init__(self):
        super().__init__()
        self.path = None
        self.body.addWidget(label("Cartoon from a video", "h1"))
        self.body.addWidget(label("Keeps the voices and acting of a video and redraws it in 2D. Only use videos you "
                                  "own or have permission to use.", "muted", True))
        self.missing = QFrame()
        self.missing.setProperty("panel", True)
        m = QHBoxLayout(self.missing)
        m.setContentsMargins(16, 12, 16, 12)
        m.addWidget(label("This needs a few free downloads first.", "h3"))
        m.addStretch(1)
        m.addWidget(button("Open Add-ons", on_click=self.get_addons.emit))
        self.body.addWidget(self.missing)
        cols = QHBoxLayout()
        cols.setSpacing(28)
        self.body.addLayout(cols)

        left = QVBoxLayout()
        left.setSpacing(14)
        cols.addLayout(left, 3)
        tabs, self.src = segments(["Video file"] if config.STORE else ["Video file", "YouTube link"], self.pick_source)
        left.addLayout(tabs)
        self.stack = QStackedWidget()
        f = QWidget()
        fv = QVBoxLayout(f)
        fv.setContentsMargins(0, 0, 0, 0)
        self.file_btn = button("Choose a video", on_click=self.choose)
        self.file_name = label("", "muted")
        fv.addLayout(hbox(self.file_btn, self.file_name, None))
        fv.addStretch(1)
        self.yt = QLineEdit()
        self.yt.setPlaceholderText("https://www.youtube.com/watch?v=...")
        ytw = QWidget()
        yv = QVBoxLayout(ytw)
        yv.setContentsMargins(0, 0, 0, 0)
        yv.addWidget(self.yt)
        yv.addStretch(1)
        self.stack.addWidget(f)
        self.stack.addWidget(ytw)
        self.stack.setFixedHeight(70)
        left.addWidget(self.stack)
        self.desc = QPlainTextEdit()
        self.desc.setFixedHeight(110)
        self.desc.setPlaceholderText("Bigger heads, warm colours, make the studio cosy")
        left.addLayout(field("Direction for the artist (optional)", self.desc))
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        cols.addLayout(right, 2)
        self.art = combo([("ai", "Drawn by AI"), ("explainer", "Simple characters, polished scenes"), ("simple", "Built-in character, no AI")])
        self.motion = combo([("director", "Clean acting, for talking videos"), ("animator", "Full body, for any video"),
                             ("tracking", "Copy every frame exactly")])
        self.people = combo([(1, "Main person only"), (2, "Up to 2"), (4, "Up to 4"), (8, "Everyone, up to 8")], 8)
        self.height_ = combo([(480, "480p, fastest"), (720, "720p"), (1080, "1080p")], 720)
        self.quality = combo([("full", "Every frame"), ("fast", "Every second frame, faster")])
        self.subs = QCheckBox("Add subtitles")
        self.start = QLineEdit()
        self.start.setPlaceholderText("Beginning")
        self.end = QLineEdit()
        self.end.setPlaceholderText("End of video")
        for box in (self.start, self.end):
            box.setValidator(QIntValidator(0, 86400, box))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for i, (name, w) in enumerate((("Art", self.art), ("Movement", self.motion), ("Characters", self.people),
                                       ("Quality", self.height_), ("Start at (seconds)", self.start), ("Stop at (seconds)", self.end))):
            grid.addLayout(field(name, w), i // 2, i % 2)
        right.addLayout(grid)
        right.addLayout(field("Tracking", self.quality))
        right.addWidget(self.subs)
        right.addStretch(1)
        self.error = label("", "error", True)
        right.addWidget(self.error)
        right.addWidget(button("Make this cartoon", "primary", self.submit))
        self.refresh()

    def refresh(self):
        self.missing.setVisible(any(addons.status(a)["status"] != "done" for a in self.NEEDS))

    def pick_source(self, i):
        self.stack.setCurrentIndex(i)

    def choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", "", "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if path:
            self.path = path
            self.file_name.setText(path.replace("\\", "/").rsplit("/", 1)[-1])

    def submit(self):
        self.error.setText("")
        if any(addons.status(a)["status"] != "done" for a in self.NEEDS):
            self.error.setText("Finish the downloads in Add-ons first.")
            return
        yt = self.yt.text().strip() if self.src.checkedId() == 1 else ""
        path = self.path if self.src.checkedId() == 0 else None
        if not path and not yt:
            self.error.setText("Choose a video first." if self.src.checkedId() == 0 else "Paste a YouTube link.")
            return
        job = jobs.new_convert(path or "", yt, self.desc.toPlainText(), self.height_.currentData(), self.quality.currentData(),
                               self.subs.isChecked(), int(self.start.text() or 0), int(self.end.text() or 0), self.art.currentData(),
                               self.people.currentData(), self.motion.currentData())
        self.path = None
        self.file_name.setText("")
        self.yt.clear()
        self.desc.clear()
        self.started.emit(job.id)
