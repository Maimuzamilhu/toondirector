"""First-run setup: where to keep things, which AI to use, what to download. Every step can be
changed later, and everything except Video tools can be skipped."""
import sys

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QFrame, QLineEdit, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from app import addons, config, hardware, providers

from .settings import AddonRow, Bg
from .widgets import button, combo, field, hbox, label

STARTERS = ["groq", "gemini", "openrouter", "deepseek", "ollama"]


class Onboarding(QWidget):
    finished = Signal(bool)  # True = the data folder changed, so restart

    def __init__(self):
        super().__init__()
        self.setObjectName("page")
        self.bg = Bg()
        self.new_folder = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 36, 0, 28)
        box = QWidget()
        box.setFixedWidth(640)
        self.main = QVBoxLayout(box)
        self.main.setContentsMargins(0, 0, 0, 0)
        self.main.setSpacing(18)
        outer.addWidget(box, 1, Qt.AlignHCenter)
        self.step_text = label("", "small")
        self.main.addWidget(self.step_text)
        self.stack = QStackedWidget()
        self.main.addWidget(self.stack, 1)
        self.pages = [self.welcome(), self.ai(), self.extras(), self.ready()]
        for p in self.pages:  # a step taller than the window scrolls; Back and Continue never move
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            area.setWidget(p)
            self.stack.addWidget(area)
        self.back_btn = button("Back", "quiet", lambda: self.go(-1))
        self.next_btn = button("Continue", "primary", lambda: self.go(1))
        self.main.addLayout(hbox(self.back_btn, None, self.next_btn))
        self.go(0)

    def go(self, d):
        i = self.stack.currentIndex() + d
        if i >= len(self.pages):
            config.save_settings({"ONBOARDED": "1"})
            self.finished.emit(bool(self.new_folder))
            return
        self.stack.setCurrentIndex(i)
        self.step_text.setText(f"Step {i + 1} of {len(self.pages)}")
        self.back_btn.setVisible(i > 0)
        self.next_btn.setText("Start making videos" if i == len(self.pages) - 1 else "Continue")

    def _page(self, title, text):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        v.addWidget(label(title, "h1", True))
        v.addWidget(label(text, "muted", True))
        return w, v

    def welcome(self):
        w, v = self._page("Make animated videos on your PC",
                          "Give it a topic and it writes, draws, voices and edits a video. Setup takes a couple of minutes.")
        self.folder = label(str(config.DATA_DIR), "h3", True)
        v.addLayout(field("Videos and models will be kept in", self.folder))
        v.addLayout(hbox(button("Choose another folder", on_click=self.pick_folder), None))
        v.addStretch(1)
        return w

    def pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a folder with plenty of free space", str(config.DATA_DIR))
        if path:
            self.new_folder = path
            config.move_data_dir(path)
            self.folder.setText(path)

    def ai(self):
        w, v = self._page("Connect an AI", "It writes the scripts and draws the scenes. Free options work, and you can add "
                                           "more later. If one reaches its limit, the next takes over.")
        self.choice = combo([(p, f"{providers.PROVIDERS[p]['name']}: {providers.PROVIDERS[p]['note']}") for p in STARTERS])
        self.choice.currentIndexChanged.connect(self.chosen)
        v.addLayout(field("Provider", self.choice))
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your key")
        self.key_label = field("API key", self.key)
        v.addLayout(self.key_label)
        self.site = button("Get a key", on_click=lambda: QDesktopServices.openUrl(QUrl(providers.PROVIDERS[self.choice.currentData()]["key_url"])))
        self.test_btn = button("Test connection", on_click=self.test)
        v.addLayout(hbox(self.site, self.test_btn, None))
        self.result = label("", "muted", True)
        v.addWidget(self.result)
        v.addWidget(label("Your topics and scripts are sent to the provider you connect, under its terms.", "small", True))
        self.chosen()
        v.addStretch(1)
        return w

    def chosen(self, *_):
        p = providers.PROVIDERS[self.choice.currentData()]
        self.key.setVisible(not p.get("nokey"))
        self.site.setText("Download" if p.get("nokey") else "Get a key")
        self.result.setText("")

    def test(self):
        pid = self.choice.currentData()
        providers.update(pid, key=self.key.text().strip(), on=True)
        self.result.setText("Testing...")
        self.bg.go(lambda: providers.test(pid), self.tested)

    def tested(self, ok, res):
        self.result.setProperty("role", "good" if ok else "error")
        self.result.style().polish(self.result)
        self.result.setText(f"Connected, using {res[0]}." if ok else
                            "Could not connect. Check the key, or for a provider on this PC, that it is running.")

    def extras(self):
        w, v = self._page("Choose your downloads", "Video tools are needed for everything. The rest are optional, and you "
                                                   "can get them later in Add-ons.")
        self.advice = label("", "muted", True)
        v.addWidget(self.advice)
        self.rows = {a: AddonRow(a) for a in addons.ITEMS}
        for r in self.rows.values():
            v.addWidget(r)
        self.bg.go(hardware.summary, self.advise)
        return w

    def advise(self, ok, s):
        if not ok:
            return
        text = f"This PC has {s['ram']} GB of memory and {s['free_gb']} GB free."
        text += " Your graphics card will speed up voice and export." if s["nvidia"] else " Everything works without a graphics card."
        self.advice.setText(text)

    def refresh_addon(self, aid):
        if aid in self.rows:
            self.rows[aid].refresh()

    def ready(self):
        w, v = self._page("You are ready", "Downloads keep going in the background. Your first video can be a short "
                                           "one to see how it looks.")
        v.addStretch(1)
        return w
