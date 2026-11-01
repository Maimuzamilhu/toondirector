"""Add-ons, AI providers and every other setting."""
import threading

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from app import addons, config, hardware, providers

from .widgets import Page, button, combo, dot_icon, field, hbox, label, report_problem

GROUPS = {"free": "Free to start", "paid": "Paid", "local": "On this PC"}


class Bg(QObject):
    """Runs slow work off the window thread and hands the result back."""
    done = Signal(object, object)

    def go(self, work, then):
        def run():
            try:
                self.done.emit(then, (True, work()))
            except Exception as e:
                self.done.emit(then, (False, e))
        threading.Thread(target=run, daemon=True).start()

    def __init__(self):
        super().__init__()
        self.done.connect(lambda then, res: then(*res))


class AddonRow(QFrame):
    def __init__(self, aid):
        super().__init__()
        self.aid = aid
        self.setProperty("row", True)
        info = addons.ITEMS[aid]
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 12, 0, 14)
        v.setSpacing(5)
        self.action = button("Download", on_click=self.act)
        self.remove = button("Remove", "quiet", lambda: addons.remove(aid))
        self.remove.setVisible(False)
        need = "Required" if info.get("required") else info["size"]
        v.addLayout(hbox(label(info["name"], "h3"), label(need, "small"), None, self.remove, self.action))
        v.addWidget(label(info["about"], "muted", True))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.note = label("", "small", True)
        v.addWidget(self.bar)
        v.addWidget(self.note)
        self.refresh()

    def act(self):
        if addons.status(self.aid)["status"] == "running":
            addons.cancel(self.aid)
        else:
            addons.install(self.aid)

    def refresh(self):
        s = addons.status(self.aid)
        run = s["status"] == "running"
        self.bar.setVisible(run)
        self.bar.setValue(int(s["progress"] * 1000))
        self.note.setText(s["message"] if s["status"] != "done" else "")
        self.note.setVisible(bool(self.note.text()))
        self.action.setText({"running": "Pause", "done": "Installed", "error": "Try again"}.get(s["status"], "Download"))
        self.action.setEnabled(s["status"] != "done")
        self.remove.setVisible(s["status"] == "done" and self.aid == "voice")


class Addons(Page):
    def __init__(self):
        super().__init__()
        self.body.addWidget(label("Add-ons", "h1"))
        self.body.addWidget(label("Download only what you use. Downloads run in the background and continue where they "
                                  "stopped.", "muted", True))
        self.rows = {a: AddonRow(a) for a in addons.ITEMS}
        for r in self.rows.values():
            self.body.addWidget(r)
        self.body.addStretch(1)

    def refresh(self, aid=None):
        for a, r in self.rows.items():
            if aid in (None, a):
                r.refresh()


class ProviderPane(QWidget):
    """Key, model and test for one provider."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.pid = None
        self.bg = Bg()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        self.name = label("", "h2")
        self.on = QCheckBox("Use this provider")
        self.on.toggled.connect(self.save)
        self.note = label("", "muted", True)
        v.addLayout(hbox(self.name, None, self.on))
        v.addWidget(self.note)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your key")
        self.key.editingFinished.connect(self.save)
        self.key_box = QWidget()
        kb = QVBoxLayout(self.key_box)
        kb.setContentsMargins(0, 0, 0, 0)
        self.get_key = button("Get a key", on_click=self.open_site)
        show = button("Show", "quiet", lambda: self.key.setEchoMode(
            QLineEdit.Normal if self.key.echoMode() == QLineEdit.Password else QLineEdit.Password))
        kb.addLayout(field("API key", self.key))
        kb.addLayout(hbox(self.get_key, show, None))
        v.addWidget(self.key_box)
        self.url = QLineEdit()
        self.url.setPlaceholderText("http://localhost:8080/v1")
        self.url.editingFinished.connect(self.save)
        self.url_box = QWidget()
        ub = QVBoxLayout(self.url_box)
        ub.setContentsMargins(0, 0, 0, 0)
        ub.addLayout(field("Server address", self.url))
        v.addWidget(self.url_box)
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.lineEdit().setPlaceholderText("Chosen automatically")
        self.model.lineEdit().editingFinished.connect(self.save)
        self.model.activated.connect(lambda *_: self.save())
        v.addLayout(field("Model", self.model))
        self.test_btn = button("Test connection", on_click=self.test)
        self.result = label("", "muted", True)
        v.addLayout(hbox(self.test_btn, None))
        v.addWidget(self.result)
        v.addStretch(1)

    def show_provider(self, pid):
        self.pid = None  # no saving while the boxes fill up
        p = providers._info(pid)
        self.name.setText(p["name"])
        self.note.setText(p["note"])
        self.on.setChecked(p["on"])
        self.key.setText(p["key"])
        self.key_box.setVisible(not p.get("nokey"))
        self.url_box.setVisible(pid == "custom" or bool(p.get("nokey")))
        self.url.setText(p["url"])
        self.model.clear()
        self.model.setEditText(p["model"])
        self.get_key.setVisible(bool(p["key_url"]))
        self.result.setText("")
        self.pid = pid

    def save(self, *_):
        if not self.pid:
            return
        providers.update(self.pid, key=self.key.text().strip(), model=self.model.currentText().strip(),
                         on=self.on.isChecked(), url=self.url.text().strip() if self.url_box.isVisible() else "")
        self.changed.emit()

    def open_site(self):
        QDesktopServices.openUrl(QUrl(providers.PROVIDERS[self.pid]["key_url"]))

    def test(self):
        self.save()
        if not self.on.isChecked():
            self.on.setChecked(True)
        self.result.setProperty("role", "muted")
        self.result.setText("Testing...")
        self.test_btn.setEnabled(False)
        pid = self.pid
        self.bg.go(lambda: (providers.test(pid), providers.list_models(pid)), self.tested)

    def tested(self, ok, res):
        self.test_btn.setEnabled(True)
        self.result.setProperty("role", "good" if ok else "error")
        self.result.style().polish(self.result)
        if ok:
            (model, _), ids = res
            keep = self.model.currentText()
            self.model.clear()
            self.model.addItems(ids)
            self.model.setEditText(keep)
            self.result.setText(f"It works, using {model}.")
        else:
            code = getattr(res, "status_code", None)
            self.result.setText({401: "The key was refused. Check it and try again.",
                                 402: "This account is out of credit.",
                                 429: "The limit is reached for now. Try again later."}.get(
                code, "Could not connect. For a provider on this PC, check that it is running."))


class AiProviders(QWidget):
    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(24)
        left = QVBoxLayout()
        left.addWidget(label("If one stops or runs out, the next takes over. Drag to change the order.", "muted", True))
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setFixedWidth(280)
        self.list.setMinimumHeight(430)
        self.list.setAccessibleName("AI providers, in the order they are tried")
        self.list.model().rowsMoved.connect(self.moved)
        self.list.currentItemChanged.connect(self.selected)
        left.addWidget(self.list)
        h.addLayout(left)
        self.pane = ProviderPane()
        self.pane.changed.connect(self.mark)
        h.addWidget(self.pane, 1)
        self.fill()

    def fill(self):
        self.list.clear()
        st = providers.load()
        for pid in st["order"]:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, pid)
            self.list.addItem(it)
        self.mark()
        self.list.setCurrentRow(0)

    def mark(self):
        for i in range(self.list.count()):
            it = self.list.item(i)
            p = providers._info(it.data(Qt.UserRole))
            usable = p["on"] and (p["key"] or p.get("nokey"))
            used = providers.used_today(p["id"])
            it.setText(f"{p['name']}\n{f'On, {used} requests today' if usable else GROUPS[p['group']]}")
            it.setIcon(dot_icon(bool(usable)))

    def moved(self, *_):
        providers.set_order([self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())])

    def selected(self, cur, _):
        if cur:
            self.pane.show_provider(cur.data(Qt.UserRole))


class Settings(Page):
    relaunch = Signal()

    def __init__(self):
        super().__init__()
        self.bg = Bg()
        self.body.addWidget(label("Settings", "h1"))
        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.stack = QStackedWidget()
        self.nav_btns = []
        for i, (name, page) in enumerate((("AI providers", AiProviders()), ("Research", self.research_page()),
                                          ("Quality", self.quality_page()), ("This PC", self.pc_page()),
                                          ("Storage", self.storage_page()), ("Notifications", self.notify_page()),
                                          ("Help", self.help_page()))):
            b = QPushButton(name)
            b.setProperty("seg", True)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setChecked(i == 0)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda *_, k=i: self.stack.setCurrentIndex(k))
            nav.addWidget(b)
            self.stack.addWidget(page)
        nav.addStretch(1)
        self.body.addLayout(nav)
        self.body.addWidget(self.stack)
        self.body.addStretch(1)

    def _simple(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(14)
        v.setAlignment(Qt.AlignTop)
        return w, v

    def research_page(self):
        w, v = self._simple()
        v.addWidget(label("Before writing, the app looks your topic up and bases the facts on what it finds. Wikipedia "
                          "always works. A free Tavily key adds the whole web. Check important facts yourself before you publish.",
                          "muted", True))
        self.tavily = QLineEdit(config.TAVILY_API_KEY)
        self.tavily.setEchoMode(QLineEdit.Password)
        self.tavily.setPlaceholderText("Paste your key (optional)")
        self.tavily.setMaximumWidth(460)
        self.tavily.editingFinished.connect(lambda: config.save_settings({"TAVILY_API_KEY": self.tavily.text().strip()}))
        v.addLayout(field("Tavily key", self.tavily, "1,000 free searches a month."))
        v.addLayout(hbox(button("Get a free key", on_click=lambda: QDesktopServices.openUrl(QUrl("https://tavily.com"))), None))
        return w

    def quality_page(self):
        w, v = self._simple()
        c1 = combo([("0", "No review, fastest"), ("1", "One review"), ("2", "Two reviews"), ("3", "Three reviews, best")], config.REVIEW_PASSES)
        c1.currentIndexChanged.connect(lambda: config.save_settings({"REVIEW_PASSES": c1.currentData()}))
        c2 = combo([("5", "5 frames a second, cheapest"), ("10", "10 frames a second"), ("15", "15 frames a second"),
                    ("30", "30 frames a second"), ("all", "Every frame")], config.WATCH_FPS)
        c2.currentIndexChanged.connect(lambda: config.save_settings({"WATCH_FPS": c2.currentData()}))
        c3 = combo([("tiny", "Fastest"), ("base", "Balanced"), ("small", "More accurate"), ("medium", "Most accurate, slow")], config.WHISPER_MODEL)
        c3.currentIndexChanged.connect(lambda: config.save_settings({"WHISPER_MODEL": c3.currentData()}))
        for c in (c1, c2, c3):
            c.setMaximumWidth(340)
        v.addLayout(field("How often the AI checks its drawings", c1, "More reviews use more of your AI quota."))
        v.addLayout(field("How closely the AI watches a video", c2, "Used when turning a video into a cartoon."))
        v.addLayout(field("Speech recognition", c3))
        return w

    def pc_page(self):
        w, v = self._simple()
        self.pc = label("Checking this PC...", "muted", True)
        v.addWidget(self.pc)
        self.gpu = QCheckBox("Use the graphics card when it helps")
        self.gpu.setChecked(config.USE_GPU != "off")
        self.gpu.toggled.connect(lambda on: config.save_settings({"USE_GPU": "auto" if on else "off"}))
        v.addWidget(self.gpu)
        v.addWidget(label("A graphics card speeds up video export and voice conversion, and lets you run AI models on "
                          "this PC with Ollama or LM Studio. Drawing the frames always uses the processor.", "muted", True))
        self.bg.go(hardware.summary, self.show_pc)
        return w

    def show_pc(self, ok, s):
        if not ok:
            self.pc.setText("Could not read this PC's details.")
            return
        lines = [f"{s['cores']} processor threads, {s['ram']} GB memory, {s['free_gb']} GB free on the data drive."]
        lines.append("Graphics: " + (", ".join(s["names"]) or "unknown") + ".")
        if s["nvidia"]:
            lines.append(f"NVIDIA card found with {s['nvidia'][1]} GB of memory.")
        lines.append(f"Fast video export: {s['encoder']}." if s["encoder"] else "Fast video export: not available, using the processor.")
        self.pc.setText("\n".join(lines))

    def storage_page(self):
        w, v = self._simple()
        self.folder = label(str(config.DATA_DIR), "h3", True)
        v.addLayout(field("Videos and models are kept in", self.folder))
        v.addLayout(hbox(button("Open folder", on_click=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.DATA_DIR)))),
                         button("Change folder", on_click=self.change_folder), None))
        v.addWidget(label("A new folder starts empty. Move your files there yourself to keep them.", "small", True))
        return w

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a folder for videos and models", str(config.DATA_DIR))
        if path:
            config.move_data_dir(path)
            self.relaunch.emit()

    def help_page(self):
        w, v = self._simple()
        v.addWidget(label(f"ToonDirector {config.VERSION}", "h3"))
        v.addWidget(label("Scripts, pictures and acting are made by generative AI, so they can contain mistakes. Check "
                          "facts before you publish. Your topics and scripts are sent to the AI provider you connect, "
                          "under that provider's terms. Voices and videos are processed on this PC.", "muted", True))
        row = [button("Report a problem", on_click=report_problem)]
        if config.PRIVACY_URL:
            row.append(button("Privacy policy", on_click=lambda: QDesktopServices.openUrl(QUrl(config.PRIVACY_URL))))
        row.append(None)
        v.addLayout(hbox(*row))
        v.addWidget(label("Use Report a problem to tell the developer about inappropriate or wrong AI content.", "small", True))
        return w

    def notify_page(self):
        w, v = self._simple()
        c = QCheckBox("Show Windows notifications when the app is in the background")
        c.setChecked(config.NOTIFY_DESKTOP != "0")
        c.toggled.connect(lambda on: config.save_settings({"NOTIFY_DESKTOP": "1" if on else "0"}))
        v.addWidget(c)
        v.addWidget(label("You are told when a video finishes or stops, when a download ends, and when the app switches "
                          "to another AI provider.", "muted", True))
        return w
