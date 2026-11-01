"""The queue of videos being made. The window calls these functions directly."""
import collections
import ctypes
import gc
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config, notify, transcribe
from .config import JOBS_DIR

# (engine stage, name shown in the app, share of the whole job in percent)
STORY_STAGES = [("prepare", "Script", 8), ("design", "Places", 12), ("audio", "Voice", 10), ("direct", "Acting", 8),
                ("lipsync", "Sound", 2), ("render", "Drawing", 60)]
CONVERT_STAGES = [("prepare", "Video", 5), ("audio", "Audio", 1), ("transcribe", "Speech", 10), ("track", "Tracking", 35),
                  ("lipsync", "Lip-sync", 4), ("design", "Drawing", 15), ("direct", "Acting", 3), ("render", "Export", 27)]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
listeners = []  # callbacks(job id), called from the worker thread on any change


FRAME = re.compile(r"frame (\d+)/(\d+)")


class Job:
    def __init__(self, id, options, dir):
        self.id, self.options, self.dir = id, options, str(dir)
        self.status, self.stage, self.progress, self.message = "queued", "", 0.0, "Waiting"
        self.logs, self.error, self.style = [], None, None
        self.script_title = None
        self.cancelled = False
        self.created = time.time()
        self.started = None
        self.samples = collections.deque(maxlen=40)  # (time, frame) while drawing

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        m = FRAME.search(self.message or "")
        if m:
            self.samples.append((time.time(), int(m.group(1))))
        self.save()

    def log(self, msg):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if self.status == "running" and not msg.startswith("Set review"):
            self.message = msg[:90]
        self.save()

    def live(self):
        """What is happening right now, measured rather than guessed."""
        m = FRAME.search(self.message or "")
        out = {"frame": None, "total": None, "fps": None, "left": None,
               "elapsed": time.time() - self.started if self.started and self.active else None}
        if m:
            out["frame"], out["total"] = int(m.group(1)), int(m.group(2))
            if len(self.samples) >= 2:
                (t0, f0), (t1, f1) = self.samples[0], self.samples[-1]
                if t1 > t0 and f1 > f0:
                    out["fps"] = (f1 - f0) / (t1 - t0)
                    out["left"] = (out["total"] - out["frame"]) / out["fps"]
        return out

    @property
    def preview(self):
        p = Path(self.dir) / "preview.jpg"
        if p.exists():
            return p
        places = sorted((Path(self.dir) / "design").glob("setpreview_*.png"), key=lambda f: f.stat().st_mtime)
        return places[-1] if places else None

    @property
    def title(self):
        name = self.script_title or self.options.get("name") or self.id
        return name[7:] if name.startswith("Story: ") else name

    @property
    def is_story(self):
        return self.options.get("mode") == "story"

    @property
    def active(self):
        return self.status in ("queued", "running")

    @property
    def stages(self):
        return STORY_STAGES if self.is_story else CONVERT_STAGES

    def stage_position(self):
        """(index of the current stage, how much of it is done 0..1)."""
        names = [x[0] for x in self.stages]
        if self.status == "done":
            return len(names), 0.0
        if self.stage not in names:
            return 0, 0.0
        i = names.index(self.stage)
        base = sum(x[2] for x in self.stages[:i])
        return i, min(max((self.progress - base) / self.stages[i][2], 0.0), 1.0)

    @property
    def output(self):
        return Path(self.dir) / "output.mp4"

    def to_dict(self):
        out = {k: getattr(self, k) for k in ("id", "status", "stage", "progress", "message", "logs", "error",
                                              "style", "options", "created", "script_title")}
        out["has_output"] = self.output.exists()
        return out

    def save(self):
        (Path(self.dir) / "job.json").write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        for cb in listeners:
            cb(self.id)


JOBS: dict[str, Job] = {}
QUEUE: "queue.Queue[Job]" = queue.Queue()


def load():
    for f in JOBS_DIR.glob("*/job.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            j = Job(d["id"], d["options"], f.parent)
            for k in ("stage", "progress", "message", "logs", "error", "style", "created", "script_title"):
                setattr(j, k, d.get(k))
            j.status = d["status"] if d["status"] in ("done", "error") else "error"
            if j.status == "error" and not j.error:
                j.error = "The app closed while this was running. Press Resume to continue."
            JOBS[j.id] = j
        except Exception:
            pass


def _stay_awake(on):
    """Windows must not sleep in the middle of a render (ends by itself when the job ends)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def _worker():
    from .pipeline import run_job
    while True:
        job = QUEUE.get()
        if job.cancelled:
            continue
        job.started = time.time()
        job.samples.clear()
        job.update(status="running", error=None)
        _stay_awake(True)
        try:
            run_job(job)
            job.update(status="done")
            notify.add("done", "Your video is ready", job.title, job.id)
        except Exception as e:
            job.log(traceback.format_exc()[-1500:])
            stopped = str(e) == "Cancelled"
            job.update(status="error", error="Stopped. Press Resume to continue." if stopped else str(e)[:500])
            if not stopped:
                notify.add("error", "A video stopped", f"{job.title}: {str(e)[:160]}", job.id)
        finally:
            _stay_awake(False)
            transcribe.release()
            gc.collect()  # long queues on small PCs must not run out of memory


def start():
    load()
    threading.Thread(target=_worker, daemon=True).start()


def _new_dir():
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    return jid, d


def _queue(jid, opts, d):
    job = Job(jid, opts, d)
    JOBS[jid] = job
    job.save()
    QUEUE.put(job)
    return job


def new_story(source, topic="", script_text="", youtube_url="", language="en", voice="male", minutes=2.0,
              aspect="16:9", height=1080, description="", music_level=0.5, captions=True, voice_file=None,
              transcript=""):
    """A brand-new original video from a topic, your script, or a YouTube link (research only)."""
    jid, d = _new_dir()
    voice_ref = None
    if voice_file:
        voice_ref = "my_voice" + (Path(voice_file).suffix.lower() or ".wav")
        shutil.copyfile(voice_file, d / voice_ref)
    name = {"topic": topic.strip()[:80], "script": script_text.strip()[:60], "youtube": "From YouTube: " + youtube_url.strip()}[source]
    return _queue(jid, {"mode": "story", "source": source, "topic": topic.strip(), "script_text": script_text.strip(),
                        "youtube_url": youtube_url.strip(), "language": language, "voice": voice,
                        "minutes": min(max(float(minutes), 0.5), 60), "aspect": aspect, "height": height,
                        "description": description.strip(), "music_level": music_level, "captions": captions,
                        "voice_ref": voice_ref, "name": name, "transcript": transcript.strip()}, d)


def new_convert(path="", youtube_url="", description="", height=720, quality="full", subtitles=False, start_at=0, end_at=0,
                art="ai", max_people=8, motion="director"):
    """Turns an existing video into a 2D cartoon."""
    jid, d = _new_dir()
    if path:
        shutil.copyfile(path, d / f"source{Path(path).suffix.lower() or '.mp4'}")
    return _queue(jid, {"youtube_url": youtube_url.strip(), "description": description.strip(), "height": height,
                        "quality": quality, "subtitles": subtitles, "start": start_at or None, "end": end_at or None,
                        "art": art, "max_people": max_people, "motion": motion,
                        "name": Path(path).name if path else youtube_url.strip()}, d)


def resume(jid):
    job = JOBS[jid]
    if not job.active:
        job.cancelled = False
        job.update(status="queued", error=None, message="Waiting")
        QUEUE.put(job)


def extend(jid, minutes):
    job = JOBS[jid]
    if job.is_story and not job.active:
        job.options["extend_to"] = min(max(float(minutes), 1.0), 60.0)
        resume(jid)


def restyle(jid, description=""):
    """Draws the characters and places again. Everything else is reused."""
    job = JOBS[jid]
    if job.active:
        return
    ddir = Path(job.dir) / "design"
    if ddir.exists():
        for f in ddir.iterdir():
            if f.name.startswith(("rig_", "style_", "bg_", "character_", "review_", "set_", "setpreview_")):
                f.unlink(missing_ok=True)
    if job.is_story:
        (Path(job.dir) / "scene_anim.json").unlink(missing_ok=True)
    job.options["description"] = description
    resume(jid)


def cancel(jid):
    job = JOBS[jid]
    job.cancelled = True
    if job.status == "queued":
        job.update(status="error", error="Stopped. Press Resume to continue.")
    else:
        job.update(message="Stopping at the next safe point")


def delete(jid):
    job = JOBS.pop(jid)
    shutil.rmtree(job.dir, ignore_errors=True)


def thumbnail(job):
    """A still from the finished video, made once."""
    t = Path(job.dir) / "thumb.jpg"
    if not t.exists() and job.output.exists():
        subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-ss", "4", "-i", str(job.output), "-frames:v", "1",
                        "-vf", "scale=640:-2", str(t)], creationflags=NO_WINDOW)
    return t if t.exists() else None


def active_count():
    return sum(1 for j in JOBS.values() if j.active)
