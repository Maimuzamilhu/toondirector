"""Optional downloads. Nothing large is fetched until you ask for it.

Downloads run in the background, survive a restart (a stopped file continues from where it
was), and never touch your existing work.
"""
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

from . import config, hardware

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HF = "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter"
OPENCV = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MP = "https://storage.googleapis.com/mediapipe-models"
VOICE_ENV = config.DATA_DIR / ".venv-voice"
VOICE_PY = VOICE_ENV / "Scripts" / "python.exe"
VOICE_CKPT = config.MODELS_DIR / "openvoice_v2"

state = {}   # id -> {"status": idle|running|done|error, "progress": 0..1, "message": ""}
on_change = []
_stop = {}


def _whisper_ok(name):
    return any((config.MODELS_DIR / "hf_cache" / "hub").glob(f"models--*whisper-{name}*/snapshots/*/model.bin"))


def _ffmpeg_ok():
    return shutil.which("ffmpeg") is not None or Path(config.FFMPEG).is_file()


ITEMS = {
    "ffmpeg": dict(name="Video tools", size="110 MB", required=True,
                   about="Needed to make any video. Includes support for fast GPU video export.", ok=_ffmpeg_ok),
    "whisper": dict(name="Speech recognition", size="150 MB",
                    about="Reads what people say in a video. Needed to turn a video into a cartoon.",
                    ok=lambda: _whisper_ok(config.WHISPER_MODEL or "base")),
    "tracking": dict(name="Face and body tracking", size="20 MB",
                     about="Follows people, faces and lips in a video you turn into a cartoon.",
                     ok=lambda: all(p.exists() for p in (config.FACE_MODEL, config.POSE_MODEL, config.FACE_DET_MODEL, config.FACE_ID_MODEL))),
    "lipsync": dict(name="Lip-sync", size="20 MB", about="Matches mouths to speech when you turn a video into a cartoon.",
                    ok=lambda: bool(config.rhubarb_exe())),
    "voice": dict(name="Use my voice", size="1.3 GB",
                  about="Turns narration into your own voice from a short recording. Works on this PC, nothing is uploaded.",
                  ok=lambda: VOICE_PY.exists() and (VOICE_CKPT / "converter" / "checkpoint.pth").exists()),
}


def status(aid):
    s = state.get(aid)
    if s and s["status"] == "running":
        return s
    if ITEMS[aid]["ok"]():
        return {"status": "done", "progress": 1, "message": "Installed"}
    return s or {"status": "idle", "progress": 0, "message": ""}


def _set(aid, status_, progress, message=""):
    state[aid] = {"status": status_, "progress": progress, "message": message}
    for cb in on_change:
        cb(aid)


def _fetch(aid, url, dst, base=0.0, span=1.0):
    """Downloads with resume: a partial .part file continues where it stopped."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "ToonDirector/1.0", **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # the server cannot continue: start this file again
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(1 << 18):
                if _stop.get(aid):
                    raise InterruptedError
                f.write(chunk)
                have += len(chunk)
                if total:
                    _set(aid, "running", base + span * have / total,
                         f"{have / 2 ** 20:.0f} of {total / 2 ** 20:.0f} MB")
    part.replace(dst)


def _ffmpeg(aid):
    z = config.TOOLS_DIR / "ffmpeg.zip"
    _fetch(aid, "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", z, 0, 0.95)
    _set(aid, "running", 0.96, "Unpacking")
    with zipfile.ZipFile(z) as f:
        f.extractall(config.TOOLS_DIR)
    z.unlink()
    config.FFMPEG = config._tool("ffmpeg")
    config.FFPROBE = config._tool("ffprobe")


def _tracking(aid):
    files = [(config.FACE_MODEL, f"{MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
             (config.POSE_MODEL, f"{MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
             (config.FACE_DET_MODEL, f"{OPENCV}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
             (config.FACE_ID_MODEL, f"{OPENCV}/face_recognition_sface/face_recognition_sface_2021dec.onnx")]
    for i, (path, url) in enumerate(files):
        if not path.exists():
            _fetch(aid, url, path, i / len(files), 1 / len(files))


def _lipsync(aid):
    z = config.TOOLS_DIR / "rhubarb.zip"
    _fetch(aid, "https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Windows.zip", z, 0, 0.95)
    with zipfile.ZipFile(z) as f:
        f.extractall(config.TOOLS_DIR)
    z.unlink()


def _whisper(aid):
    from faster_whisper import download_model
    _set(aid, "running", 0.2, "Downloading the speech model")
    download_model(config.WHISPER_MODEL or "base")


def _uv():
    here = Path(sys.executable).parent
    for p in (here / "uv.exe", here / "bin" / "uv.exe", config.ROOT / "bin" / "uv.exe"):
        if p.exists():
            return str(p)
    return shutil.which("uv") or "uv"


def _sh(aid, args, progress, message):
    _set(aid, "running", progress, message)
    p = subprocess.run(args, capture_output=True, text=True, creationflags=NO_WINDOW)
    if p.returncode:
        raise RuntimeError(p.stderr[-400:] or "Install step failed")


def _voice(aid):
    py, uv = str(VOICE_PY), _uv()
    if not VOICE_PY.exists():
        _sh(aid, [uv, "venv", "--python", "3.12", str(VOICE_ENV)], 0.03, "Preparing a private Python")
    index = "https://download.pytorch.org/whl/" + ("cu126" if hardware.nvidia() and config.USE_GPU != "off" else "cpu")
    _sh(aid, [uv, "pip", "install", "--python", py, "torch", "torchaudio", "--index-url", index], 0.08,
        "Installing the voice engine (the largest step)")
    _sh(aid, [uv, "pip", "install", "--python", py, "--no-deps", "git+https://github.com/myshell-ai/OpenVoice.git"], 0.6,
        "Installing OpenVoice")
    _sh(aid, [uv, "pip", "install", "--python", py, "librosa", "soundfile", "unidecode", "inflect", "eng_to_ipa", "pypinyin",
              "jieba", "cn2an", "langid", "huggingface_hub"], 0.7, "Installing voice helpers")
    (VOICE_CKPT / "converter").mkdir(parents=True, exist_ok=True)
    _fetch(aid, f"{HF}/config.json", VOICE_CKPT / "converter" / "config.json", 0.8, 0.02)
    _fetch(aid, f"{HF}/checkpoint.pth", VOICE_CKPT / "converter" / "checkpoint.pth", 0.82, 0.18)


_INSTALL = {"ffmpeg": _ffmpeg, "tracking": _tracking, "lipsync": _lipsync, "whisper": _whisper, "voice": _voice}


def install(aid):
    if state.get(aid, {}).get("status") == "running":
        return

    def work():
        from . import notify
        _stop[aid] = False
        _set(aid, "running", 0, "Starting")
        try:
            _INSTALL[aid](aid)
            _set(aid, "done", 1, "Installed")
            notify.add("done", f"{ITEMS[aid]['name']} is ready", "The download finished.")
        except InterruptedError:
            _set(aid, "idle", 0, "Paused. Download again to continue.")
        except Exception as e:
            _set(aid, "error", 0, str(e)[:240])
            notify.add("error", f"{ITEMS[aid]['name']} did not finish",
                       "Check your internet connection, then press Try again. It continues where it stopped.")

    threading.Thread(target=work, daemon=True).start()


def cancel(aid):
    _stop[aid] = True


def remove(aid):
    """Frees the disk space of an add-on. Your videos are not touched."""
    if aid == "voice":
        shutil.rmtree(VOICE_ENV, ignore_errors=True)
        shutil.rmtree(VOICE_CKPT, ignore_errors=True)
    state.pop(aid, None)
    for cb in on_change:
        cb(aid)


def snapshot():
    return json.loads(json.dumps({k: status(k) for k in ITEMS}))
