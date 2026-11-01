"""Paths and settings. Values come from .env in the data folder.

Program files and your data live apart, so the installed app can stay on one drive while
videos and models go on another. The chosen data folder is remembered in a small file.
"""
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

FROZEN = getattr(sys, "frozen", False)
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))  # program files
VERSION = "1.0.0"
# The Microsoft Store edition never downloads from YouTube (see store/README.md)
STORE = (ROOT / "store_edition.txt").exists() or os.getenv("STORE_EDITION") == "1"
SUPPORT_EMAIL = "muzzamilkhaliddev@gmail.com"   # shown to users for reports; must be filled in before a Store build
PRIVACY_URL = "https://docs.google.com/document/d/1MKTsx4XfDaxB8rEctsaN_J2eY_NpAy5X/preview"     # the public page made from store/privacy-policy.html
# run.bat and the installed app keep separate settings, so one never opens the other's folder
_POINTER = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "2D Animator" / ("data_folder.txt" if FROZEN else "data_folder_dev.txt")


def _data_dir():
    try:
        saved = _POINTER.read_text(encoding="utf-8").strip()
        if saved:
            return Path(saved)
    except OSError:
        pass
    if not FROZEN:
        return ROOT
    # installed: the emptiest drive that is not the system drive, changeable in first-run setup
    system = os.getenv("SystemDrive", "C:").upper()
    drives = [Path(f"{d}:\\") for d in "DEFGHIJ" if Path(f"{d}:\\").exists() and f"{d}:" != system]
    drives = [d for d in drives if _fixed(d)]
    best = max(drives, key=lambda d: shutil.disk_usage(d).free, default=None)
    path = best / "ToonDirector" if best else Path.home() / "Documents" / "ToonDirector"
    _POINTER.parent.mkdir(parents=True, exist_ok=True)
    _POINTER.write_text(str(path), encoding="utf-8")
    return path


def _fixed(drive):
    try:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(str(drive)) == 3 and shutil.disk_usage(drive).free > 5 * 2 ** 30
    except OSError:
        return False


DATA_DIR = _data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(DATA_DIR / ".env")

JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
TOOLS_DIR = DATA_DIR / "tools"
JOBS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TOOLS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))  # never fill the system drive

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
# model that can see images (draws characters/backgrounds from video frames)
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-flash").strip()
# how many times the AI compares its drawing with the video and fixes it (0-3)
REVIEW_PASSES = os.getenv("REVIEW_PASSES", "1").strip()
# frames per second the AI watches: 10, 15, 30, or "all" (every frame of the video)
WATCH_FPS = os.getenv("WATCH_FPS", "10").strip()

# Whisper model size: tiny / base / small. "base" is a good speed/quality balance.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
USE_GPU = os.getenv("USE_GPU", "auto").strip()          # auto | off
NOTIFY_DESKTOP = os.getenv("NOTIFY_DESKTOP", "1").strip()
ONBOARDED = os.getenv("ONBOARDED", "").strip()

FACE_MODEL = MODELS_DIR / "face_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
# face fingerprints, so the same person is one character across cuts
FACE_ID_MODEL = MODELS_DIR / "face_recognition_sface.onnx"
# face finder that also catches small / far-away faces (group shots, panels)
FACE_DET_MODEL = MODELS_DIR / "face_detection_yunet.onnx"


def _tool(name):
    for p in TOOLS_DIR.glob(f"ffmpeg*/**/{name}.exe"):
        os.environ["PATH"] = str(p.parent) + os.pathsep + os.environ.get("PATH", "")
        return str(p)
    return shutil.which(name) or name


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")


def rhubarb_exe():
    """Path to rhubarb.exe if it was downloaded, else None."""
    for p in TOOLS_DIR.rglob("rhubarb.exe"):
        return p
    found = shutil.which("rhubarb")
    return Path(found) if found else None


SETTING_KEYS = ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_VISION_MODEL",
                "REVIEW_PASSES", "WATCH_FPS", "WHISPER_MODEL", "TAVILY_API_KEY", "USE_GPU",
                "NOTIFY_DESKTOP", "ONBOARDED"]


def save_settings(values):
    """Updates the running settings and writes them to .env (other lines are kept)."""
    g = globals()
    env = DATA_DIR / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    for key in SETTING_KEYS:
        if key not in values:
            continue
        val = "".join(str(values[key]).split())  # keys/URLs never contain spaces or newlines
        g[key] = val
        os.environ[key] = val
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key:
                lines[i] = f"{key}={val}"
                break
        else:
            lines.append(f"{key}={val}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")


def move_data_dir(path):
    """Chooses where videos and models live. Takes effect at the next start."""
    _POINTER.parent.mkdir(parents=True, exist_ok=True)
    _POINTER.write_text(str(path), encoding="utf-8")
