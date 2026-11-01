# Builds the app folder:  .venv\Scripts\pyinstaller packaging\2d_animator.spec --noconfirm
# One folder rather than one .exe: it starts faster, and the installer ships it whole.
import os
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
uv = shutil.which("uv") or str(ROOT / ".venv" / "Scripts" / "uv.exe")
STORE = os.getenv("STORE_EDITION") == "1"  # the Store edition leaves out the YouTube downloader

datas = [(str(ROOT / "app" / "voice_clone.py"), "app"), (str(ROOT / "studio" / "assets"), "studio/assets")]
binaries = [(uv, "bin")]
hidden = ["edge_tts", "resvg_py", "faster_whisper", "ctranslate2", "mediapipe", "cairo"]
if STORE:
    datas.append((str(ROOT / "packaging" / "store_edition.txt"), "."))
else:
    hidden += ["yt_dlp"] + collect_submodules("yt_dlp")
for pkg in ("mediapipe", "faster_whisper", "ctranslate2", "certifi"):
    datas += collect_data_files(pkg)
for pkg in ("mediapipe", "ctranslate2", "onnxruntime", "cv2", "cairo"):
    binaries += collect_dynamic_libs(pkg)
hidden += collect_submodules("app")

a = Analysis([str(ROOT / "packaging" / "launch.py")], pathex=[str(ROOT)], datas=datas, binaries=binaries,
             hiddenimports=hidden,
             excludes=(["yt_dlp"] if STORE else []) + ["tkinter", "IPython", "pytest", "fastapi", "uvicorn",
                       "PySide6.Qt3DCore", "PySide6.QtWebEngineCore", "PySide6.QtQuick3D", "PySide6.QtCharts", "PySide6.QtPdf"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ToonDirector", console=False,
          icon=str(ROOT / "packaging" / "icon.ico"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ToonDirector")
