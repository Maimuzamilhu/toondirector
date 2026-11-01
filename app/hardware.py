"""What this PC has, and which fast video encoder actually works on it.

An encoder is only used after a tiny test clip succeeds, because a driver can be listed and
still fail. Anything that fails falls back to the normal CPU encoder.
"""
import ctypes
import os
import re
import shutil
import subprocess

from . import config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_cache = {}


def _run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None


def ram_gb():
    class Mem(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("v", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]
    m = Mem()
    m.len = ctypes.sizeof(Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return round(m.total / 2 ** 30)


def nvidia():
    """(name, memory GB) of an NVIDIA card, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    p = _run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not p or p.returncode or not p.stdout.strip():
        return None
    name, mem = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")[:2]]
    return name, round(float(mem) / 1024)


def graphics_names():
    p = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
    return [x.strip() for x in p.stdout.splitlines() if x.strip()] if p else []


ENCODERS = {
    "h264_nvenc": ("NVIDIA", ["-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p"]),
    "h264_amf": ("AMD", ["-rc", "cqp", "-qp_i", "20", "-qp_p", "22", "-pix_fmt", "yuv420p"]),
    "h264_qsv": ("Intel", ["-global_quality", "22", "-preset", "medium", "-pix_fmt", "nv12"]),
}
CPU_ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


def _works(enc):
    extra = ENCODERS[enc][1]
    p = _run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=0.2",
              "-c:v", enc, *extra, "-f", "null", "-"], 20)
    return bool(p and p.returncode == 0)


def best_encoder():
    """The name of a hardware encoder that really works here, or None."""
    if "enc" not in _cache:
        found = None
        p = _run([config.FFMPEG, "-hide_banner", "-encoders"])
        listed = p.stdout if p else ""
        for enc in ENCODERS:
            if re.search(rf"\b{enc}\b", listed) and _works(enc):
                found = enc
                break
        _cache["enc"] = found
    return _cache["enc"]


def encoder_args():
    if config.USE_GPU == "off":
        return CPU_ENCODER
    enc = best_encoder()
    return ["-c:v", enc, *ENCODERS[enc][1]] if enc else CPU_ENCODER


def summary():
    """Plain facts for the Settings page and the first-run advice."""
    nv = nvidia()
    enc = best_encoder()
    return {"cores": os.cpu_count() or 1, "ram": ram_gb(), "nvidia": nv, "names": graphics_names(),
            "encoder": ENCODERS[enc][0] if enc else None,
            "free_gb": round(shutil.disk_usage(config.DATA_DIR).free / 2 ** 30)}
