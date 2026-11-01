"""ffmpeg / ffprobe / yt-dlp helpers."""
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import hardware
from .config import FFMPEG, FFPROBE

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{p.stderr[-2000:]}")
    return p.stdout


def probe(path):
    out = run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise RuntimeError("No video stream found in the file.")
    fps = float(Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"))
    if not fps or fps != fps:
        fps = 30.0
    return {
        "fps": fps,
        "duration": float(info["format"].get("duration", 0) or 0),
        "has_audio": audio is not None,
    }


def normalize(src, dst, fps, start=None, end=None, max_height=720):
    """Constant frame rate, rotation applied, height <= max_height. Video only."""
    cmd = [FFMPEG, "-y"]
    if start:
        cmd += ["-ss", str(start)]
    if end:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(src), "-an",
            "-vf", f"scale=-2:'min({max_height},ih)'",
            "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(dst)]
    run(cmd)


def extract_audio(src, dst, start=None, end=None, has_audio=True, duration=None):
    """16 kHz mono 16-bit wav (what Whisper and Rhubarb want). Silent track if no audio."""
    cmd = [FFMPEG, "-y"]
    if not has_audio:
        dur = (end or duration or 1) - (start or 0)
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(max(dur, 0.1))]
    else:
        if start:
            cmd += ["-ss", str(start)]
        if end:
            cmd += ["-to", str(end)]
        cmd += ["-i", str(src), "-vn"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    run(cmd)


def download_youtube(url, out_dir, progress=None):
    """Downloads a video (<=720p) with yt-dlp. Returns the file path."""
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total, "Downloading from YouTube")

    opts = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/bv*[height<=720]+ba/b",
        "outtmpl": str(Path(out_dir) / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": str(Path(FFMPEG).parent) if Path(FFMPEG).exists() else None,
        "progress_hooks": [hook],
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():
        path = next(Path(out_dir).glob("source.*"))
    return path, info.get("title", "")


class PreviewPipe:
    """Stands in front of the encoder and saves a small copy of a frame now and then, so the
    app can show the video as it is being drawn. A failed preview never stops a render."""
    EVERY = 45

    def __init__(self, pipe, path, w, h):
        self.pipe, self.path, self.w, self.h = pipe, Path(path), w, h
        self.count = 0

    def write(self, frame):
        if self.count % self.EVERY == 0:
            self._save(frame)
        self.count += 1
        return self.pipe.write(frame)

    def _save(self, frame):
        try:
            img = np.frombuffer(frame, np.uint8).reshape(self.h, self.w, 4)[:, :, :3]
            small = cv2.resize(img, (640, 640 * self.h // self.w), interpolation=cv2.INTER_AREA)
            tmp = self.path.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            os.replace(tmp, self.path)
        except (OSError, ValueError, cv2.error):
            pass

    def close(self):
        self.pipe.close()


def open_encoder(dst, w, h, fps, audio, n_frames):
    """ffmpeg process that takes raw BGRx frames on stdin and muxes the original voice."""
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
           "-i", str(audio),
           *hardware.encoder_args(),
           "-c:a", "aac", "-b:a", "160k", "-t", f"{n_frames / fps:.4f}", "-movflags", "+faststart", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    p.stdin = PreviewPipe(p.stdin, Path(dst).parent / "preview.jpg", w, h)
    return p
