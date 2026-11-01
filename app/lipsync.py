"""Audio-driven mouth shapes, used when the face isn't visible (or no video face at all)."""
import json
import subprocess
import wave

import numpy as np

from .media import NO_WINDOW

# Rhubarb mouth shapes -> (open, width, round). width: -1 narrow .. 1 wide.
VISEMES = {
    "A": (0.00, 0.00, 0.0),   # M B P (closed)
    "B": (0.28, 0.30, 0.0),   # K S T EE (slightly open, teeth)
    "C": (0.55, 0.15, 0.0),   # EH AE
    "D": (0.85, 0.10, 0.0),   # AA (wide open)
    "E": (0.50, -0.35, 0.6),  # AO ER (rounded)
    "F": (0.28, -0.55, 1.0),  # UW OW W (pucker)
    "G": (0.18, 0.05, 0.0),   # F V
    "H": (0.45, 0.00, 0.0),   # L
    "X": (0.00, 0.00, 0.0),   # rest
}


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def frame_energy(wav_path, fps, n):
    """Loudness per video frame, 0..1 (normalized to the 95th percentile)."""
    data, sr = read_wav(wav_path)
    hop = sr / fps
    out = np.zeros(n, np.float32)
    for i in range(n):
        a, b = int(i * hop), int((i + 1) * hop)
        seg = data[a:b]
        if seg.size:
            out[i] = np.sqrt(np.mean(seg * seg))
    ref = np.percentile(out[out > 0], 95) if np.any(out > 0) else 1.0
    out = np.clip(out / (ref or 1.0), 0, 1)
    out[out < 0.08] = 0  # silence
    return out


def run_rhubarb(exe, wav_path, out_json, dialog_text=None):
    cmd = [str(exe), "-f", "json", "-r", "phonetic", "-o", str(out_json)]
    if dialog_text:
        dpath = out_json.with_suffix(".txt")
        dpath.write_text(dialog_text, encoding="utf-8")
        cmd += ["--dialogFile", str(dpath)]
    cmd.append(str(wav_path))
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError("Rhubarb failed: " + p.stderr[-800:])
    return json.loads(out_json.read_text(encoding="utf-8"))["mouthCues"]


def cues_to_frames(cues, fps, n):
    """Rhubarb cue list -> arrays (open, width, round) per frame."""
    out = np.zeros((n, 3), np.float32)
    j = 0
    for i in range(n):
        t = i / fps
        while j + 1 < len(cues) and cues[j]["end"] <= t:
            j += 1
        if cues and cues[j]["start"] <= t < cues[j]["end"]:
            out[i] = VISEMES.get(cues[j]["value"], VISEMES["X"])
    return out


def energy_to_frames(energy):
    """Fallback without Rhubarb: mouth opens with loudness, shape varies a little."""
    n = len(energy)
    out = np.zeros((n, 3), np.float32)
    t = np.arange(n)
    out[:, 0] = np.clip(energy * 0.9, 0, 0.8)
    out[:, 1] = 0.3 * np.sin(t * 0.9) * (energy > 0)
    out[:, 2] = np.clip(0.5 * np.sin(t * 0.37), 0, 1) * (energy > 0)
    return out


def speech_mask(segments, fps, n, pad=0.06):
    """True on frames where words are actually spoken (Whisper's speech detection skips
    music, laughter and noise). Falls back to whole segments when words are missing."""
    m = np.zeros(n, bool)
    for seg in segments or []:
        words = seg.get("words") or []
        spans = [(w["s"], w["e"]) for w in words] or [(seg["start"], seg["end"])]
        for a, b in spans:
            i0, i1 = int(max(a - pad, 0) * fps), int((b + pad) * fps) + 1
            m[max(i0, 0):min(i1, n)] = True
    return m
