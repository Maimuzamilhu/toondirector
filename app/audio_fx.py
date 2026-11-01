"""Music and sound effects made by code (no samples, no copyright issues).

music(mood, seconds)  -> background track: soft chord pads, a plucked arpeggio and light
                         percussion; tempo and chords depend on the mood
sfx(name)             -> whoosh, pop, ding, click, swoosh_up, thud
mix(...)              -> narration + music (ducked under the voice) + effects
"""
import wave

import numpy as np

SR = 44100

MOODS = {
    #            bpm  root(Hz)  chord progression (semitones of each chord's notes from root)
    "curious":   (96, 220.0, [(0, 4, 7, 11), (5, 9, 12, 16), (-3, 0, 4, 7), (7, 11, 14, 17)]),
    "upbeat":    (118, 261.6, [(0, 4, 7, 12), (7, 11, 14, 19), (9, 12, 16, 21), (5, 9, 12, 17)]),
    "calm":      (78, 196.0, [(0, 4, 7, 11), (-3, 0, 4, 7), (5, 9, 12, 16), (2, 5, 9, 12)]),
    "dramatic":  (84, 174.6, [(0, 3, 7, 10), (-4, 0, 3, 7), (-2, 2, 5, 9), (-5, -1, 2, 5)]),
    "happy":     (110, 293.7, [(0, 4, 7, 12), (5, 9, 12, 17), (7, 11, 14, 19), (0, 4, 7, 12)]),
    "mysterious": (72, 164.8, [(0, 3, 7, 10), (1, 5, 8, 12), (-2, 2, 5, 8), (0, 3, 7, 11)]),
}


def _env(n, attack, release):
    e = np.ones(n, np.float32)
    a, r = min(int(attack * SR), n), min(int(release * SR), n)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _tone(freq, dur, kind="sine"):
    t = np.arange(int(dur * SR)) / SR
    if kind == "pad":  # soft detuned saw-ish pad
        x = sum(np.sin(2 * np.pi * freq * m * t + d) / m for m, d in ((1, 0), (2, 0.3), (3, 0.7))) * 0.5
        x += 0.5 * np.sin(2 * np.pi * freq * 1.004 * t)
        return x.astype(np.float32)
    if kind == "pluck":
        x = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
        return (x * np.exp(-t * 6)).astype(np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def music(mood, seconds, seed=1):
    """A looping background track for the whole video."""
    bpm, root, prog = MOODS.get(mood, MOODS["curious"])
    rng = np.random.default_rng(seed)
    n = int(seconds * SR) + SR
    out = np.zeros(n, np.float32)
    beat = 60.0 / bpm
    bar = beat * 4
    t = 0.0
    k = 0
    while t < seconds + bar:
        chord = prog[k % len(prog)]
        i0 = int(t * SR)
        # pad: whole bar
        pad = sum(_tone(root * 2 ** (s / 12) / 2, bar, "pad") for s in chord) / len(chord)
        pad *= _env(len(pad), 0.4, 0.6) * 0.16
        seg = out[i0:i0 + len(pad)]
        seg += pad[:len(seg)]
        # arpeggio: 8th notes
        for j in range(8):
            s = chord[(j * 3 + k) % len(chord)] + (12 if j % 4 == 3 else 0)
            note = _tone(root * 2 ** (s / 12), beat * 0.9, "pluck") * 0.11
            a = i0 + int(j * beat / 2 * SR)
            seg2 = out[a:a + len(note)]
            seg2 += note[:len(seg2)]
        # light percussion: soft kick on 1 and 3, shaker on 8ths
        for j in range(4):
            a = i0 + int(j * beat * SR)
            if j % 2 == 0:
                kt = np.arange(int(0.18 * SR)) / SR
                kick = np.sin(2 * np.pi * (90 * np.exp(-kt * 25) + 40) * kt) * np.exp(-kt * 18) * 0.22
                seg3 = out[a:a + len(kick)]
                seg3 += kick[:len(seg3)].astype(np.float32)
            for h in (0, 0.5):
                b = a + int(h * beat * SR)
                sh = rng.normal(0, 1, int(0.05 * SR)).astype(np.float32)
                sh = np.diff(sh, prepend=0) * np.exp(-np.arange(len(sh)) / SR * 60) * 0.035
                seg4 = out[b:b + len(sh)]
                seg4 += sh[:len(seg4)]
        t += bar
        k += 1
    out = out[:int(seconds * SR)]
    fade = min(int(2 * SR), len(out) // 4)
    if fade:
        out[:fade] *= np.linspace(0, 1, fade)
        out[-fade:] *= np.linspace(1, 0, fade)
    return out


def sfx(name):
    """Short sound effects."""
    rng = np.random.default_rng(len(name))
    if name in ("whoosh", "swoosh_up"):
        d = 0.45
        x = rng.normal(0, 1, int(d * SR)).astype(np.float32)
        # moving band-pass: blend of smoothed noise at changing widths
        t = np.linspace(0, 1, len(x))
        lo = np.convolve(x, np.ones(40) / 40, mode="same")
        hi = x - np.convolve(x, np.ones(6) / 6, mode="same")
        mixw = t if name == "swoosh_up" else 1 - t
        y = lo * (1 - mixw) + hi * mixw * 0.6
        return (y * np.sin(np.pi * t) ** 1.5 * 0.5).astype(np.float32)
    if name == "pop":
        t = np.arange(int(0.12 * SR)) / SR
        return (np.sin(2 * np.pi * (600 + 900 * np.exp(-t * 40)) * t) * np.exp(-t * 35) * 0.45).astype(np.float32)
    if name == "ding":
        t = np.arange(int(0.9 * SR)) / SR
        x = np.sin(2 * np.pi * 1318 * t) + 0.5 * np.sin(2 * np.pi * 2637 * t) + 0.2 * np.sin(2 * np.pi * 3951 * t)
        return (x * np.exp(-t * 5) * 0.22).astype(np.float32)
    if name == "click":
        t = np.arange(int(0.03 * SR)) / SR
        return (rng.normal(0, 1, len(t)) * np.exp(-t * 200) * 0.3).astype(np.float32)
    if name == "thud":
        t = np.arange(int(0.3 * SR)) / SR
        return (np.sin(2 * np.pi * (60 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 12) * 0.6).astype(np.float32)
    return np.zeros(1, np.float32)


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:  # simple resample
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((x * 32767).astype(np.int16).tobytes())


def mix(voice, music_track, effects, music_level=0.55, duck=0.25):
    """voice: float array; music_track: float array (same length or longer);
    effects: list of (time_seconds, array). Music is turned down while the voice talks."""
    n = len(voice)
    out = np.zeros(n, np.float32)
    m = music_track[:n] if len(music_track) >= n else np.pad(music_track, (0, n - len(music_track)))
    # ducking: smoothed voice envelope pushes the music down
    env = np.abs(voice).astype(np.float64)
    k = int(0.25 * SR)
    c = np.cumsum(np.concatenate([[0.0], env]))  # fast moving average (running sums)
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k // 2, 0, n)
    env = ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)
    env = np.clip(env / (np.percentile(env[env > 0], 90) if np.any(env > 0) else 1), 0, 1)
    gain = music_level * (1 - (1 - duck) * env)
    out += m * gain + voice
    for t, fx in effects:
        a = int(t * SR)
        if 0 <= a < n:
            seg = out[a:a + len(fx)]
            seg += fx[:len(seg)]
    peak = np.max(np.abs(out)) or 1
    return out / max(peak, 1.0) * 0.95
