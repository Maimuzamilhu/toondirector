"""Turns tracking + audio + emotions into a smooth per-frame animation state.

Two sources are blended per frame:
  * video mode  - the person's face/body drive the cartoon (exact copy of each frame)
  * voice mode  - when no face is visible, the voice drives mouth/head and idle motion
"""
import numpy as np

from .tracking import IDX

EMOTION_FX = {
    #            smile brow_up tilt  down  blush eye_smile wide gaze_x gaze_y bounce
    "neutral":  (0.0,  0.0,    0.0,  0.0,  0.0,  0.0,     0.0,  0.0,   0.0,   0.0),
    "happy":    (0.45, 0.1,    0.0,  0.0,  0.5,  0.45,    0.0,  0.0,   0.0,   0.3),
    "sad":      (-0.35, 0.0,   0.7,  0.0,  0.0,  0.0,     0.0,  0.0,   0.3,   0.0),
    "angry":    (-0.2, 0.0,   -0.6,  0.8,  0.2,  0.0,     0.0,  0.0,   0.0,   0.2),
    "surprised": (0.0, 0.8,    0.2,  0.0,  0.0,  0.0,     0.5,  0.0,   0.0,   0.0),
    "thinking": (0.0,  0.2,    0.3,  0.0,  0.0,  0.0,     0.0,  0.4,  -0.6,   0.0),
    "excited":  (0.5,  0.35,   0.0,  0.0,  0.4,  0.3,     0.2,  0.0,   0.0,   1.0),
}
FX_KEYS = ["smile", "brow_up", "brow_tilt", "brow_down", "blush", "eye_smile",
           "eye_wide", "gaze_x", "gaze_y", "bounce"]


def smooth(x, a):
    """Zero-lag exponential smoothing (forward + backward). a=1 means no smoothing."""
    y = np.array(x, dtype=np.float64)
    for i in range(1, len(y)):
        y[i] = a * y[i] + (1 - a) * y[i - 1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = a * y[i] + (1 - a) * y[i + 1]
    return y


def fill(x, mask=None):
    """Linear interpolation over NaN (and over frames where mask is False)."""
    x = np.array(x, dtype=np.float64)
    if mask is not None:
        x[~mask] = np.nan
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _blinks(n, fps, seed=7):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    shape = np.array([0.5, 1.0, 1.0, 0.6, 0.2])
    t = rng.uniform(1, 3) * fps
    while t < n:
        i = int(t)
        k = shape[: max(min(n - i, len(shape)), 0)]
        out[i:i + len(k)] = np.maximum(out[i:i + len(k)], k)
        t += rng.uniform(2.0, 5.0) * fps
    return out


def build(F, n, fps, W, H, energy, audio_mouth, segments, emotions, solo=True, cuts=None):
    """F: tracking features of ONE person [frames, features] (None = no person, voice only).
    solo=False when several people share the video: a person who is not visible keeps
    still at their last place (or fades out), and the voice does not move their mouth."""
    t = np.arange(n) / fps
    # smoothing never crosses a cut: at a cut everything jumps, like the video
    bounds = sorted({0, n, *[c for c in (cuts or []) if 0 < c < n]})

    def sm(x, a):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        for a0, b0 in zip(bounds, bounds[1:]):
            out[a0:b0] = smooth(x[a0:b0], a) if b0 - a0 > 1 else x[a0:b0]
        return out
    zeros = np.zeros(n)

    def col(name):
        if F is None or len(F) == 0:
            return None
        c = F[:n, IDX[name]]
        if len(c) < n:  # tracking may have a few frames less than the audio
            c = np.concatenate([c, np.full(n - len(c), np.nan)])
        return c

    # ---------- weights: how much the video drives each frame ----------
    face_ok = col("face_ok")
    if face_ok is None:
        face_w = zeros.copy()
    else:
        filled = fill(face_ok)
        filled = np.zeros(n) if filled is None else filled
        face_w = sm(filled, 0.15)
        face_w = np.clip((face_w - 0.2) / 0.6, 0, 1)
    face_mask = None if face_ok is None else (fill(face_ok) if fill(face_ok) is not None else np.zeros(n)) > 0.5

    def fcol(name, default=0.0):
        c = col(name)
        if c is None:
            return np.full(n, default)
        f = fill(c, face_mask)
        return np.full(n, default) if f is None else f

    # ---------- voice-mode (fallback) motion ----------
    e_slow = sm(energy, 0.08)
    r_v = np.full(n, H * 0.17)
    hx_v = W / 2 + np.sin(0.6 * t) * 0.05 * r_v
    hy_v = H * 0.42 - e_slow * 0.05 * r_v
    roll_v = 0.05 * np.sin(0.45 * t) + 0.05 * np.sin(2.3 * t) * e_slow
    yaw_v = 0.25 * np.sin(0.27 * t) + 0.1 * np.sin(1.3 * t) * e_slow
    pitch_v = 0.12 * np.sin(2.0 * t) * e_slow
    blink_v = _blinks(n, fps)
    am = np.array([sm(audio_mouth[:, k], 0.6) for k in range(3)]).T

    # ---------- video-mode motion ----------
    r_f = sm(fcol("fh", 0.3) * H * 0.62, 0.2)
    hx_f = sm(fcol("fx", 0.5) * W, 0.35)
    hy_f = sm(fcol("fy", 0.45) * H, 0.35)
    valid = face_mask if face_mask is not None and face_mask.any() else np.ones(n, bool)
    yaw_raw, pitch_raw = fcol("yaw"), fcol("pitch", 0.5)
    yaw_f = sm(np.clip((yaw_raw - np.median(yaw_raw[valid])) * 1.4, -1, 1), 0.4)
    pitch_f = sm(np.clip((pitch_raw - np.median(pitch_raw[valid])) * 4, -1, 1), 0.4)
    roll_f = sm(fcol("roll"), 0.4)
    open_f = np.clip(0.5 * fcol("lip_gap") / 0.12 + 0.6 * fcol("jaw_open"), 0, 1)
    width_f = np.clip(fcol("smile") * 1.2 + fcol("stretch") * 0.8 - (fcol("pucker") + fcol("funnel")) * 0.8, -1, 1)
    round_f = np.clip(fcol("pucker") + fcol("funnel"), 0, 1)

    alpha = np.ones(n)
    if not solo:
        # with several people: no idle 'presenter' motion, no voice-driven mouth;
        # a person who left the shot fades out after ~0.7 s
        r_v, hx_v, hy_v = r_f, hx_f, hy_f
        roll_v = roll_f
        am = am * 0
        present = face_mask.astype(float) if face_mask is not None else np.zeros(n)
        hold = int(fps * 0.7)
        if hold > 1:  # bridge short detection gaps, but never across a cut
            out = np.zeros(n, bool)
            for a0, b0 in zip(bounds, bounds[1:]):
                seg = present[a0:b0]
                out[a0:b0] = np.convolve(seg, np.ones(2 * hold + 1), mode="same")[:len(seg)] > 0 if len(seg) else seg
            present = out
        alpha = np.clip(sm(present.astype(float), 0.4), 0, 1)

    w = face_w
    v = 1 - w

    def mix(a, b):
        return w * a + v * b

    # ---------- emotions per frame ----------
    emo = np.zeros((n, len(FX_KEYS)))
    for seg, name in zip(segments or [], emotions or []):
        a, b = int(seg["start"] * fps), int(seg["end"] * fps) + 1
        emo[max(a, 0):min(b, n)] = EMOTION_FX.get(name, EMOTION_FX["neutral"])
    emo = np.array([sm(emo[:, k], 0.06) for k in range(len(FX_KEYS))]).T
    fx = {k: emo[:, i] for i, k in enumerate(FX_KEYS)}
    k = 1 - 0.6 * w  # real face already shows emotion; overlay is lighter in video mode

    bounce = fx["bounce"] * e_slow * 0.06
    r = mix(r_f, r_v)
    s = {
        "r": r,
        "hx": mix(hx_f, hx_v),
        "hy": mix(hy_f, hy_v) - bounce * r,
        "roll": mix(roll_f, roll_v),
        "yaw": mix(yaw_f, yaw_v),
        "pitch": mix(pitch_f, pitch_v),
        "blink_l": np.clip(mix(sm(fcol("blink_l"), 0.7), blink_v), 0, 1),
        "blink_r": np.clip(mix(sm(fcol("blink_r"), 0.7), blink_v), 0, 1),
        "gaze_x": np.clip(mix(sm(fcol("gaze_x") * 1.5, 0.5), yaw_v * 0.6) + fx["gaze_x"] * k, -1, 1),
        "gaze_y": np.clip(mix(sm(fcol("gaze_y") * 1.5, 0.5), zeros) + fx["gaze_y"] * k, -1, 1),
        "brow_up": np.clip(mix(sm(fcol("brow_up") * 1.3, 0.5), e_slow * 0.25) + fx["brow_up"] * k, 0, 1),
        "brow_down": np.clip(mix(sm(fcol("brow_down") * 1.3, 0.5), zeros) + fx["brow_down"] * k, 0, 1),
        "brow_tilt": np.clip(mix(sm(fcol("brow_tilt"), 0.5), zeros) + fx["brow_tilt"] * k, -1, 1),
        "open": np.clip(mix(sm(open_f, 0.7), am[:, 0]), 0, 1),
        "width": np.clip(mix(sm(width_f, 0.6), am[:, 1]), -1, 1),
        "round": np.clip(mix(sm(round_f, 0.6), am[:, 2]), 0, 1),
        "smile": np.clip(mix(sm(fcol("smile") * 1.2, 0.5), zeros) + fx["smile"] * k, -1, 1),
        "blush": np.clip(fx["blush"], 0, 1),
        "eye_smile": np.clip(fx["eye_smile"] * k + w * np.clip(sm(fcol("smile"), 0.5) - 0.4, 0, 1), 0, 1),
        "eye_wide": np.clip(fx["eye_wide"] * k, 0, 1),
    }

    # ---------- body: shoulders, elbows, wrists ----------
    hx, hy = s["hx"], s["hy"]
    pose_ok = col("pose_ok")
    pose_mask = None if pose_ok is None else (fill(pose_ok) > 0.5)
    sway = np.sin(1.7 * t) * e_slow
    for side, sign in (("l", -1), ("r", 1)):
        # default (no pose) positions relative to the head
        sh_x = hx + sign * r * 1.15
        sh_y = hy + r * 1.65
        lift = sm(np.clip(e_slow * 1.4 + (fx["bounce"] * 0.6), 0, 1), 0.1)
        if side == "r":
            lift = lift * (0.6 + 0.4 * np.sin(0.9 * t + 1.0))
        el_x = sh_x + sign * r * (0.25 + 0.15 * lift)
        el_y = sh_y + r * (1.0 - 0.15 * lift)
        wr_x = el_x - sign * r * (0.1 + 0.55 * lift) + sway * r * 0.1
        wr_y = el_y + r * (0.9 - 1.3 * lift)

        def joint(name, dx, dy):
            if pose_mask is None or not pose_mask.any():
                return dx, dy, zeros
            vis = fill(col(name + "_v"), pose_mask)
            vis = np.zeros(n) if vis is None else vis
            m = pose_mask & (vis > 0.5)
            if not m.any():
                return dx, dy, zeros
            px = fill(col(name + "_x"), m) * W
            py = fill(col(name + "_y"), m) * H
            pw = sm(m.astype(float), 0.15)
            return sm(px, 0.35), sm(py, 0.35), pw

        px, py, pw = joint(f"{side}sh", sh_x, sh_y)
        s[f"{side}sh_x"], s[f"{side}sh_y"] = pw * px + (1 - pw) * sh_x, pw * py + (1 - pw) * sh_y
        # keep the arm attached: default elbow/wrist follow the (possibly tracked) shoulder
        ox, oy = s[f"{side}sh_x"] - sh_x, s[f"{side}sh_y"] - sh_y
        px, py, pw = joint(f"{side}el", el_x + ox, el_y + oy)
        s[f"{side}el_x"], s[f"{side}el_y"] = pw * px + (1 - pw) * (el_x + ox), pw * py + (1 - pw) * (el_y + oy)
        px, py, pw = joint(f"{side}wr", wr_x + ox, wr_y + oy)
        s[f"{side}wr_x"], s[f"{side}wr_y"] = pw * px + (1 - pw) * (wr_x + ox), pw * py + (1 - pw) * (wr_y + oy)

    # subtitle segment per frame
    seg_idx = np.full(n, -1, dtype=np.int64)
    for i, seg in enumerate(segments or []):
        a, b = int(seg["start"] * fps), int(seg["end"] * fps) + 1
        seg_idx[max(a, 0):min(b, n)] = i
    s["seg"] = seg_idx
    s["alpha"] = alpha
    s["face_w"] = face_w
    return s
