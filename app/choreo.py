"""Director mode - animation the way Claude Design builds it.

Claude does not copy a video frame by frame. It writes a shot list, places each
character once per shot, and plays clean, eased gestures keyed to time (its engine:
"define three motion helpers up front and use no easing or transform outside them").
This module does the same:

  * shots come from cut detection; tracking is used only to know WHO is in a shot
    and roughly WHERE (median position) - never to move hands or heads per frame
  * every character stands in one fixed spot per shot (no overlaps)
  * arms play poses from a gesture library, blended with ease-in-out
  * the mouth moves with the voice only for the person speaking in that moment
  * blinks on a timer, listeners turn their heads toward the speaker
  * a slow camera push-in per shot
Every value is a pure function of time, so every frame is clean.
"""
import math

import numpy as np

from .tracking import IDX

# ---------------- motion helpers (the only easing used) ----------------


def ease(x):
    """easeInOutSine on 0..1."""
    x = np.clip(x, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def ramp(t, start, end, fade):
    """1 inside [start, end], eased fades of `fade` seconds at both ends."""
    up = ease((t - start) / max(fade, 1e-3) + 0.5)
    down = ease((end - t) / max(fade, 1e-3) + 0.5)
    return np.minimum(up, down)


# ---------------- gesture library ----------------
# Arm poses per side: (elbow_out, elbow_down, wrist_out, wrist_down) in head radii,
# relative to that side's shoulder. "out" = away from the body centre.
POSES = {
    "idle":          (0.22, 1.00, 0.12, 1.95),
    "hands_on_desk": (0.42, 0.95, -0.25, 1.45),
    "arms_crossed":  (0.30, 1.00, -0.95, 0.95),
    "hold_mic":      (0.35, 0.85, -0.70, -0.50),   # fist beside the chin, mic points at the mouth
    "raise_hand":    (0.55, -0.25, 0.55, -1.35),
    "point":         (0.85, 0.25, 1.75, 0.10),
    "hand_on_head":  (0.95, -0.55, 0.10, -2.15),
    "hand_on_chin":  (0.35, 1.00, -0.85, -0.55),
    "hand_on_chest": (0.30, 1.00, -0.55, 0.75),
    "shrug":         (0.45, 0.70, 0.95, 0.35),
}
GESTURES = ["talk", "laugh", "clap", "raise_hand", "point", "wave", "hands_on_head",
            "hand_on_chin", "shrug", "nod", "shake_head", "lean_in", "surprised", "drink"]
BASE_POSES = ["idle", "hands_on_desk", "arms_crossed", "hold_mic_left", "hold_mic_right"]
MOODS = {"neutral": 0.1, "happy": 0.4, "excited": 0.5, "amused": 0.45, "serious": -0.05,
         "sad": -0.25, "angry": -0.2, "surprised": 0.1, "thinking": 0.0}


def _arm_targets(name, t, side, speaking):
    """(elbow_out, elbow_down, wrist_out, wrist_down) arrays for one gesture over time t."""
    n = len(t)
    if name == "talk":  # open hands in front, moving on a slow beat
        ph = 0.0 if side == "l" else 1.7
        k = 0.5 + 0.5 * np.sin(t * 3.1 + ph)
        return (np.full(n, 0.40), np.full(n, 0.95), -0.15 + 0.35 * k, 1.25 - 0.35 * k)
    if name == "clap":
        k = 0.5 + 0.5 * np.sin(t * 14)
        return (np.full(n, 0.40), np.full(n, 0.90), -0.85 + 0.35 * k, np.full(n, 0.80))
    if name == "laugh":
        k = 0.5 + 0.5 * np.sin(t * 9)
        return (np.full(n, 0.35), np.full(n, 0.95), -0.45 + 0.05 * k, 1.25 + 0.06 * k)
    if name == "wave":
        if side == "r":
            k = np.sin(t * 8)
            return (np.full(n, 0.55), np.full(n, -0.25), 0.55 + 0.25 * k, np.full(n, -1.30))
        name = "idle"
    if name == "drink":
        name = "hold_mic" if side == "r" else "idle"
    if name == "hands_on_head":
        name = "hand_on_head"
    if name in ("raise_hand", "point", "hand_on_chin", "shrug") and side == "l" and name != "shrug":
        name = "idle"  # one-handed gestures use the right hand
    p = POSES.get(name, POSES["idle"])
    return tuple(np.full(n, v) for v in p)


# ---------------- shot analysis from tracking (who / where / who speaks) ----------------
def shot_layout(feats, people, shots, fps, W, H, min_share=0.12):
    """For every shot: which characters are in it and their fixed (hx, hy, r).

    Two 'people' in one shot who are almost never visible together, or who are always
    in the same spot when they are, are one real person split by the tracker: only the
    stronger one is drawn and the other becomes its alias (its talking is passed on)."""
    layouts = []
    for a, b in shots:
        cands = []
        for p in people:
            col = feats[a:b, p["id"], :]
            ok = col[:, IDX["face_ok"]] == 1
            if ok.sum() < max(3, (b - a) * min_share):
                continue
            cands.append((int(ok.sum()), p["id"], ok, col))
        cands.sort(key=lambda c: -c[0])
        here = []
        for cnt, pid, ok, col in cands:
            dup_of = None
            for k in here:
                both = ok & k["_ok"]
                if both.sum() < 0.05 * min(cnt, k["_cnt"]):
                    dup_of = k
                    break
                d = np.hypot(col[both, IDX["fx"]] - k["_col"][both, IDX["fx"]],
                             col[both, IDX["fy"]] - k["_col"][both, IDX["fy"]])
                if np.mean(d < np.maximum(col[both, IDX["fh"]], k["_col"][both, IDX["fh"]]) * 0.8) >= 0.6:
                    dup_of = k
                    break
            if dup_of is not None:
                dup_of["aliases"].append(pid)
                continue
            fx = np.median(col[ok, IDX["fx"]])
            fy = np.median(col[ok, IDX["fy"]])
            fh = np.median(col[ok, IDX["fh"]])
            r = min(fh * H * 0.62, H * 0.3)
            here.append({"pid": pid, "hx": fx * W, "hy": max(fy * H, r * 1.05), "r": r, "aliases": [],
                         "_ok": ok, "_col": col, "_cnt": cnt})
        for c in here:
            for k in ("_ok", "_col", "_cnt"):
                c.pop(k)
        layouts.append(_spread(here, W))
    return layouts


def _spread(chars, W):
    """Pushes characters apart so heads/bodies never overlap, keeps them on screen."""
    if not chars:
        return chars
    chars = sorted(chars, key=lambda c: c["hx"])
    need = sum(c["r"] * 2.3 for c in chars)
    if need > W * 0.98:  # too many for the width: shrink everyone a little
        k = W * 0.98 / need
        for c in chars:
            c["r"] *= k
    for _ in range(30):
        moved = False
        for c1, c2 in zip(chars, chars[1:]):
            gap = (c1["r"] + c2["r"]) * 1.15 - (c2["hx"] - c1["hx"])
            if gap > 0:
                c1["hx"] -= gap / 2
                c2["hx"] += gap / 2
                moved = True
        for c in chars:
            c["hx"] = min(max(c["hx"], c["r"] * 1.15), W - c["r"] * 1.15)
        if not moved:
            break
    return chars


def speakers(feats, layouts, shots, energy, fps, speech=None):
    """Per frame: pid of the person talking (-1 = nobody). The face whose lips move most
    while there is voice; in a shot with one person, that person."""
    n = len(energy)
    who = np.full(n, -1, dtype=np.int64)
    voice = np.convolve(energy, np.ones(5) / 5, mode="same") > 0.12
    if speech is not None:  # only real speech, never music or laughter
        voice = speech[:len(voice)] & (energy > 0.04)
    win = max(int(fps * 0.6), 3)
    for (a, b), lay in zip(shots, layouts):
        if not lay:
            continue
        if len(lay) == 1:
            who[a:b][voice[a:b]] = lay[0]["pid"]
            continue
        acts = []
        for c in lay:
            col = feats[a:b, c["pid"], :]
            m = np.nan_to_num(0.5 * col[:, IDX["lip_gap"]] / 0.12 + 0.6 * col[:, IDX["jaw_open"]], nan=0.0)
            m[col[:, IDX["face_ok"]] != 1] = 0
            d = np.abs(np.diff(m, prepend=m[:1]))
            acts.append(np.convolve(d, np.ones(win) / win, mode="same"))
        acts = np.array(acts)
        best = np.argmax(acts, axis=0)
        has = acts.max(axis=0) > 0.01
        biggest = int(np.argmax([c["r"] for c in lay]))  # no lip data: the closest person
        for k in range(b - a):
            if voice[a + k]:
                who[a + k] = lay[best[k] if has[k] else biggest]["pid"]
        # no flicker: a speaker turn lasts at least ~0.4 s
        seg = who[a:b]
        hold = int(fps * 0.4)
        k = 0
        while k < len(seg):
            j = k
            while j < len(seg) and seg[j] == seg[k]:
                j += 1
            if j - k < hold and k > 0:
                seg[k:j] = seg[k - 1]
            k = j
    return who


# ---------------- build per-frame states ----------------
def build_states(feats, people, shots, scripts, energy, audio_mouth, fps, n, W, H, speech=None):
    """Returns one state dict of arrays per character (same format the Puppet draws)."""
    t = np.arange(n) / fps
    layouts = shot_layout(feats, people, shots, fps, W, H)
    if speech is None:
        speech = np.ones(n, bool)
    who = speakers(feats, layouts, shots, energy, fps, speech)
    P = max(len(people), 1)
    e_s = np.convolve(energy, np.ones(7) / 7, mode="same")

    keys = ["r", "hx", "hy", "roll", "yaw", "pitch", "blink_l", "blink_r", "gaze_x", "gaze_y",
            "brow_up", "brow_down", "brow_tilt", "open", "width", "round", "smile", "blush",
            "eye_smile", "eye_wide", "alpha", "mic_l", "mic_r"]
    keys += [f"{s}{j}_{a}" for s in ("l", "r") for j in ("sh", "el", "wr") for a in ("x", "y")]
    states = [{k: np.zeros(n) for k in keys} for _ in range(P)]
    rng = np.random.default_rng(3)

    for si, ((a, b), lay) in enumerate(zip(shots, layouts)):
        if b <= a:
            continue
        tt = t[a:b] - t[a]
        dur = tt[-1] if len(tt) else 0
        script = scripts.get(si, {}) if scripts else {}
        by_pid = {c["pid"]: c for c in lay}
        for c in lay:
            pid = c["pid"]
            s = states[pid]
            sl = slice(a, b)
            r, hx, hy = c["r"], c["hx"], c["hy"]
            ids = {str(pid)} | {str(x) for x in c.get("aliases", [])}  # this character + its split-off copies
            info = next((script.get("people", {})[k] for k in [str(pid)] + sorted(ids)
                         if k in script.get("people", {})), {})
            mood = MOODS.get(info.get("mood", "neutral"), 0.1)
            base = info.get("base_pose", "idle")
            seed = rng.uniform(0, 100)
            # who talks: DeepSeek watched the shot at 10 fps -> use its talking turns;
            # otherwise fall back to lip movement in the tracking
            watched = bool(script.get("talking"))
            if watched:
                speaking = np.zeros(b - a, bool)
                for x in script["talking"]:
                    if x["who"] in ids:
                        speaking |= (tt >= x["start"]) & (tt < x["end"])
                speaking &= speech[sl]  # DeepSeek's talking turns, but only while words are spoken
                talker = np.full(b - a, -1)
                for x in script["talking"]:
                    m = (tt >= x["start"]) & (tt < x["end"]) & (talker < 0)
                    try:
                        o = int(x["who"])
                        talker[m] = next((cc["pid"] for cc in lay if o == cc["pid"] or o in cc.get("aliases", [])), o)
                    except ValueError:
                        pass
            else:
                speaking = who[sl] == pid
                talker = who[sl]
            spk_w = np.convolve(speaking.astype(float), np.ones(5) / 5, mode="same")

            # --- head: idle sway, look at whoever talks (or where DeepSeek saw them look)
            look = np.zeros(b - a)
            for k, o in enumerate(talker):
                if o >= 0 and o != pid and o in by_pid:
                    look[k] = np.sign(by_pid[o]["hx"] - hx) * 0.45
            for x in script.get("looks", []):
                if x["who"] not in ids:
                    continue
                m = (tt >= x["start"]) & (tt < x["end"])
                at = x.get("at")
                if at == "camera":
                    look[m] = 0.0
                elif at in ("left", "right"):
                    look[m] = -0.5 if at == "left" else 0.5
                else:
                    try:
                        o = int(at)
                        if o in by_pid and o != pid:
                            look[m] = np.sign(by_pid[o]["hx"] - hx) * 0.45
                    except (TypeError, ValueError):
                        pass
            look = np.convolve(look, np.ones(9) / 9, mode="same")
            s["r"][sl], s["hx"][sl] = r, hx
            s["hy"][sl] = hy + np.sin(tt * 1.3 + seed) * r * 0.012
            s["yaw"][sl] = look + 0.08 * np.sin(tt * 0.4 + seed)
            s["roll"][sl] = 0.035 * np.sin(tt * 0.5 + seed * 2)
            s["pitch"][sl] = spk_w * e_s[sl] * 0.25 * np.sin(tt * 6.0 + seed)
            s["gaze_x"][sl] = s["yaw"][sl] * 0.8

            # --- blinks on a timer (each person their own rhythm)
            period = 3.2 + (seed % 2.0)
            ph = (tt + seed) % period
            blink = np.where(ph < 0.13, 1.0, 0.0)
            s["blink_l"][sl] = s["blink_r"][sl] = blink

            # --- mouth: follows the loudness of every syllable, only while this person talks
            am = audio_mouth[sl]
            loud = np.clip(energy[sl] * 1.1, 0, 0.95) * speech[sl]  # music/laughter never moves a mouth
            s["open"][sl] = np.maximum(am[:, 0], loud) * spk_w
            s["width"][sl] = am[:, 1] * spk_w
            s["round"][sl] = am[:, 2] * spk_w
            s["smile"][sl] = mood
            s["brow_up"][sl] = 0.1 + 0.15 * spk_w * e_s[sl]

            # --- shoulders fixed under the head
            for side, sign in (("l", -1), ("r", 1)):
                s[f"{side}sh_x"][sl] = hx + sign * r * 1.15
                s[f"{side}sh_y"][sl] = hy + r * 1.65

            # --- arms: base pose for the shot, gestures blended on top
            mic_side = "l" if base == "hold_mic_left" else "r" if base == "hold_mic_right" else None
            base_name = "hold_mic" if mic_side else base
            arm = {}
            for side in ("l", "r"):
                name = base_name if (mic_side is None or side == mic_side) else "idle"
                arm[side] = [np.asarray(v, float) for v in _arm_targets(name, tt, side, speaking)]
                if mic_side == side:
                    s[f"mic_{side}"][sl] = 1.0
                # the free hand talks while this person speaks (if no gesture overrides it)
                if side != mic_side:
                    talk = _arm_targets("talk", tt, side, speaking)
                    w = np.clip(spk_w * 0.85, 0, 1)
                    arm[side] = [x * (1 - w) + np.asarray(y) * w for x, y in zip(arm[side], talk)]

            expr = {"smile": np.zeros(b - a), "eye_smile": np.zeros(b - a), "open": np.zeros(b - a),
                    "brow_up": np.zeros(b - a), "eye_wide": np.zeros(b - a), "pitch": np.zeros(b - a),
                    "nod": np.zeros(b - a), "shake": np.zeros(b - a), "lean": np.zeros(b - a)}
            for act in script.get("actions", []):
                if str(act.get("who")) not in ids:
                    continue
                g = act.get("gesture", "")
                st = float(act.get("start", 0))
                en = float(act.get("end", st + 1.5))
                w = ramp(tt, st, min(en, dur + 1), 0.35)
                if not w.any():
                    continue
                if g in ("laugh",):
                    expr["smile"] += w * 0.8
                    expr["eye_smile"] += w * 0.9
                    expr["open"] += w * (0.35 + 0.15 * np.abs(np.sin(tt * 9)))
                    expr["pitch"] -= w * 0.25
                elif g == "surprised":
                    expr["brow_up"] += w * 0.8
                    expr["eye_wide"] += w * 0.6
                    expr["open"] += w * 0.35
                elif g == "nod":
                    expr["nod"] += w
                elif g == "shake_head":
                    expr["shake"] += w
                elif g == "lean_in":
                    expr["lean"] += w
                for side in ("l", "r"):
                    if side == mic_side and g not in ("clap", "hands_on_head"):
                        continue  # the mic hand keeps the mic
                    if g in ("nod", "shake_head", "lean_in", "surprised") or (
                            g in ("raise_hand", "point", "wave", "hand_on_chin") and side == "l"):
                        continue
                    tgt = _arm_targets(g, tt, side, speaking)
                    arm[side] = [x * (1 - w) + np.asarray(y) * w for x, y in zip(arm[side], tgt)]
                    if side == mic_side:
                        s[f"mic_{side}"][sl] *= 1 - w

            s["smile"][sl] = np.clip(s["smile"][sl] + expr["smile"], -1, 1)
            s["eye_smile"][sl] = np.clip(expr["eye_smile"] + max(mood - 0.3, 0), 0, 1)
            s["open"][sl] = np.clip(s["open"][sl] + expr["open"], 0, 1)
            s["brow_up"][sl] = np.clip(s["brow_up"][sl] + expr["brow_up"], 0, 1)
            s["eye_wide"][sl] = np.clip(expr["eye_wide"], 0, 1)
            s["pitch"][sl] += expr["pitch"] + expr["nod"] * 0.18 * np.sin(tt * 7)
            s["yaw"][sl] += expr["shake"] * 0.35 * np.sin(tt * 8)
            s["r"][sl] = r * (1 + 0.06 * expr["lean"])
            s["blush"][sl] = np.clip(expr["eye_smile"] * 0.5, 0, 1)

            for side, sign in (("l", -1), ("r", 1)):
                eo, ed, wo, wd = arm[side]
                sx, sy = s[f"{side}sh_x"][sl], s[f"{side}sh_y"][sl]
                breathe = np.sin(tt * 1.6 + seed) * 0.03
                s[f"{side}el_x"][sl] = sx + sign * eo * r
                s[f"{side}el_y"][sl] = sy + (ed + breathe) * r
                s[f"{side}wr_x"][sl] = sx + sign * wo * r
                s[f"{side}wr_y"][sl] = sy + (wd + breathe) * r

            # fade in/out at the very start/end of the shot is not needed: cuts are hard
            s["alpha"][sl] = 1.0
    return states, layouts, who
