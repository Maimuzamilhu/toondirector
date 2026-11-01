"""Animator mode - works on any video (real people, cartoons, animals, full-body action).

No face detector is needed. Like Claude Design, the AI does the understanding and
writes the animation; our code only renders it:

  1. cut detection splits the video into shots
  2. CAST: DeepSeek looks at one keyframe per shot and lists every character
     (person, cartoon, animal) with a description and where they are
  3. DeepSeek draws every character full body (head, torso, arms, legs, feet) as SVG
  4. ANIMATE: DeepSeek watches EVERY frame of each shot (contact sheets, one per
     second) and writes keyframes per character: position, lean, head tilt, the
     angle of every arm/leg joint, mouth, eyes, expression; plus simple effects
  5. keyframes are eased to a smooth 30 fps and drawn over the shot's background
"""
import json
import math

import cv2
import numpy as np

from . import providers
from . import config

# ----------------------------------------------------------------- shots
def detect_shots(video, progress=None):
    """Reads every frame once: cut points + one small signature per frame."""
    from .tracking import shot_signature
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cuts, prev_sig, prev_gray, recent, last = [0], None, None, [], 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sig = shot_signature(frame)
        gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_sig is not None:
            hdiff = 0.5 * float(np.abs(sig - prev_sig).sum())
            mad = float(np.mean(np.abs(gray - prev_gray)))
            base = float(np.median(recent)) if recent else mad
            if (hdiff > 0.3 or (mad > 12 and mad > 3.5 * base + 4)) and i - last >= max(fps * 0.4, 2):
                cuts.append(i)
                last = i
                recent = []
            else:
                recent = (recent + [mad])[-int(max(fps, 3)):]
        prev_sig, prev_gray = sig, gray
        i += 1
        if progress and i % 30 == 0:
            progress(min(i / total, 1.0), f"Reading frame {i}/{total}")
    cap.release()
    n = i
    shots = [[a, b] for a, b in zip(cuts, cuts[1:] + [n]) if b > a]
    return {"fps": fps, "n": n, "w": w, "h": h, "shots": shots}


# ----------------------------------------------------------------- DeepSeek calls
CAST_PROMPT = """You are casting a 2D cartoon remake of a video. Image N is the middle frame of shot S{{N}} ({k} shots).
List EVERY character that appears (people, cartoon characters, animals, creatures). The same character in different shots must get the SAME id - judge by face, hair, body shape and clothes.
For every shot where a character is visible, give its bounding box [x0, y0, x1, y1] as fractions of the image (0..1, whole body that is visible).
Write a short visual description per character (body build, head/face, hair, clothes, colours, accessories, art style if it is already a drawing).
Also give head_ratio = head height / full body height as drawn in the video (a real adult is about 0.13; caricatures with big heads 0.25-0.4).
Give "kind": human, crowd (a group drawn as one), bird, seal (any blob-shaped sea animal), or animal (other four-legged animals).
Also give "look" for a simple explainer-style version of the character: body = main clothing colour, accent = a second colour (hat, scarf), accessory = one of none, cap, beanie, bun, long_hair, ponytail, hat, crown, glasses, scarf, dress, beard (the single most recognisable feature), height = relative body height (1.0 normal, 0.6 child, 1.15 tall).
Background extras in a crowd count as ONE character called "crowd" at most - list the main characters first.

Reply with JSON only:
{{"characters": [{{"id": "c1", "name": "short name", "description": "...", "head_ratio": 0.13, "kind": "human", "look": {{"body": "#3aa6a0", "accent": "#e8584a", "accessory": "cap", "height": 1.0}}, "boxes": {{"S1": [0.1, 0.2, 0.5, 0.9]}}}}]}}"""


ANIMATE_PROMPT = """You are a 2D animator. Watch this shot of a video: each image is one second of it, showing EVERY frame of that second in order (left-to-right, top-to-bottom), each labelled with its time from the start of this part. This part runs from {t0:.2f}s to {t1:.2f}s of the shot ({fps:g} frames per second in the source).
Characters in this shot:
{chars}

ONLY write keys for characters you can actually SEE in these frames - never list a character that is not on screen (if nobody is visible, "keys" is {{}}). Measure positions carefully from the frames: x, y, h must match where and how big the character really is.
Write KEYFRAMES so a cartoon puppet of each character copies what it does. Add a key whenever its pose or position changes (at least every 0.5 s while it moves; 0.1 s precision). Copy the real motion: walking, jumping, dancing, punching, flexing, turning, sitting, talking, etc.
Key fields (all required):
- t: seconds from the start of THIS part
- x, y: the character's HIP centre in the image, fractions 0..1 (y grows downwards)
- h: the character's full standing height as a fraction of image height (head top to feet, even if the feet are off screen)
- facing: 1 = facing right / towards camera-right, -1 = facing left
- lean: torso lean in degrees (positive = top of the body tilts to image-right)
- head: head tilt in degrees (positive = towards image-right)
- arm_l, arm_r: [shoulder, elbow] angles in degrees for the arm on the image-left / image-right side. 0 = hanging straight down; positive rotates the limb towards image-right, negative towards image-left (so raising the image-right arm sideways is about +90, straight up is +170; the image-left arm raised sideways is -90). elbow is relative to the upper arm.
- leg_l, leg_r: [hip, knee] same convention (standing = [0, 0]; a knee bent backwards is negative for a right-facing character).
- pose: the overall body pose, one of stand, sit, walk, run, jump, crouch, lie, wave, point, cheer, think (always give it - e.g. people on chairs/benches/sofas are "sit")
- mouth: 0 closed .. 1 wide open; talking: true if this character is the one speaking at that moment
- eyes: "open", "closed", "wide" or "squint"; mood: neutral, happy, laughing, angry, sad, surprised, smug, scared
Overlays (optional): on-screen graphics that are part of the video (badges like "+75 YEARS", counters like "POPULATION 128", small caption pills, big titles) - copy the text exactly: {{"type": "badge" | "label" | "title", "text": "...", "x": 0.5, "y": 0.1, "start": t, "end": t}} (x, y = centre of the graphic, fractions of the image).
Effects (optional): straight things drawn on top, like a lasso rope, laser beams, a thrown line: {{"type": "line", "color": "#rrggbb", "width": 0.01, "from": [x, y], "to": [x, y], "start": t, "end": t}} (positions as fractions of the image, width as a fraction of image width).

Reply with JSON only:
{{"keys": {{"<character id>": [{{"t": 0.0, "x": 0.5, "y": 0.6, "h": 0.8, "facing": 1, "lean": 0, "head": 0, "arm_l": [0, 0], "arm_r": [0, 0], "leg_l": [0, 0], "leg_r": [0, 0], "pose": "stand", "mouth": 0, "talking": false, "eyes": "open", "mood": "neutral"}}]}},
 "effects": [], "overlays": []}}"""


def cast(video, shots, log=print):
    """One keyframe per shot -> characters with ids, descriptions and boxes per shot."""
    from . import design
    sel = list(range(len(shots)))
    if len(sel) > 40:  # very long videos: a spread of shots
        sel = [int(round(k)) for k in np.linspace(0, len(shots) - 1, 40)]
    from .tracking import read_frames
    mids = {k: (shots[k][0] + shots[k][1]) // 2 for k in sel}
    fr = read_frames(video, list(mids.values()))
    imgs = [(k, fr[mids[k]]) for k in sel if mids[k] in fr]
    content = [{"type": "text", "text": CAST_PROMPT.format(k=len(imgs))}]
    content += [{"type": "image_url", "image_url": {"url": design._jpeg_data_url(img, 640)}} for _, img in imgs]
    data = design._json_call(content, max_tokens=6000)
    label = {f"S{n + 1}": k for n, (k, _) in enumerate(imgs)}
    chars = []
    for c in data.get("characters") or []:
        boxes = {}
        for s, b in (c.get("boxes") or {}).items():
            k = label.get(str(s).upper())
            try:
                bx = [min(max(float(v), 0.0), 1.0) for v in b][:4]
            except (TypeError, ValueError):
                continue
            if k is not None and len(bx) == 4 and bx[2] > bx[0] and bx[3] > bx[1]:
                boxes[k] = bx
        if boxes:
            try:
                hr = min(max(float(c.get("head_ratio", 0.13)), 0.1), 0.45)
            except (TypeError, ValueError):
                hr = 0.13
            look = c.get("look") if isinstance(c.get("look"), dict) else {}
            chars.append({"id": str(c.get("id") or f"c{len(chars) + 1}"), "name": str(c.get("name", ""))[:40],
                          "description": str(c.get("description", ""))[:500], "boxes": boxes, "head_ratio": hr,
                          "kind": str(c.get("kind", "human")),
                          "look": _clean_look(look)})
    # main characters first: big on screen and in many shots; crowd extras last
    chars.sort(key=lambda c: -(sum((b[2] - b[0]) * (b[3] - b[1]) for b in c["boxes"].values())
                               + 0.05 * len(c["boxes"])) * (0.2 if "crowd" in c["name"].lower() else 1.0))
    log(f"DeepSeek found {len(chars)} characters: " + ", ".join(c["name"] or c["id"] for c in chars))
    return chars


def animate_window(video, fps, a, b, shot_start, chars_desc, rate):
    """DeepSeek watches frames [a, b) at `rate` fps and writes keyframes (times relative
    to the shot start)."""
    from . import design, watch
    empty = np.zeros((b, 0, 40), np.float32)  # no tracking boxes in animator mode
    sheets = watch.contact_sheets(video, empty, [], a, b, fps, rate=rate)
    t0, t1 = (a - shot_start) / fps, (b - shot_start) / fps
    content = [{"type": "text", "text": ANIMATE_PROMPT.format(t0=t0, t1=t1, fps=fps, chars=chars_desc)}]
    content += [{"type": "image_url", "image_url": {"url": design._jpeg_data_url(img, 1000)}} for _, img in sheets]
    data = design._json_call(content, max_tokens=8000, temperature=0.2)
    keys = {}
    for cid, ks in (data.get("keys") or {}).items():
        good = []
        for kf in ks or []:
            try:
                kf = dict(kf)
                kf["t"] = float(kf.get("t", 0)) + t0
                good.append(kf)
            except (TypeError, ValueError):
                pass
        if good:
            keys[str(cid)] = sorted(good, key=lambda k: k["t"])
    effects = []
    for e in data.get("effects") or []:
        try:
            effects.append({"type": "line", "color": str(e.get("color", "#ffffff")), "width": float(e.get("width", 0.01)),
                            "from": [float(v) for v in e["from"]][:2], "to": [float(v) for v in e["to"]][:2],
                            "start": float(e["start"]) + t0, "end": float(e["end"]) + t0})
        except (KeyError, TypeError, ValueError):
            pass
    overlays = []
    for o in data.get("overlays") or []:
        try:
            overlays.append({"type": o.get("type") if o.get("type") in ("badge", "label", "title") else "label",
                             "text": str(o.get("text", ""))[:80], "x": float(o.get("x", 0.5)),
                             "y": float(o.get("y", 0.1)), "start": float(o["start"]) + t0, "end": float(o["end"]) + t0})
        except (KeyError, TypeError, ValueError):
            pass
    return keys, effects, overlays


ACCESSORIES = ("none", "cap", "beanie", "bun", "long_hair", "ponytail", "hat", "crown", "turban", "helmet",
               "glasses", "scarf", "dress", "robe", "beard")


def _clean_look(look):
    import re as _re
    out = {"body": "#3aa6a0", "accent": "#e8584a", "accessory": "none", "height": 1.0}
    for k in ("body", "accent"):
        v = str(look.get(k, ""))
        if _re.match(r"^#[0-9a-fA-F]{6}$", v):
            out[k] = v
    r, g, b = (int(out["body"][i:i + 2], 16) for i in (1, 3, 5))
    if 0.3 * r + 0.59 * g + 0.11 * b < 70:  # black clothes vanish against the black head and limbs
        out["body"] = "#{:02x}{:02x}{:02x}".format(*(min(int(v * 0.6 + 95), 255) for v in (r, g, b)))
    if look.get("accessory") in ACCESSORIES:
        out["accessory"] = look["accessory"]
    try:
        out["height"] = min(max(float(look.get("height", 1.0)), 0.5), 1.3)
    except (TypeError, ValueError):
        pass
    return out


# ----------------------------------------------------------------- keyframes -> per-frame states
MOOD = {"neutral": (0.1, 0, 0, 0), "happy": (0.5, 0.3, 0, 0.1), "laughing": (0.8, 0.8, 0, 0.2),
        "angry": (-0.3, 0, 0.7, 0), "sad": (-0.35, 0, 0, 0), "surprised": (0.0, 0, 0, 0.8),
        "smug": (0.35, 0.4, 0.2, 0), "scared": (-0.2, 0, 0, 0.6)}  # smile, eye_smile, brow_down, brow_up


def _num(kf, key, default=0.0):
    try:
        return float(kf.get(key, default))
    except (TypeError, ValueError):
        return default


def _pair(kf, key):
    v = kf.get(key) or [0, 0]
    try:
        return float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0


def _ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


def sample(keys, t):
    """Eased interpolation of a character's keyframes at time t (seconds in the shot)."""
    if t <= keys[0]["t"]:
        a = b = keys[0]
        u = 0.0
    elif t >= keys[-1]["t"]:
        a = b = keys[-1]
        u = 0.0
    else:
        j = next(k for k in range(1, len(keys)) if keys[k]["t"] >= t)
        a, b = keys[j - 1], keys[j]
        span = max(b["t"] - a["t"], 1e-3)
        u = _ease((t - a["t"]) / span)

    def lerp(x, y):
        return x + (y - x) * u
    s = {}
    for k, d in (("x", 0.5), ("y", 0.6), ("h", 0.7), ("lean", 0), ("head", 0), ("mouth", 0)):
        s[k] = lerp(_num(a, k, d), _num(b, k, d))
    for k in ("arm_l", "arm_r", "leg_l", "leg_r"):
        pa, pb = _pair(a, k), _pair(b, k)
        s[k] = (lerp(pa[0], pb[0]), lerp(pa[1], pb[1]))
    near = a if u < 0.5 else b
    s["facing"] = 1 if _num(near, "facing", 1) >= 0 else -1
    s["pose"] = str(near.get("pose", "stand"))
    s["talking"] = bool(near.get("talking"))
    s["eyes"] = str(near.get("eyes", "open"))
    ma, mb = MOOD.get(str(a.get("mood")), MOOD["neutral"]), MOOD.get(str(b.get("mood")), MOOD["neutral"])
    s["smile"], s["eye_smile"], s["brow_down"], s["brow_up"] = (lerp(x, y) for x, y in zip(ma, mb))
    return s


# clean limb angles for each named pose; f = facing (+1 right, -1 left)
def pose_limbs(pose, f):
    fwd = "r" if f > 0 else "l"
    base = {"arm_l": (-8, 0), "arm_r": (8, 0), "leg_l": (-3, 0), "leg_r": (3, 0)}
    if pose == "sit":
        base.update(leg_l=(f * 85, -f * 85), leg_r=(f * 80, -f * 80), arm_l=(f * 25, f * 45), arm_r=(f * 25, f * 45))
    elif pose == "crouch":
        base.update(leg_l=(f * 55, -f * 105), leg_r=(f * 50, -f * 100), arm_l=(f * 30, f * 30), arm_r=(f * 30, f * 30))
    elif pose == "jump":
        base.update(leg_l=(f * 30, -f * 60), leg_r=(f * 20, -f * 50), arm_l=(-140, -10), arm_r=(140, 10))
    elif pose == "cheer":
        base.update(arm_l=(-160, -10), arm_r=(160, 10))
    elif pose == "wave":
        base["arm_" + fwd] = (f * 150, f * 25)
    elif pose == "point":
        base["arm_" + fwd] = (f * 90, 0)
    elif pose == "think":
        base["arm_" + fwd] = (f * 25, f * 140)
    elif pose == "lie":
        base.update(leg_l=(f * 90, 0), leg_r=(f * 90, 0))
    elif pose == "open_arms":
        base.update(arm_l=(-72, -22), arm_r=(72, 22))
    elif pose == "hips":
        base.update(arm_l=(-52, 98), arm_r=(52, -98))
    elif pose == "shrug":
        base.update(arm_l=(-58, -78), arm_r=(58, 78))
    elif pose == "salute":
        base["arm_" + fwd] = (f * 105, f * 100)
    elif pose == "bow":
        base.update(arm_l=(f * 16, 0), arm_r=(f * 16, 0))
    elif pose == "fist":
        base["arm_" + fwd] = (f * 152, f * 12)
    elif pose in ("read", "write"):
        base.update(arm_l=(f * 38, f * 72), arm_r=(f * 30, f * 88))
    elif pose == "kneel":
        base.update(leg_l=(f * 80, -f * 80), leg_r=(-f * 12, -f * 78), arm_l=(f * 18, f * 10), arm_r=(f * 18, f * 10))
    return base


def apply_pose(kf):
    """Fills in clean limbs from the pose name when DeepSeek gave no real angles, and
    always for sitting/crouching (those are hard to describe as raw angles)."""
    kf = dict(kf)
    pose = str(kf.get("pose", "stand"))
    limbs = [_pair(kf, k) for k in ("arm_l", "arm_r", "leg_l", "leg_r")]
    empty = sum(abs(v) for pr in limbs for v in pr) < 1
    if empty or pose in ("sit", "crouch", "lie"):
        f = 1 if _num(kf, "facing", 1) >= 0 else -1
        for k, v in pose_limbs(pose, f).items():
            if pose in ("sit", "crouch", "lie") and k.startswith("arm") and not empty:
                continue  # keep the arm angles DeepSeek saw
            kf[k] = list(v)
        if pose == "lie":
            kf["lean"] = 90 * f
    return kf


def head_scale(c):
    """How much bigger than normal this character's head is drawn (caricatures: 2-3x)."""
    return min(max(float(c.get("head_ratio", 0.13)) / 0.13, 1.0), 3.0)


def blink(t, seed):
    period = 3.1 + (seed % 1.7)
    return 1.0 if (t + seed) % period < 0.12 else 0.0


# ----------------------------------------------------------------- the whole mode
def _crop_box(img, box, pad=0.08):
    hh, ww = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, 1), min(y1 + pad, 1)
    crop = img[int(y0 * hh):int(y1 * hh), int(x0 * ww):int(x1 * ww)].copy()
    # outline the character so the artist draws the right one
    bx0, by0 = int((box[0] - x0) * ww), int((box[1] - y0) * hh)
    bx1, by1 = int((box[2] - x0) * ww), int((box[3] - y0) * hh)
    cv2.rectangle(crop, (bx0, by0), (bx1, by1), (0, 255, 255), 2)
    return crop


def face_keys(open_, smile, blink_, eye_smile, brow_down, brow_up, eye_wide):
    return {"open": open_, "width": 0.1, "round": 0.0, "smile": smile, "blink_l": blink_, "blink_r": blink_,
            "gaze_x": 0, "gaze_y": 0, "brow_up": brow_up, "brow_down": brow_down, "brow_tilt": 0,
            "blush": eye_smile * 0.4, "eye_smile": eye_smile, "eye_wide": eye_wide, "pitch": 0}


def neutral(W, H, h=0.8):
    k = h * H / 1020
    s = {"x": W / 2, "y": H * 0.55, "k": k, "lean": 0, "head": 0, "facing": 1,
         "arm_l": (-12, 0), "arm_r": (12, 0), "leg_l": (-4, 0), "leg_r": (4, 0), "alpha": 1}
    s.update(face_keys(0.2, 0.3, 0, 0.3, 0, 0, 0))
    return s


def preview_png(style, parts, W=480, H=640):
    import io
    from .character import Painter
    from .sprites import build_rig
    p = Painter(W, H, [(style, build_rig(parts, 1.0))], {}, bg_style={"bg_top": "#f2efe8", "bg_bottom": "#dcd6cc"},
                full_body=True)
    p.draw([neutral(W, H)])
    buf = io.BytesIO()
    p.surface.write_to_png(buf)
    return buf.getvalue()


def _box_for(c, k):
    return c["boxes"].get(k) if k in c["boxes"] else c["boxes"].get(str(k))


def run(job, d, video, wav, info, opts, prog, log):
    """Runs animator mode after prepare/audio/transcribe. Writes d/output.mp4."""
    import shutil
    import time
    from . import design, director, lipsync, media, pipeline
    from .character import Painter
    from .sprites import background_surface, build_rig
    from .tracking import read_frames
    t_start = time.time()
    if not providers.configured():
        raise RuntimeError("Animator mode needs a DeepSeek API key (Settings).")
    ddir = d / "design"
    ddir.mkdir(exist_ok=True)
    desc = opts.get("description", "")

    # 1. shots (every frame is read once)
    prog.stage("track")
    spath = ddir / "anim_shots.json"
    sh = pipeline._load_json(spath)
    if not sh:
        sh = detect_shots(video, prog.cb("track"))
        pipeline._save_json(spath, sh)
    fps, n, vw, vh, shots = sh["fps"], sh["n"], sh["w"], sh["h"], sh["shots"]
    log(f"{n} frames at {fps:g} fps, {len(shots)} shots")

    # 2. cast: every character in the video
    prog.stage("lipsync")
    cpath = ddir / "anim_cast.json"
    chars = pipeline._load_json(cpath)
    if chars is None:
        log("DeepSeek is looking at every shot to find all the characters...")
        chars = cast(video, shots, log)
        for c in chars:  # json keys must be strings
            c["boxes"] = {str(k): v for k, v in c["boxes"].items()}
        pipeline._save_json(cpath, chars)
    if not chars:
        raise RuntimeError("DeepSeek found no characters in this video.")
    chars = chars[: int(opts.get("max_people") or 8)]

    # 3. draw characters (full body, in the source's style) + backgrounds
    prog.stage("design")
    best = {}
    for c in chars:
        k = max(c["boxes"], key=lambda s: (c["boxes"][s][2] - c["boxes"][s][0]) * (c["boxes"][s][3] - c["boxes"][s][1]))
        best[c["id"]] = (int(k), c["boxes"][k])
    fr = read_frames(video, [(shots[k][0] + shots[k][1]) // 2 for k, _ in best.values()])
    crops = {}
    for i, c in enumerate(chars):
        k, box = best[c["id"]]
        img = fr.get((shots[k][0] + shots[k][1]) // 2)
        if img is not None:
            crops[c["id"]] = _crop_box(img, box)
            cv2.imwrite(str(ddir / f"person_{i}.jpg"), crops[c["id"]])
    base_style = dict(director.DEFAULT_STYLE)

    def make(c):
        i = chars.index(c)
        rp = ddir / f"rig_{i}.json"
        cached = pipeline._load_json(rp)
        if cached:
            return cached
        others = [f"{o['name']}: {o['description']}" for o in chars if o is not c]
        log(f"DeepSeek is drawing {c['name'] or c['id']} (full body, in the video's style)...")
        try:
            dz = design.draw_character(crops.get(c["id"]), desc, log, note=f"{c['name']}: {c['description']}",
                                       others=others, full_body=True, style=design.STYLE_MATCH)
            for npass in range(int(config.REVIEW_PASSES or 0)):
                if c["id"] not in crops or not dz["parts"]:
                    break
                png = preview_png(pipeline._apply_colors(base_style, dz["colors"]), dz["parts"])
                (ddir / f"review_{i}_{npass + 1}.png").write_bytes(png)
                dz = design.review_character(crops[c["id"]], dz, png, log)
        except Exception as e:
            log(f"Drawing {c['name']} failed ({e}); simple drawing used.")
            dz = {"parts": {}, "colors": {}}
        pipeline._save_json(rp, dz)
        return dz

    explainer_style = opts.get("art") == "explainer"
    if explainer_style:
        # simple code-drawn characters: nothing to draw with DeepSeek, just previews
        rigs = [{} for _ in chars]
        styles = [base_style for _ in chars]
        for i, c in enumerate(chars):
            (ddir / f"character_{i}.png").write_bytes(stick_preview(c.get("look", {})))
    else:
        rigs = design.run_parallel(make, chars, workers=3)
        styles = [pipeline._apply_colors(base_style, r.get("colors", {})) for r in rigs]
        for i, (st, rg) in enumerate(zip(styles, rigs)):
            (ddir / f"character_{i}.png").write_bytes(preview_png(st, rg.get("parts", {})))
    track = {"shots": shots, "w": vw, "h": vh, "fps": fps}
    o2 = dict(opts, _no_chars=True)
    if explainer_style:
        o2["_set_style"] = design.EXPLAINER_STYLE
        o2.setdefault("max_backgrounds", 40)
    _, _, bg_svgs, shot_group, fg_svgs = pipeline.design_stage(job, d, video, track, [], o2, prog.cb("design"), log)

    # 4. DeepSeek watches every frame of every shot and writes keyframes
    prog.stage("direct")
    kpath = ddir / "anim_keys.json"
    anim = {int(k): v for k, v in (pipeline._load_json(kpath) or {}).items()}
    want = config.WATCH_FPS.lower()
    rate = int(round(fps)) if want in ("all", "max", "every") else max(1, int(float(want or 10)))
    rate = max(1, min(rate, int(round(fps))))
    log(f"Watching {rate} of {fps:g} frames per second" + (" (every frame)" if rate >= int(round(fps)) else ""))
    windows = []
    for k, (a, b) in enumerate(shots):
        if k in anim:
            continue
        step = int(max(fps * 3, 1))  # 3-second parts: DeepSeek looks closely instead of skimming
        for w0 in range(a, b, step):
            windows.append((k, w0, min(w0 + step, b)))

    def in_shot(k):
        """Characters the casting saw in shot k (all of them if the casting did not cover it)."""
        seen = [c for c in chars if _box_for(c, k)]
        return seen if seen else chars

    def chars_for(k):
        lines = []
        for c in in_shot(k):
            where = _box_for(c, k)
            hint = f" (seen around box {[round(v, 2) for v in where]})" if where else ""
            lines.append(f'- id "{c["id"]}" {c["name"]}: {c["description"][:160]}{hint}')
        return "\n".join(lines) + "\nOnly give keys for characters that are visible in this part."

    done = [0]

    def work(win):
        k, a, b = win
        try:
            res = animate_window(video, fps, a, b, shots[k][0], chars_for(k), rate)
        except Exception as e:
            log(f"Shot {k + 1}: animating failed ({e})")
            res = None
        done[0] += 1
        prog.cb("direct")(done[0] / max(len(windows), 1), f"DeepSeek animating part {done[0]}/{len(windows)}")
        return res

    for (k, a, b), res in zip(windows, design.run_parallel(work, windows, workers=4)):
        if not res:
            continue
        keys, effects, overlays = res
        sc = anim.setdefault(k, {"keys": {}, "effects": [], "overlays": []})
        sc.setdefault("overlays", []).extend(overlays)
        allowed = {c["id"] for c in in_shot(k)}
        for cid, ks in keys.items():
            if cid not in allowed:
                continue  # DeepSeek listed someone who is not in this shot
            sc["keys"].setdefault(cid, []).extend(apply_pose(q) for q in ks)
            sc["keys"][cid].sort(key=lambda x: x["t"])
        sc["effects"].extend(effects)
    pipeline._save_json(kpath, {str(k): v for k, v in anim.items()})
    nkeys = sum(len(ks) for v in anim.values() for ks in v["keys"].values())
    log(f"DeepSeek wrote {nkeys} keyframes for {len(anim)} shots")

    # 5. render at a smooth frame rate
    prog.stage("render")
    H = pipeline._even(opts.get("height") or 720)
    W = pipeline._even(H * vw / vh)
    out_fps = 30.0 if fps < 24 else min(fps, 60.0)
    data, sr = lipsync.read_wav(wav)
    duration = min(len(data) / sr, n / fps)
    n_out = int(round(duration * out_fps))
    energy = lipsync.frame_energy(wav, out_fps, n_out)
    tpath = d / "transcript.json"
    segs = (pipeline._load_json(tpath) or {}).get("segments", []) if tpath.exists() else []
    speech = lipsync.speech_mask(segs, out_fps, n_out) if segs else np.ones(n_out, bool)
    k_max = max((float(kf.get("h", 0.7)) for v in anim.values() for ks in v["keys"].values() for kf in ks), default=0.8)
    zoom = min(max(k_max * H / 1020 * 1.2, 0.4), 4.0)
    puppets = [(styles[i], build_rig(rigs[i].get("parts", {}), zoom)) for i in range(len(chars))]
    backgrounds, foregrounds = {}, {}
    for g, svg in bg_svgs.items():
        try:
            backgrounds[g] = background_surface(svg, W, H)
            if g in fg_svgs:
                foregrounds[g] = background_surface(fg_svgs[g], W, H)
        except Exception as e:
            log(f"Background {g} could not be rendered: {e}")
    grey = {"bg_top": "#8a8a8a", "bg_bottom": "#7a7a7a"}
    painter = Painter(W, H, puppets, backgrounds, bg_style=grey, foregrounds=foregrounds, full_body=True)
    if explainer_style:
        render_explainer(job, d, chars, anim, shots, shot_group, backgrounds, foregrounds, fps, W, H, wav,
                         energy, out_fps, n_out, prog, log, t_start, speech)
        return
    shot_start = [a / fps for a, _ in shots]
    idx = {c["id"]: i for i, c in enumerate(chars)}
    out = d / "output.tmp.mp4"
    enc = media.open_encoder(out, W, H, out_fps, wav, n_out)
    cb = prog.cb("render")
    try:
        for f in range(n_out):
            if job.cancelled:
                raise RuntimeError("Cancelled")
            t = f / out_fps
            si = max(0, int(np.searchsorted(shot_start, t, side="right")) - 1)
            sc = anim.get(si, {"keys": {}, "effects": []})
            ts = t - shot_start[si]
            states = [None] * len(chars)
            for cid, ks in sc["keys"].items():
                i = idx.get(cid)
                if i is None or not ks:
                    continue
                kv = sample(ks, ts)
                eyes = kv["eyes"]
                bl = 1.0 if eyes == "closed" else 0.55 if eyes == "squint" else blink(t, i * 1.3)
                loud = float(energy[f]) if kv["talking"] and speech[min(f, len(speech) - 1)] else 0.0
                open_ = max(kv["mouth"] * 0.6, loud * 1.1)
                hs = head_scale(chars[i])
                st = {"x": kv["x"] * W, "y": kv["y"] * H, "k": max(kv["h"], 0.05) * H / (860 + 265 * hs),
                      "head_scale": hs,
                      "lean": kv["lean"], "head": kv["head"], "facing": kv["facing"],
                      "arm_l": kv["arm_l"], "arm_r": kv["arm_r"], "leg_l": kv["leg_l"], "leg_r": kv["leg_r"],
                      "alpha": 1.0}
                st.update(face_keys(min(open_, 1.0), kv["smile"], bl, kv["eye_smile"], kv["brow_down"],
                                    kv["brow_up"], 0.6 if eyes == "wide" else 0.0))
                states[i] = st
            fx = [(e["from"][0] * W, e["from"][1] * H, e["to"][0] * W, e["to"][1] * H, e["color"],
                   max(e["width"] * W, 1.5)) for e in sc["effects"] if e["start"] <= ts <= e["end"]]
            group = shot_group[si] if si < len(shot_group) else None
            enc.stdin.write(painter.draw(states, group, None, effects=fx))
            if f % 30 == 0:
                cb(f / max(n_out, 1), f"Drawing frame {f}/{n_out}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore")
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed: " + err[-1500:])
    except BaseException:
        enc.kill()
        raise
    shutil.move(out, d / "output.mp4")
    log(f"Done: {n_out} frames at {out_fps:g} fps, {W}x{H}, {len(chars)} characters, in {time.time() - t_start:.0f}s")
    job.update(progress=100, stage="done", message="Finished")



def stick_preview(look, W=360, H=480):
    import io
    import cairo
    from . import explainer
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    c = cairo.Context(surf)
    g = cairo.LinearGradient(0, 0, 0, H)
    g.add_color_stop_rgb(0, 0.95, 0.93, 0.9)
    g.add_color_stop_rgb(1, 0.85, 0.82, 0.78)
    c.set_source(g)
    c.paint()
    explainer.StickCharacter(c, look).draw(dict(x=W / 2, y=H * 0.9, size=H * 0.75, facing=1, arm_l=(-12, 0),
                                                arm_r=(25, 40), leg_l=(-4, 0), leg_r=(4, 0), smile=0.5))
    buf = io.BytesIO()
    surf.write_to_png(buf)
    return buf.getvalue()


def render_explainer(job, d, chars, anim, shots, shot_group, backgrounds, foregrounds, fps, W, H, wav,
                     energy, out_fps, n_out, prog, log, t_start, speech=None):
    if speech is None:
        speech = np.ones(n_out, bool)
    """Explainer look: scene background, simple characters, table/props in front, effects,
    badges/labels/titles, then glow + vignette + grain on every frame."""
    import shutil
    import time
    import cairo
    from . import explainer, media
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    c = cairo.Context(surf)
    sky = cairo.LinearGradient(0, 0, 0, H)
    sky.add_color_stop_rgb(0, 0.55, 0.6, 0.78)
    sky.add_color_stop_rgb(1, 0.9, 0.82, 0.75)
    shot_start = [a / fps for a, _ in shots]
    idx = {ch["id"]: i for i, ch in enumerate(chars)}
    out = d / "output.tmp.mp4"
    enc = media.open_encoder(out, W, H, out_fps, wav, n_out)
    cb = prog.cb("render")
    fsize = H / 720.0
    try:
        for f in range(n_out):
            if job.cancelled:
                raise RuntimeError("Cancelled")
            t = f / out_fps
            si = max(0, int(np.searchsorted(shot_start, t, side="right")) - 1)
            sc = anim.get(si, {"keys": {}, "effects": [], "overlays": []})
            ts = t - shot_start[si]
            group = shot_group[si] if si < len(shot_group) else None
            # slow camera push-in over each shot
            dur = max((shots[si][1] - shots[si][0]) / fps, 0.1)
            zoom = 1.0 + 0.03 * _ease(ts / dur)
            c.save()
            c.translate(W / 2, H / 2)
            c.scale(zoom, zoom)
            c.translate(-W / 2, -H / 2)
            bg = backgrounds.get(group)
            c.set_source_surface(bg, 0, 0) if bg is not None else c.set_source(sky)
            c.paint()
            people = []
            for cid, ks in sc["keys"].items():
                i = idx.get(cid)
                if i is None or not ks:
                    continue
                kv = sample(ks, ts)
                loud = float(energy[min(f, len(energy) - 1)]) if kv["talking"] and speech[min(f, len(speech) - 1)] else 0.0
                talk = min(max(kv["mouth"] * 0.6, loud * 1.1), 1.0) if kv["talking"] and loud > 0 else 0.0
                eyes = kv["eyes"]
                bl = 1.0 if eyes == "closed" else 0.6 if eyes == "squint" else blink(t, i * 1.3)
                h = max(kv["h"], 0.04) * H
                if kv["pose"] in ("walk", "run"):  # procedural stride on top of the keyframes
                    amp, hz = (24, 1.7) if kv["pose"] == "walk" else (42, 2.6)
                    sw = math.sin(t * 2 * math.pi * hz)
                    kv["leg_l"] = (kv["leg_l"][0] + amp * sw, kv["leg_l"][1] - abs(amp * 0.5 * sw))
                    kv["leg_r"] = (kv["leg_r"][0] - amp * sw, kv["leg_r"][1] - abs(amp * 0.5 * sw))
                    kv["arm_l"] = (kv["arm_l"][0] - amp * 0.6 * sw, kv["arm_l"][1])
                    kv["arm_r"] = (kv["arm_r"][0] + amp * 0.6 * sw, kv["arm_r"][1])
                people.append((h, dict(chars[i].get("look", {}), kind=kind_of(chars[i])), dict(
                    x=kv["x"] * W, y=kv["y"] * H + 0.34 * h, size=h, facing=kv["facing"], lean=kv["lean"],
                    head=kv["head"], arm_l=kv["arm_l"], arm_r=kv["arm_r"], leg_l=kv["leg_l"], leg_r=kv["leg_r"],
                    blink=bl, talk=talk, smile=kv["smile"], look_x=0.35 * kv["facing"])))
            for _, look, st in sorted(people, key=lambda p: p[0]):  # far (small) first
                kind = look.get("kind", "human")
                if kind == "crowd":
                    explainer.draw_crowd(c, st, look)
                elif kind != "human":
                    explainer.draw_animal(c, kind, st, look)
                else:
                    explainer.StickCharacter(c, look).draw(st)
            fg = foregrounds.get(group)
            if fg is not None:
                c.set_source_surface(fg, 0, 0)
                c.paint()
            for e in sc["effects"]:
                if e["start"] <= ts <= e["end"]:
                    c.move_to(e["from"][0] * W, e["from"][1] * H)
                    c.line_to(e["to"][0] * W, e["to"][1] * H)
                    c.set_source_rgb(*explainer.rgb(e["color"]))
                    c.set_line_width(max(e["width"] * W, 1.5))
                    c.stroke()
            c.restore()
            for o in sc.get("overlays", []):  # graphics stay fixed on screen (no zoom)
                if not (o["start"] <= ts <= o["end"]):
                    continue
                pop = min((ts - o["start"]) / 0.25, 1.0)  # quick fade-in
                c.push_group()
                if o["type"] == "badge":
                    explainer.badge(c, min(o["x"] * W + 120 * fsize, W - 20 * fsize), o["y"] * H - 25 * fsize,
                                    o["text"], 26 * fsize)
                elif o["type"] == "title":
                    explainer.title(c, o["x"] * W, o["y"] * H, o["text"], 46 * fsize)
                else:
                    explainer.label(c, o["x"] * W, o["y"] * H - 18 * fsize, o["text"], 20 * fsize)
                c.pop_group_to_source()
                c.paint_with_alpha(pop)
            explainer.post_process(surf)
            surf.flush()
            enc.stdin.write(surf.get_data())
            if f % 30 == 0:
                cb(f / max(n_out, 1), f"Drawing frame {f}/{n_out}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore")
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed: " + err[-1500:])
    except BaseException:
        enc.kill()
        raise
    shutil.move(out, d / "output.mp4")
    log(f"Done: {n_out} frames at {out_fps:g} fps, {W}x{H}, {len(chars)} characters, explainer style, "
        f"in {time.time() - t_start:.0f}s")
    job.update(progress=100, stage="done", message="Finished")



def kind_of(c):
    """What to draw: human, crowd, or a simple animal shape (whole words only, so a
    king's 'crown' is not a 'crow')."""
    import re as _re
    k = str(c.get("kind", "")).lower()
    if k in ("bird", "seal", "animal", "crowd", "human"):
        return k
    words = set(_re.findall(r"[a-z]+", (c.get("name", "") + " " + c.get("description", "")[:80]).lower()))

    def has(*ws):
        return any(w in words for w in ws)
    if has("crowd", "audience", "villagers") or ("group" in words and "people" in words):
        return "crowd"
    if has("bird", "robin", "sparrow", "crow", "parrot", "pigeon", "birds", "robins"):
        return "bird"
    if has("seal", "seals", "walrus", "whale", "fish", "dolphin"):
        return "seal"
    if has("dog", "cat", "horse", "cow", "sheep", "elephant", "lion", "goat", "animal"):
        return "animal"
    return "human"
