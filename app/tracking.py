"""Per-frame tracking of the person in the video with MediaPipe (CPU).

Every value is in *image space* (left/right as seen in the video), so the cartoon
mirrors the video exactly. Missing frames are NaN and get filled in later.
"""
import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

from .config import FACE_DET_MODEL, FACE_ID_MODEL, FACE_MODEL, POSE_MODEL

NAMES = [
    "face_ok", "fx", "fy", "fh", "yaw", "pitch", "roll",
    "blink_l", "blink_r", "gaze_x", "gaze_y",
    "brow_up", "brow_down", "brow_tilt",
    "jaw_open", "lip_gap", "mouth_w", "funnel", "pucker", "smile", "stretch",
    "pose_ok",
    "lsh_x", "lsh_y", "lsh_v", "rsh_x", "rsh_y", "rsh_v",
    "lel_x", "lel_y", "lel_v", "rel_x", "rel_y", "rel_v",
    "lwr_x", "lwr_y", "lwr_v", "rwr_x", "rwr_y", "rwr_v",
]
IDX = {n: i for i, n in enumerate(NAMES)}

# Pose landmarks: image-left side is the person's right side.
POSE_JOINTS = {"lsh": 12, "rsh": 11, "lel": 14, "rel": 13, "lwr": 16, "rwr": 15}


def _dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def _face_row(row, lm, bs, w, h):
    g = lambda k: bs.get(k, 0.0)
    top, chin = lm[10], lm[152]
    cheek_l, cheek_r = lm[234], lm[454]
    nose = lm[1]
    eye_l, eye_r = lm[33], lm[263]

    face_h = _dist(top, chin, w, h)
    face_w = _dist(cheek_l, cheek_r, w, h) or 1.0
    row[IDX["face_ok"]] = 1
    row[IDX["fx"]] = (top.x + chin.x) / 2
    row[IDX["fy"]] = (top.y + chin.y) / 2
    row[IDX["fh"]] = face_h / h
    span_x = (cheek_r.x - cheek_l.x) or 1e-6
    row[IDX["yaw"]] = ((nose.x - cheek_l.x) / span_x - 0.5) * 2
    span_y = (chin.y - top.y) or 1e-6
    row[IDX["pitch"]] = (nose.y - top.y) / span_y
    row[IDX["roll"]] = math.atan2((eye_r.y - eye_l.y) * h, (eye_r.x - eye_l.x) * w)

    # Blendshape names use the person's left/right; image-left eye = person's right eye.
    row[IDX["blink_l"]] = g("eyeBlinkRight")
    row[IDX["blink_r"]] = g("eyeBlinkLeft")
    row[IDX["gaze_x"]] = ((g("eyeLookInRight") - g("eyeLookOutRight")) +
                          (g("eyeLookOutLeft") - g("eyeLookInLeft"))) / 2
    row[IDX["gaze_y"]] = ((g("eyeLookDownLeft") + g("eyeLookDownRight")) -
                          (g("eyeLookUpLeft") + g("eyeLookUpRight"))) / 2
    outer_up = (g("browOuterUpLeft") + g("browOuterUpRight")) / 2
    row[IDX["brow_up"]] = max(g("browInnerUp"), outer_up)
    row[IDX["brow_down"]] = (g("browDownLeft") + g("browDownRight")) / 2
    row[IDX["brow_tilt"]] = g("browInnerUp") - outer_up

    row[IDX["jaw_open"]] = g("jawOpen")
    row[IDX["lip_gap"]] = _dist(lm[13], lm[14], w, h) / (face_h or 1.0)
    row[IDX["mouth_w"]] = _dist(lm[61], lm[291], w, h) / face_w
    row[IDX["funnel"]] = g("mouthFunnel")
    row[IDX["pucker"]] = g("mouthPucker")
    row[IDX["smile"]] = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
    row[IDX["stretch"]] = (g("mouthStretchLeft") + g("mouthStretchRight")) / 2


def _pose_row(row, lm):
    row[IDX["pose_ok"]] = 1
    for key, i in POSE_JOINTS.items():
        p = lm[i]
        row[IDX[key + "_x"]] = p.x
        row[IDX[key + "_y"]] = p.y
        row[IDX[key + "_v"]] = p.visibility if p.visibility is not None else 1.0


def _sample_colors(frame_bgr, lm, w, h, skin, shirt, hair, chin):
    """Collects skin (cheeks), shirt (below chin), hair (above forehead) and chin colours."""
    def patch(cx, cy, rad):
        x0, x1 = int(max(cx - rad, 0)), int(min(cx + rad, w - 1))
        y0, y1 = int(max(cy - rad, 0)), int(min(cy + rad, h - 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return np.median(frame_bgr[y0:y1, x0:x1].reshape(-1, 3), axis=0)

    face_h = _dist(lm[10], lm[152], w, h)
    rad = max(face_h * 0.05, 2)
    for i in (50, 280):
        c = patch(lm[i].x * w, lm[i].y * h, rad)
        if c is not None:
            skin.append(c)
    ch = lm[152]
    sy = ch.y * h + face_h * 0.75
    if sy < h - rad:
        c = patch(ch.x * w, sy, rad * 2)
        if c is not None:
            shirt.append(c)
    top = lm[10]
    hy = top.y * h - face_h * 0.12
    if hy > rad:
        c = patch(top.x * w, hy, rad)
        if c is not None:
            hair.append(c)
    c = patch(lm[199].x * w, lm[199].y * h, rad)  # chin, where a beard would be
    if c is not None:
        chin.append(c)


def _bgr_hex(c):
    b, g, r = (int(v) for v in c)
    return f"#{r:02x}{g:02x}{b:02x}"


def _frame_signature(frame):
    """Small colour histogram used for cut detection and grouping similar shots."""
    small = cv2.resize(frame, (96, 54))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return hist / (hist.sum() or 1.0)


def shot_signature(frame):
    return _frame_signature(frame)


class _Tracklet:
    """One face followed inside one shot. Tracklets are grouped into people afterwards."""

    def __init__(self, tid, shot, start):
        self.tid, self.shot, self.start, self.end = tid, shot, start, start
        self.cx = self.cy = self.fh = 0.0
        self.frames = 0
        self.lm_frames = 0  # frames where the face could actually be read (not a hand / blur)
        self.score_sum = 0.0  # face-finder confidence, summed
        self.alt = (-1.0, start)  # a second good frame (other moment) for the casting photos
        self.shirt_col = None  # running clothes colour just below the chin (stops identity swaps)
        self.embs = []
        self.best = (-1.0, start, None)  # (score, frame index, crop box normalized)
        self.skin, self.shirt, self.hair, self.chin = [], [], [], []

    def emb(self):
        if not self.embs:
            return None
        e = np.mean(self.embs, axis=0)
        return e / (np.linalg.norm(e) or 1.0)


EMB_AT = {1, 6, 20, 50, 120, 300}  # tracklet ages (frames) at which we take a face fingerprint


def _face_box_row(lm, w, h):
    """MediaPipe landmarks -> the 5-point face row SFace expects (YuNet format)."""
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    x0, y0 = min(xs), min(ys)
    pt = lambda *ids: (sum(lm[i].x for i in ids) / len(ids) * w, sum(lm[i].y for i in ids) / len(ids) * h)
    re, le = pt(33, 133), pt(362, 263)   # person's right eye is on the image left
    nose, rm, lmo = pt(1), pt(61), pt(291)
    return np.array([[x0, y0, max(xs) - x0, max(ys) - y0, *re, *le, *nose, *rm, *lmo, 0.99]], np.float32)


class _LM:
    """Minimal landmark (x, y normalized to the full frame), like MediaPipe's."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _crop(img, x0, y0, x1, y1, max_side):
    """Crops (clamped, padded with black) and scales so the long side is max_side.
    Returns (crop, (x0, y0, scale)) to map crop pixels back to the image."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    pad = [max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W)]
    sub = img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
    if any(pad):
        sub = cv2.copyMakeBorder(sub, *pad, cv2.BORDER_CONSTANT, value=0)
    s = max_side / max(sub.shape[:2])
    sub = cv2.resize(sub, (max(int(sub.shape[1] * s), 1), max(int(sub.shape[0] * s), 1)))
    return sub, (x0, y0, s)


def _yunet_row(b, ds, w, h):
    """Face row from YuNet alone: position, size and head angle. Eye/mouth values stay
    NaN so they are interpolated from frames where the expression was readable."""
    row = np.full(len(NAMES), np.nan, dtype=np.float32)
    x, y, bw, bh = (float(v) / ds for v in b[:4])
    eyes = sorted([(float(b[4]) / ds, float(b[5]) / ds), (float(b[6]) / ds, float(b[7]) / ds)])
    nose_x, nose_y = float(b[8]) / ds, float(b[9]) / ds
    row[IDX["face_ok"]] = 1
    row[IDX["fx"]] = (x + bw / 2) / w
    row[IDX["fy"]] = (y + bh / 2) / h
    row[IDX["fh"]] = bh * 1.05 / h
    row[IDX["yaw"]] = ((nose_x - x) / max(bw, 1) - 0.5) * 2
    row[IDX["pitch"]] = (nose_y - y) / max(bh, 1)
    (lx, ly), (rx, ry) = eyes
    row[IDX["roll"]] = math.atan2(ry - ly, rx - lx)
    return row


def track_video(path, step=1, progress=None, max_visible=4, max_characters=8):
    """Finds every person in the video (across cuts) and tracks them on every frame.

    Faces are found with YuNet (good at small faces), then each face is zoomed in and
    read by MediaPipe (expression, eyes, lips) and the body pose is read around it.
    Returns dict(fps, w, h, feats[N, P, F], people[...], shots[[start, end), ...]).
    One column of feats = one person (character) for the whole video.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ds = min(1.0, 960 / max(w, h))  # detection size: long side at most 960 px
    dw, dh = int(w * ds) // 2 * 2, int(h * ds) // 2 * 2

    detector = cv2.FaceDetectorYN.create(str(FACE_DET_MODEL), "", (dw, dh), 0.6, 0.3, 50)
    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3))
    pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))
    recog = cv2.FaceRecognizerSF.create(str(FACE_ID_MODEL), "") if FACE_ID_MODEL.exists() else None

    tracklets, active = [], []
    per_frame = []          # per frame: list of (tracklet id, feature row)
    cuts = [0]
    prev_sig, prev_gray, last_cut = None, None, 0
    recent = []             # recent pixel differences, for adaptive cut detection
    color_every = max(int(fps), 1)
    tracked_count = 0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ---- cut detection: colour histogram jump OR pixel change far above recent motion
            sig = _frame_signature(frame)
            gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_sig is not None:
                hdiff = 0.5 * float(np.abs(sig - prev_sig).sum())
                mad = float(np.mean(np.abs(gray - prev_gray)))
                base = float(np.median(recent)) if recent else mad
                is_cut = hdiff > 0.3 or (mad > 12 and mad > 3.5 * base + 4)
                if is_cut and i - last_cut >= fps * 0.4:
                    cuts.append(i)
                    last_cut = i
                    active = []  # a cut ends every tracklet
                    recent = []
                else:
                    recent = (recent + [mad])[-int(fps):]
            prev_sig, prev_gray = sig, gray
            shot = len(cuts) - 1

            if i % step:
                per_frame.append(None)
                i += 1
                continue

            det_img = cv2.resize(frame, (dw, dh)) if ds < 1 else frame
            _, boxes = detector.detect(det_img)
            boxes = [] if boxes is None else sorted(boxes, key=lambda b: -b[2] * b[3])
            # the same face found twice (or a hand next to the face): keep only the bigger box
            kept = []
            for b in boxes:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                # duplicate only if its centre is INSIDE a bigger face box (two people
                # leaning together have their faces side by side, not inside each other)
                if any(abs(cx - (k[0] + k[2] / 2)) < k[2] * 0.45 and abs(cy - (k[1] + k[3] / 2)) < k[3] * 0.45
                       for k in kept):
                    continue
                kept.append(b)
            boxes = kept[:max_visible]
            do_pose = tracked_count % 2 == 0
            tracked_count += 1

            faces = []
            for b in boxes:
                bx, by, bw, bh = (float(v) / ds for v in b[:4])
                if bw < 14 or bh < 14:
                    continue
                cx, cy, side = bx + bw / 2, by + bh / 2, max(bw, bh) * 1.9
                crop, (ox, oy, sc) = _crop(frame, cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, 256)
                res = face_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if not res.face_landmarks:
                    # sunglasses / mic in front of the mouth: keep the person on screen using
                    # YuNet's box and 5 points; eyes and mouth get filled in from nearby frames
                    faces.append((_yunet_row(b, ds, w, h), None, b))
                    continue
                ch, cw = crop.shape[:2]
                lm = [_LM((ox + p.x * cw / sc) / w, (oy + p.y * ch / sc) / h) for p in res.face_landmarks[0]]
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
                row = np.full(len(NAMES), np.nan, dtype=np.float32)
                _face_row(row, lm, bs, w, h)
                faces.append((row, lm, b))

            # ---- match faces to tracklets of this shot (distance in face heights)
            active = [t for t in active if i - t.end <= fps]
            # clothes colour below each face: people whispering lean their heads together,
            # but their shirts stay where they are - this stops the tracker swapping them
            cols = []
            for r, _, _ in faces:
                fh_px = r[IDX["fh"]] * h
                cx, cy = r[IDX["fx"]] * w, r[IDX["fy"]] * h + fh_px * 1.05
                rad = max(int(fh_px * 0.22), 2)
                x0, x1 = int(max(cx - rad, 0)), int(min(cx + rad, w - 1))
                y0, y1 = int(max(cy - rad, 0)), int(min(cy + rad, h - 1))
                cols.append(frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) if x1 > x0 and y1 > y0 else None)

            def cost(fi, t):
                r = faces[fi][0]
                c = math.hypot((r[IDX["fx"]] - t.cx) * w / h, r[IDX["fy"]] - t.cy) / max(t.fh, 0.05)
                if cols[fi] is not None and t.shirt_col is not None:
                    c += float(np.linalg.norm(cols[fi] - t.shirt_col)) / 70.0
                return c
            pairs = sorted((cost(fi, t), fi, ti) for fi in range(len(faces)) for ti, t in enumerate(active))
            assigned, used = {}, set()
            for c, fi, ti in pairs:
                if fi in assigned or ti in used or c > 1.9:
                    continue
                assigned[fi] = active[ti]
                used.add(ti)

            # Two faces meeting in one spot (whispering, leaning, one head hiding the
            # other): afterwards we cannot know who is who, so cut the track there.
            # The pieces are grouped into people later (by DeepSeek / fingerprints).
            def near(t, fi):
                r = faces[fi][0]
                return math.hypot((r[IDX["fx"]] - t.cx) * w / h, r[IDX["fy"]] - t.cy) < max(t.fh, 0.05) * 0.7
            lost = [t for ti, t in enumerate(active) if ti not in used and i - t.end <= 2]
            newborn = [fi for fi in range(len(faces)) if fi not in assigned]
            for fi, t in list(assigned.items()):
                if t.frames < 5:
                    continue
                if any(near(u, fi) for u in lost) or any(
                        math.hypot((faces[nb][0][IDX["fx"]] - faces[fi][0][IDX["fx"]]) * w / h,
                                   faces[nb][0][IDX["fy"]] - faces[fi][0][IDX["fy"]]) < max(t.fh, 0.05) * 0.7
                        for nb in newborn):
                    active.remove(t)
                    t2 = _Tracklet(len(tracklets), shot, i)
                    tracklets.append(t2)
                    active.append(t2)
                    assigned[fi] = t2
            for fi in range(len(faces)):
                if fi not in assigned:
                    t = _Tracklet(len(tracklets), shot, i)
                    tracklets.append(t)
                    active.append(t)
                    assigned[fi] = t

            matched = {}
            for fi, t in assigned.items():
                row, lm, b = faces[fi]
                # every 10 frames: does this face still match the track's fingerprint?
                # if clearly not, the tracker jumped to someone else -> split the track
                if recog is not None and t.frames >= 10 and t.frames % 10 == 0 and t.embs and row[IDX["fh"]] * h > 40:
                    try:
                        e = recog.feature(recog.alignCrop(det_img, b.reshape(1, -1))).flatten()
                        e = e / (np.linalg.norm(e) or 1.0)
                        if float(np.dot(e, t.emb())) < 0.2:
                            active.remove(t)
                            t = _Tracklet(len(tracklets), shot, i)
                            tracklets.append(t)
                            active.append(t)
                            t.embs.append(e)
                    except cv2.error:
                        pass
                if cols[fi] is not None:
                    t.shirt_col = cols[fi] if t.shirt_col is None else t.shirt_col * 0.9 + cols[fi] * 0.1
                t.cx, t.cy, t.fh = float(row[IDX["fx"]]), float(row[IDX["fy"]]), float(row[IDX["fh"]])
                t.end = i
                t.frames += 1
                if lm is not None:
                    t.lm_frames += 1
                t.score_sum += float(b[14]) if len(b) > 14 else 0.9
                fh_px = t.fh * h
                if recog is not None and t.frames in EMB_AT and fh_px > 40:
                    try:  # YuNet's 5 points are exactly what SFace needs
                        t.embs.append(recog.feature(recog.alignCrop(det_img, b.reshape(1, -1))).flatten())
                    except cv2.error:
                        pass
                blink = float(np.nan_to_num(row[IDX["blink_l"]]))
                score = t.fh * (1.2 - min(abs(float(row[IDX["yaw"]])), 1.0)) * (1.0 - blink * 0.5)
                # the casting photo must show this person alone: another face close by
                # (leaning in, whispering) makes the frame a bad photo
                near = min([math.hypot((r2[IDX["fx"]] - row[IDX["fx"]]) * w / h, r2[IDX["fy"]] - row[IDX["fy"]])
                            / max(float(row[IDX["fh"]]), 0.05) for r2, _, _ in faces if r2 is not row] or [9.0])
                if near < 1.6:
                    score *= 0.25
                if lm is None and t.best[2] is not None:
                    score = -1.0  # the photo for DeepSeek must show a readable face
                elif lm is None:
                    score *= 0.1
                if score > t.best[0]:
                    fw = t.fh * h / w  # face height in units of frame width
                    box = (max(t.cx - fw * 0.95, 0), max(t.cy - t.fh * 0.9, 0),
                           min(t.cx + fw * 0.95, 1), min(t.cy + t.fh * 1.9, 1))
                    if abs(i - t.best[1]) > fps * 1.5:
                        t.alt = (t.best[0], t.best[1])
                    t.best = (score, i, box)
                if lm is not None and i % color_every == 0 and len(t.skin) < 20:
                    _sample_colors(frame, lm, w, h, t.skin, t.shirt, t.hair, t.chin)

                row[IDX["pose_ok"]] = np.nan  # no pose this frame: interpolated later
                if do_pose:
                    row[IDX["pose_ok"]] = 0
                    px0, px1 = t.cx * w - fh_px * 2.6, t.cx * w + fh_px * 2.6
                    py0, py1 = t.cy * h - fh_px * 1.3, t.cy * h + fh_px * 5.0
                    crop, (ox, oy, sc) = _crop(frame, px0, py0, px1, py1, 320)
                    pr = pose_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    if pr.pose_landmarks:
                        ch, cw = crop.shape[:2]
                        plm = [_LM((ox + p.x * cw / sc) / w, (oy + p.y * ch / sc) / h,
                                   p.visibility if p.visibility is not None else 1.0) for p in pr.pose_landmarks[0]]
                        nose = plm[0]
                        # make sure the body belongs to this face, not a neighbour
                        if math.hypot((nose.x - t.cx) * w, (nose.y - t.cy) * h) < fh_px * 0.8:
                            _pose_row(row, plm)
                matched[t.tid] = row
            per_frame.append(list(matched.items()))
            i += 1
            if progress and i % 15 == 0:
                progress(min(i / total, 1.0), f"Tracking frame {i}/{total} - {len(cuts)} shots, {len(tracklets)} face tracks")
    finally:
        cap.release()
        face_lm.close()
        pose_lm.close()
    if i < total * 0.9:  # a broken read must never pass as a finished video
        raise RuntimeError(f"Could only read {i} of {total} frames. Close other apps and press Resume.")

    n = len(per_frame)
    min_frames = max(3, int(fps * 0.3 / step))
    # a real face: readable in some frames, or found with high confidence (a mic in front
    # of the mouth stops the expression reader, not the face finder). Hands/objects that
    # slip through are removed by DeepSeek's casting.
    good = [t for t in tracklets if t.frames >= min_frames and
            (t.lm_frames >= max(3, 0.3 * t.frames) or (t.score_sum / t.frames >= 0.8 and t.lm_frames >= 1))]
    rows_by = {t.tid: np.full((t.end - t.start + 1, len(NAMES)), np.nan, np.float32) for t in good}
    tracked = np.array([e is not None for e in per_frame], bool)
    for fi, entries in enumerate(per_frame):
        for tid, row in entries or []:
            if tid in rows_by:
                rows_by[tid][fi - tracklets[tid].start] = row
    tl_info = []
    for t in good:
        e = t.emb()
        tl_info.append({"tid": t.tid, "shot": t.shot, "start": t.start, "end": t.end, "frames": t.frames,
                        "best_frame": t.best[1], "alt_frame": t.alt[1], "box": t.best[2], "colors": _colors(t),
                        "emb": e.tolist() if e is not None else None})
    # first guess at who is who: face fingerprints (DeepSeek can regroup later)
    groups = [p["tracklets"] for p in _group_people(tracklets, fps, step, max_characters)]
    feats, people = assemble(groups, tl_info, rows_by, tracked, n)
    shots = [[a, b] for a, b in zip(cuts, cuts[1:] + [n])] if n else []
    return {"fps": fps, "w": w, "h": h, "feats": feats, "people": people, "shots": shots,
            "tracklets": tl_info, "rows_by": rows_by, "tracked": tracked}


def assemble(groups, tl_info, rows_by, tracked, n, notes=None):
    """groups: list of lists of tracklet ids (one list = one person).
    Returns (feats[N, P, F], people[...]) - one column per person for the whole video."""
    info = {t["tid"]: t for t in tl_info}
    groups = [[tid for tid in g if tid in info and tid in rows_by] for g in groups]
    groups = [g for g in groups if g]
    groups.sort(key=lambda g: -sum(info[t]["frames"] for t in g))
    P = len(groups)
    feats = np.full((n, P, len(NAMES)), np.nan, np.float32)
    feats[tracked, :, IDX["face_ok"]] = 0
    feats[tracked, :, IDX["pose_ok"]] = 0
    people = []
    for pi, g in enumerate(groups):
        for tid in g:
            t = info[tid]
            rows = rows_by[tid]
            seg = feats[t["start"]:t["end"] + 1, pi]
            m = rows[:, IDX["face_ok"]] == 1
            m &= seg[:, IDX["face_ok"]] != 1
            seg[m] = rows[m]
        best = max(g, key=lambda tid: info[tid]["frames"])
        colors = {}
        for tid in g:
            for k, v in info[tid]["colors"].items():
                colors.setdefault(k, v)
        people.append({"id": pi, "seen": int(sum(info[t]["frames"] for t in g)),
                       "best_frame": info[best]["best_frame"], "box": info[best]["box"], "colors": colors,
                       "shots": sorted({info[t]["shot"] for t in g}), "tracklets": list(g),
                       "note": (notes or {}).get(pi, "")})
    return feats, people


def _group_people(tracklets, fps, step, max_characters):
    """Groups tracklets into people using face fingerprints.

    Two tracklets visible at the same time are always different people.
    """
    min_frames = max(3, int(fps * 0.3 / step))
    # a real face is readable (landmarks found) in a good share of its frames;
    # hands, mics and blurry blobs that fooled the face finder are not
    tl = sorted((t for t in tracklets if t.frames >= min_frames and t.lm_frames >= max(3, 0.3 * t.frames)),
                key=lambda t: -t.frames)
    groups = []  # each: dict(members, emb, frames)

    def overlaps(g, t):
        return any(m.start <= t.end and t.start <= m.end for m in g["members"])

    for t in tl:
        e = t.emb()
        best, best_sim = None, -1.0
        for g in groups:
            if overlaps(g, t):
                continue
            if e is not None and g["emb"] is not None:
                samples = [m.emb() for m in g["members"] if m.emb() is not None]
                sim = max([float(np.dot(e, g["emb"]))] + [float(np.dot(e, x)) for x in samples])
            else:  # no fingerprint: fall back to screen position in the previous shot
                last = max(g["members"], key=lambda m: m.end)
                sim = 0.5 - math.hypot(t.cx - last.cx, t.cy - last.cy) / max(t.fh, 0.05) * 0.1
            if sim > best_sim:
                best, best_sim = g, sim
        if best is not None and best_sim > 0.42:
            best["members"].append(t)
            best["frames"] += t.frames
            embs = [m.emb() for m in best["members"] if m.emb() is not None]
            if embs:
                m = np.mean(embs, axis=0)
                best["emb"] = m / (np.linalg.norm(m) or 1.0)
        else:
            groups.append({"members": [t], "emb": e, "frames": t.frames})

    groups.sort(key=lambda g: -g["frames"])
    keep, extra = groups[:max_characters], groups[max_characters:]
    for g in extra:  # too many characters: give minor ones to the most similar main one
        if g["emb"] is None:
            continue
        cands = [k for k in keep if k["emb"] is not None and not any(overlaps(k, m) for m in g["members"])]
        if cands:
            k = max(cands, key=lambda k: float(np.dot(k["emb"], g["emb"])))
            k["members"].extend(g["members"])
            k["frames"] += g["frames"]

    people = []
    for pid, g in enumerate(keep):
        best = max(g["members"], key=lambda m: m.best[0])
        pool = lambda attr: [c for m in g["members"] for c in getattr(m, attr)]
        tmp = _Tracklet(-1, 0, 0)
        tmp.skin, tmp.shirt, tmp.hair, tmp.chin = pool("skin"), pool("shirt"), pool("hair"), pool("chin")
        people.append({"id": pid, "seen": g["frames"], "best_frame": best.best[1], "box": best.best[2],
                       "colors": _colors(tmp), "shots": sorted({m.shot for m in g["members"]}),
                       "tracklets": [m.tid for m in g["members"]]})
    return people


def _colors(t):
    out = {}
    if t.skin:
        out["skin"] = _bgr_hex(np.median(t.skin, axis=0))
    if t.shirt:
        out["shirt"] = _bgr_hex(np.median(t.shirt, axis=0))
    if t.hair:
        out["hair"] = _bgr_hex(np.median(t.hair, axis=0))
    if t.skin and t.chin:
        sk, cn = np.median(t.skin, axis=0), np.median(t.chin, axis=0)
        if cn.mean() < sk.mean() * 0.62:  # chin much darker than cheeks -> beard
            out["beard"] = _bgr_hex(cn)
    return out


def read_frames(path, indices):
    """Returns {index: BGR frame} for the requested frame numbers."""
    cap = cv2.VideoCapture(str(path))
    out = {}
    try:
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                out[idx] = frame
    finally:
        cap.release()
    return out
