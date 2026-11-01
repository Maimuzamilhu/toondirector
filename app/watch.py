"""Lets DeepSeek 'watch' the original video at 10 frames per second.

Each second of video becomes one contact sheet: 10 frames in a grid, each with a
timestamp and every tracked person's face boxed and labelled (P1, P2, ...), so the
model can tell who is talking, laughing, pointing or looking at whom - and when.
"""
import cv2
import numpy as np

from .tracking import IDX

COLORS = [(66, 135, 245), (60, 200, 90), (245, 90, 66), (220, 60, 220),
          (40, 200, 220), (240, 200, 40), (160, 100, 240), (250, 140, 180)]  # BGR


def label_frame(img, feats_row, pids, w, h, scale=1.0):
    """Draws a box + 'P<n>' on each visible tracked face."""
    out = img.copy()
    for pid in pids:
        r = feats_row[pid]
        if r[IDX["face_ok"]] != 1:
            continue
        cx, cy, fh = r[IDX["fx"]] * w * scale, r[IDX["fy"]] * h * scale, r[IDX["fh"]] * h * scale
        x0, y0 = int(cx - fh * 0.45), int(cy - fh * 0.6)
        x1, y1 = int(cx + fh * 0.45), int(cy + fh * 0.6)
        col = COLORS[pid % len(COLORS)]
        th = max(1, int(round(2 * scale)))
        cv2.rectangle(out, (x0, y0), (x1, y1), col, th)
        cv2.putText(out, f"P{pid + 1}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * max(scale, 0.8), col, th, cv2.LINE_AA)
    return out


def contact_sheets(video, feats, pids, start, end, fps, rate=10, cols=5, thumb=200):
    """Contact sheets (one per second) for frames [start, end). Returns list of (t0, image)."""
    step = max(fps / rate, 1.0)
    if rate > 10:  # more frames per sheet: more columns, smaller thumbnails
        cols = 6 if rate <= 30 else 8
        thumb = 170 if rate <= 30 else 140
    idx = [int(round(start + k * step)) for k in range(int((end - start) / step) + 1)]
    idx = [i for i in idx if start <= i < end]
    if not idx:
        return []
    want = set(idx)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = {}
    i = start
    while i < end:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
        i += 1
    cap.release()
    if not frames:
        return []
    h, w = next(iter(frames.values())).shape[:2]
    s = thumb / max(w, h)
    tw, th = int(w * s), int(h * s)
    per = rate  # frames per sheet = one second
    sheets = []
    for k in range(0, len(idx), per):
        chunk = [j for j in idx[k:k + per] if j in frames]
        if not chunk:
            continue
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (th + 18), cols * tw, 3), 20, np.uint8)
        for m, j in enumerate(chunk):
            small = cv2.resize(frames[j], (tw, th), interpolation=cv2.INTER_AREA)
            small = label_frame(small, feats[j], pids, w, h, s)
            y, x = (m // cols) * (th + 18), (m % cols) * tw
            sheet[y + 18:y + 18 + th, x:x + tw] = small
            cv2.putText(sheet, f"{(j - start) / fps:.2f}s" if rate > 10 else f"{(j - start) / fps:.1f}s", (x + 4, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (230, 230, 230), 1, cv2.LINE_AA)
        sheets.append(((chunk[0] - start) / fps, sheet))
    return sheets


def mark_person(crop, box_in_crop):
    """Circles the target person in their photo so the artist draws the right one."""
    out = crop.copy()
    x0, y0, x1, y1 = (int(v) for v in box_in_crop)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.ellipse(out, (cx, cy), (max((x1 - x0) // 2, 4), max((y1 - y0) // 2, 4)), 0, 0, 360, (0, 255, 255), 2)
    return out
