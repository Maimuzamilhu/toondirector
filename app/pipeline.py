"""Runs all stages for one job. Each stage caches its output in the job folder,
so a re-run (or a crash halfway through a long video) resumes where it stopped.

Stages (like Claude Design's cartoon rebuild):
  prepare -> audio -> transcribe -> track (people + cuts) -> lipsync
  -> design (DeepSeek draws characters + backgrounds, then reviews) -> direct -> render
"""
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import providers
from . import animate, choreo, config, director, lipsync, media
from .character import Painter
from .config import rhubarb_exe
from .tracking import IDX, read_frames, shot_signature, track_video
from .tracking import assemble as assemble_people
from .transcribe import captions, transcribe

# bump when tracking.json/npz change shape, so old saved jobs are tracked again
TRACK_VERSION = 11

class Progress:
    def __init__(self, job):
        self.job = job
        self.done = 0
        self.table = [(n, w) for n, _, w in job.stages]

    def stage(self, name):
        names = [n for n, _ in self.table]
        self.done = sum(w for _, w in self.table[:names.index(name)])
        self.job.update(stage=name, progress=self.done)

    def cb(self, name):
        weight = dict(self.table)[name]

        def f(frac, msg=None):
            if self.job.cancelled:
                raise RuntimeError("Cancelled")
            self.job.update(progress=round(self.done + weight * frac, 1), message=msg or self.job.message)
        return f


def _even(x):
    return int(x) // 2 * 2


def _load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def neutral_state(W, H, r=None):
    """A front-facing rest pose, used for previews and the review pass."""
    r = r or H * 0.2
    hx, hy = W / 2, H * 0.36
    s = dict(r=r, hx=hx, hy=hy, roll=0, yaw=0, pitch=0, blink_l=0, blink_r=0, gaze_x=0, gaze_y=0,
             brow_up=0.1, brow_down=0, brow_tilt=0, open=0.25, width=0.2, round=0, smile=0.35, blush=0,
             eye_smile=0, eye_wide=0, alpha=1)
    for side, sign in (("l", -1), ("r", 1)):
        s[f"{side}sh_x"], s[f"{side}sh_y"] = hx + sign * r * 1.15, hy + r * 1.65
        s[f"{side}el_x"], s[f"{side}el_y"] = hx + sign * r * 1.4, hy + r * 2.95
        s[f"{side}wr_x"], s[f"{side}wr_y"] = hx + sign * r * 1.3, hy + r * 4.1
    return s


def _style_for(person, description, log):
    style = director.design_character(description, person.get("colors", {}) if person else {}, log)
    return style


def _apply_colors(style, colors):
    st = dict(style)
    if colors.get("skin"):
        st["skin"] = colors["skin"]
    if colors.get("iris"):
        st["eye_color"] = colors["iris"]
    if colors.get("brow"):
        st["brow_color"] = colors["brow"]
    return st


def render_preview(style, parts, size=640, bg_svg=None):
    """PNG bytes of one puppet in the neutral pose (used for DeepSeek's review)."""
    from .sprites import background_surface, build_rig
    W = H = size
    rig = build_rig(parts, zoom=2.0)
    bgs = {0: background_surface(bg_svg, W, H)} if bg_svg else {}
    p = Painter(W, H, [(style, rig)], bgs, bg_style={"bg_top": "#f2efe8", "bg_bottom": "#dcd6cc"})
    st = neutral_state(W, H, r=H * 0.17)
    st["hy"] = H * 0.3
    for k in list(st):
        if k.endswith("_y") and k[:3] in ("lsh", "rsh", "lel", "rel", "lwr", "rwr"):
            st[k] -= H * 0.06
    p.draw([st], 0 if bg_svg else None)
    import io
    buf = io.BytesIO()
    p.surface.write_to_png(buf)
    return buf.getvalue()


def _track_conflicts(tl, rows_by):
    """Pairs of tracks visible in the same frames at clearly different places
    (= two different people)."""
    out = set()
    for i, a in enumerate(tl):
        for j in range(i + 1, len(tl)):
            b = tl[j]
            lo, hi = max(a["start"], b["start"]), min(a["end"], b["end"])
            if lo > hi:
                continue
            ra = rows_by[a["tid"]][lo - a["start"]:hi - a["start"] + 1]
            rb = rows_by[b["tid"]][lo - b["start"]:hi - b["start"] + 1]
            both = (ra[:, IDX["face_ok"]] == 1) & (rb[:, IDX["face_ok"]] == 1)
            if not both.any():
                continue
            dist = np.hypot(ra[both, IDX["fx"]] - rb[both, IDX["fx"]], ra[both, IDX["fy"]] - rb[both, IDX["fy"]])
            if int((dist > np.maximum(ra[both, IDX["fh"]], rb[both, IDX["fh"]]) * 0.8).sum()) > 3:
                out.add((i, j))
    return out


def _track_crop(img, t, row):
    from . import watch
    hh, ww = img.shape[:2]
    x0, y0, x1, y1 = t["box"]
    X0, Y0, X1, Y1 = int(x0 * ww), int(y0 * hh), int(x1 * ww), int(y1 * hh)
    crop = img[Y0:Y1, X0:X1]
    if not crop.size:
        return None
    if row is not None and row[IDX["face_ok"]] == 1:
        cx, cy, fh = row[IDX["fx"]] * ww - X0, row[IDX["fy"]] * hh - Y0, row[IDX["fh"]] * hh
        crop = watch.mark_person(crop, (cx - fh * 0.5, cy - fh * 0.65, cx + fh * 0.5, cy + fh * 0.65))
    return crop


def cast_stage(d, video, feats, people, track, rows_by, tracked, opts, log):
    """Who is who. DeepSeek looks at a photo of every face track and groups them into
    real people (faces, hair, glasses and - above all - clothes), drops tracks that are
    not people, and writes a note per person on what makes them different.
    Tracks seen side by side at the same time are never put together.
    Returns (feats, people, notes). Without DeepSeek, the face-fingerprint guess is used."""
    from . import design
    ddir = d / "design"
    ddir.mkdir(exist_ok=True)
    tl = track.get("tracklets") or []
    fps = track["fps"]
    n = feats.shape[0]
    max_people = int(opts.get("max_people") or 8)
    use_ai = opts.get("art", "ai") in ("ai", "explainer") and providers.configured() and tl and rows_by

    gpath = ddir / "cast_groups.json"
    saved = _load_json(gpath)
    if use_ai and saved is None:
        cand = sorted(tl, key=lambda t: -t["frames"])[:40]  # the 40 longest tracks go to DeepSeek
        frames = read_frames(video, [t["best_frame"] for t in cand] + [t.get("alt_frame", t["best_frame"]) for t in cand])
        crops, keep = [], []
        for t in cand:
            pics = []
            for fi in dict.fromkeys([t["best_frame"], t.get("alt_frame", t["best_frame"])]):
                img = frames.get(fi)
                if img is None or not t.get("box") or not (t["start"] <= fi <= t["end"]):
                    continue
                row = rows_by[t["tid"]][fi - t["start"]]
                if row[IDX["face_ok"]] != 1:
                    continue
                # crop around where the face is in THAT frame
                hh, ww = img.shape[:2]
                fw = row[IDX["fh"]] * hh / ww
                box = (max(row[IDX["fx"]] - fw * 0.95, 0), max(row[IDX["fy"]] - row[IDX["fh"]] * 0.9, 0),
                       min(row[IDX["fx"]] + fw * 0.95, 1), min(row[IDX["fy"]] + row[IDX["fh"]] * 1.9, 1))
                c = _track_crop(img, {"box": box}, row)
                if c is not None:
                    pics.append(cv2.resize(c, (int(c.shape[1] * 300 / c.shape[0]), 300)))
            if pics:  # two moments of the same track side by side
                crops.append(np.hstack(pics) if len(pics) > 1 else pics[0])
                keep.append(t)
        conflicts = _track_conflicts(keep, rows_by)
        try:
            log(f"DeepSeek is casting: grouping {len(keep)} face tracks into real people...")
            groups_i, notes_l, not_person = design.group_tracks(crops, sorted((i + 1, j + 1) for i, j in conflicts))
        except Exception as e:
            log(f"Casting failed ({e}); using face fingerprints instead.")
            groups_i, notes_l, not_person = None, [], set()
        if groups_i is not None:
            # enforce the hard rule: no two side-by-side tracks in one person
            groups, notes = [], []
            for g, note in zip(groups_i, notes_l):
                bins = []
                for i in g:
                    for b_ in bins:
                        if all((min(i, k), max(i, k)) not in conflicts for k in b_):
                            b_.append(i)
                            break
                    else:
                        bins.append([i])
                for b_ in bins:
                    groups.append([keep[i]["tid"] for i in b_])
                    notes.append(note if b_ is bins[0] else "")
            assigned = {tid for g in groups for tid in g} | {keep[i]["tid"] for i in not_person}
            log(f"Casting: {len(keep)} tracks -> {len(groups)} people"
                + (f", {len(not_person)} not a person (hand/object/blur)" if not_person else ""))
            # shorter tracks DeepSeek did not see: join the person with the closest face fingerprint
            info = {t["tid"]: t for t in tl}
            for t in tl:
                if t["tid"] in assigned or not t.get("emb"):
                    continue
                e = np.array(t["emb"])
                best, best_sim = None, 0.35
                for gi, g in enumerate(groups):
                    sims = [float(np.dot(e, np.array(info[m]["emb"]))) for m in g if info[m].get("emb")]
                    if sims and max(sims) > best_sim and not any(
                            max(info[m]["start"], t["start"]) <= min(info[m]["end"], t["end"]) for m in g):
                        best, best_sim = gi, max(sims)
                if best is not None:
                    groups[best].append(t["tid"])
            saved = {"groups": groups, "notes": notes}
            _save_json(gpath, saved)
    if saved:
        groups, notes = saved["groups"], saved["notes"]
        order = sorted(range(len(groups)), key=lambda g: -sum(
            next((t["frames"] for t in tl if t["tid"] == tid), 0) for tid in groups[g]))[:max_people]
        groups = [groups[g] for g in order]
        notes = [notes[g] if g < len(notes) else "" for g in order]
        feats, people = assemble_people(groups, tl, rows_by, tracked, n)
    else:
        notes = [""] * len(people)

    # final check: DeepSeek sees one photo of EVERY character side by side and says which
    # are the same real person; those are merged into one character
    if use_ai and saved and len(people) > 1:
        dpath = ddir / "dedupe.json"
        same = _load_json(dpath)
        if same is None:
            frames = read_frames(video, [p["best_frame"] for p in people])
            crops = []
            for p in people:
                img = frames.get(p["best_frame"])
                crop = _track_crop(img, {"box": p["box"]}, feats[p["best_frame"], p["id"]]) \
                    if img is not None and p.get("box") else None
                crops.append(crop if crop is not None else np.zeros((64, 64, 3), np.uint8))
            try:
                same = design.same_people(crops)
            except Exception as e:
                log(f"Duplicate check failed ({e}); keeping the characters as they are.")
                same = []
            _save_json(dpath, same)
        vis = feats[:, :, IDX["face_ok"]] == 1
        merged_into = {}
        for i, j in same:
            i, j = merged_into.get(i, i), merged_into.get(j, j)
            if i == j or i >= len(people) or j >= len(people):
                continue
            keep, drop = (i, j) if people[i]["seen"] >= people[j]["seen"] else (j, i)
            both = vis[:, keep] & vis[:, drop]
            apart = 0
            if both.any():
                a_, b_ = feats[both, keep], feats[both, drop]
                d_ = np.hypot(a_[:, IDX["fx"]] - b_[:, IDX["fx"]], a_[:, IDX["fy"]] - b_[:, IDX["fy"]])
                apart = int((d_ > np.maximum(a_[:, IDX["fh"]], b_[:, IDX["fh"]]) * 0.8).sum())
            # DeepSeek saw both photos and says it is one person: trust it unless the two are
            # clearly side by side most of the time (overlaps come from tracker noise, like a
            # face detected twice while two people lean together)
            if apart > 0.8 * min(vis[:, keep].sum(), vis[:, drop].sum()):
                log(f"Kept characters {keep + 1} and {drop + 1} apart: they are on screen together too often.")
                continue
            m = (feats[:, drop, IDX["face_ok"]] == 1) & (feats[:, keep, IDX["face_ok"]] != 1)
            feats[m, keep] = feats[m, drop]
            feats[:, drop, IDX["face_ok"]] = 0
            people[keep]["seen"] += people[drop]["seen"]
            people[keep]["shots"] = sorted(set(people[keep].get("shots", [])) | set(people[drop].get("shots", [])))
            notes[keep] = notes[keep] or notes[drop]
            merged_into[drop] = keep
            for k2, v2 in list(merged_into.items()):
                if v2 == drop:
                    merged_into[k2] = keep
            log(f"Duplicate check: characters {keep + 1} and {drop + 1} are the same person - merged.")
        if merged_into:
            keep_ids = [p["id"] for p in people if p["id"] not in merged_into]
            feats = feats[:, keep_ids, :]
            notes = [notes[i] for i in keep_ids]
            people = [people[i] for i in keep_ids]
            for new_id, p in enumerate(people):
                p["id"] = new_id

    # photo of each person for the artist (target circled)
    frames = read_frames(video, [p["best_frame"] for p in people])
    for p in people:
        img = frames.get(p["best_frame"])
        if img is None or not p.get("box"):
            continue
        row = feats[p["best_frame"], p["id"]]
        crop = _track_crop(img, {"box": p["box"]}, row)
        if crop is not None:
            cv2.imwrite(str(ddir / f"person_{p['id']}.jpg"), crop)
    return feats, people, notes


def watch_shots(d, video, feats, people, shots, notes, energy, fps, W, H, cb, log):
    """DeepSeek watches every shot at 10 frames per second (contact sheets with labelled
    faces) and reports who talks when, gestures, looks, poses and moods. Cached per job."""
    from . import design, watch
    path = d / "design" / "shot_watch.json"
    cached = {int(k): v for k, v in (_load_json(path) or {}).items()}
    if not providers.configured():
        log("No DeepSeek key: shots get automatic blocking.")
        return cached
    layouts = choreo.shot_layout(feats, people, shots, fps, W, H)
    total = sum(b - a for (a, b), lay in zip(shots, layouts) if lay) / fps
    want = config.WATCH_FPS.lower()
    want = int(round(fps)) if want in ("all", "max", "every") else max(1, int(float(want or 10)))
    want = min(want, int(round(fps)))  # can't watch more frames than the video has
    # one contact sheet per second either way, so cost grows with length, not with fps;
    # very long videos are watched at a lower rate to keep DeepSeek costs down
    rate = want if total <= 600 else max(2, int(want * 600 / total))
    log(f"Watching at {rate} of {fps:.0f} frames per second" + (" (every frame)" if rate >= int(round(fps)) else ""))
    windows = []
    for k, ((a, b), lay) in enumerate(zip(shots, layouts)):
        if not lay or k in cached or b - a < fps * 0.3:
            continue
        step = int(fps * 8)
        for w0 in range(a, b, step):
            windows.append((k, w0, min(w0 + step, b)))
    if not windows:
        return cached
    log(f"DeepSeek is watching {len(windows)} parts of the video at {rate} frames per second...")

    def voiced_text(a, b):
        v = energy[a:b] > 0.1
        spans, k = [], 0
        while k < len(v):
            if v[k]:
                j = k
                while j < len(v) and v[j]:
                    j += 1
                if j - k >= fps * 0.15:
                    spans.append(f"{k / fps:.1f}-{j / fps:.1f}s")
                k = j
            else:
                k += 1
        return ", ".join(spans[:30])

    done = [0]

    def work(win):
        k, a, b = win
        pids = [c["pid"] for c in layouts[k]]
        sheets = watch.contact_sheets(video, feats, pids, a, b, fps, rate=rate)
        desc = "\n".join(f'- id "{pid + 1}" (box P{pid + 1}): {notes[pid] if pid < len(notes) else ""}'[:300]
                         for pid in pids)
        try:
            res = design.watch_window(sheets, (a - shots[k][0]) / fps, (b - shots[k][0]) / fps, desc,
                                      voiced_text(a, b), choreo.BASE_POSES, list(choreo.MOODS), choreo.GESTURES)
        except Exception as e:
            log(f"Shot {k + 1}: watching failed ({e}); automatic blocking used there.")
            res = None
        done[0] += 1
        cb(done[0] / len(windows), f"DeepSeek watching part {done[0]}/{len(windows)}")
        return res

    for (k, a, b), res in zip(windows, design.run_parallel(work, windows, workers=4)):
        if not res:
            continue
        off = (a - shots[k][0]) / fps
        sc = cached.setdefault(k, {"people": {}, "talking": [], "actions": [], "looks": []})

        def pid_of(x):
            try:
                return str(int(x) - 1)  # labels are 1-based (P1 = id "1")
            except (TypeError, ValueError):
                return None
        for who, info in res["people"].items():
            p = pid_of(who)
            if p is not None and p not in sc["people"]:
                sc["people"][p] = info
        for key in ("talking", "actions", "looks"):
            for item in res[key]:
                p = pid_of(item["who"])
                if p is None:
                    continue
                item = dict(item, who=p, start=item["start"] + off, end=item["end"] + off)
                if key == "looks" and item["at"] not in ("camera", "left", "right"):
                    item["at"] = pid_of(item["at"]) or "camera"
                sc[key].append(item)
    _save_json(path, {str(k): v for k, v in cached.items()})
    talks = sum(len(v.get("talking", [])) for v in cached.values())
    acts = sum(len(v.get("actions", [])) for v in cached.values())
    log(f"DeepSeek watched {len(cached)} shots: {talks} talking turns, {acts} gestures")
    return cached


def render_set_preview(bg, fg, w, h):
    """PNG of a scene's set (background + foreground), for the set review."""
    import io
    from .sprites import background_surface
    import cairo
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    c = cairo.Context(surf)
    c.set_source_surface(background_surface(bg, w, h), 0, 0)
    c.paint()
    if fg:
        c.set_source_surface(background_surface(fg, w, h), 0, 0)
        c.paint()
    buf = io.BytesIO()
    surf.write_to_png(buf)
    return buf.getvalue()


def design_stage(job, d, video, track, people, opts, cb, log, notes=()):
    """DeepSeek draws every character and one background per group of similar shots."""
    from . import design
    ddir = d / "design"
    ddir.mkdir(exist_ok=True)
    desc = opts.get("description", "")
    use_ai = opts.get("art", "ai") in ("ai", "explainer") and providers.configured()
    if opts.get("art", "ai") in ("ai", "explainer") and not providers.configured():
        log("No DeepSeek key: using the built-in simple character. Add a key in Settings for AI-drawn art.")

    shots = track.get("shots") or []
    # keyframes: best frame per person + middle frame per shot
    want = [p["best_frame"] for p in people] + [(a + b) // 2 for a, b in shots]
    frames = read_frames(video, want) if want else {}

    # ---- characters ----
    count = 0 if opts.get("_no_chars") else max(len(people), 1)  # animator mode draws its own cast
    styles, rigs = [None] * count, [{} for _ in range(count)]
    crops = []
    for i in range(count):
        person = people[i] if people else None
        cp = ddir / f"person_{i}.jpg"  # written by cast_stage, target person circled
        crops.append(cv2.imread(str(cp)) if cp.exists() else None)
        sp = ddir / f"style_{i}.json"
        styles[i] = _load_json(sp) or _style_for(person, desc, log)
        _save_json(sp, styles[i])

    def make_character(i):
        rp = ddir / f"rig_{i}.json"
        cached = _load_json(rp)
        if cached:
            return cached
        try:
            log(f"DeepSeek is drawing character {i + 1}/{count}...")
            note = notes[i] if i < len(notes) else ""
            others = [n for j, n in enumerate(notes) if j != i and n]
            dz = design.draw_character(crops[i], desc, log, note=note, others=others)
            for n in range(int(config.REVIEW_PASSES or 0)):
                if crops[i] is None or not dz["parts"]:
                    break
                log(f"Reviewing character {i + 1} (pass {n + 1}): comparing the render with the video")
                png = render_preview(_apply_colors(styles[i], dz["colors"]), dz["parts"])
                (ddir / f"review_{i}_{n + 1}.png").write_bytes(png)
                dz = design.review_character(crops[i], dz, png, log)
            _save_json(rp, dz)
            return dz
        except Exception as e:
            log(f"DeepSeek character {i + 1} failed ({e}); using the built-in drawing.")
            return {"parts": {}, "colors": {}}

    # ---- backgrounds: one per group of similar-looking shots ----
    groups, shot_group, sigs = [], [], []
    max_bg = int(opts.get("max_backgrounds") or 12)
    for a, b in shots:
        mid = (a + b) // 2
        img = frames.get(mid)
        if img is None:
            shot_group.append(groups[-1]["id"] if groups else 0)
            continue
        sig = shot_signature(img)
        best, best_sim = None, -1
        for g, gsig in zip(groups, sigs):
            sim = float(np.minimum(sig, gsig).sum())  # histogram overlap, 1 = identical
            if sim > best_sim:
                best, best_sim = g, sim
        if best is not None and (best_sim > 0.75 or len(groups) >= max_bg):
            shot_group.append(best["id"])
            best["length"] += b - a
        else:
            groups.append({"id": len(groups), "frame": mid, "length": b - a})
            sigs.append(sig)
            shot_group.append(groups[-1]["id"])
            cv2.imwrite(str(ddir / f"shot_{groups[-1]['id']}.jpg"), img)
    _save_json(ddir / "scenes.json", {"groups": groups, "shot_group": shot_group, "shots": shots})
    log(f"{len(shots)} shots -> {len(groups)} different backgrounds")

    vw, vh = track["w"], track["h"]

    # which shots belong to each background group (for the 10-frame sheets)
    group_shots = {}
    for (sa, sb), gid in zip(shots, shot_group):
        group_shots.setdefault(gid, []).append((sa, sb))

    def make_background(g):
        """Background + foreground (table, props) for one scene, from 10 frames, then reviewed."""
        path, fpath = ddir / f"bg_{g['id']}.svg", ddir / f"fg_{g['id']}.svg"
        if path.exists():
            return path.read_text(encoding="utf-8")
        from . import watch
        # 10 frames spread over every shot of this scene: people move, so together they
        # show the whole set
        spans = group_shots.get(g["id"], [])
        total = sum(b2 - a2 for a2, b2 in spans) or 1
        picks = []
        for k in range(10):
            pos = (k + 0.5) / 10 * total
            for a2, b2 in spans:
                if pos < b2 - a2:
                    picks.append(a2 + int(pos))
                    break
                pos -= b2 - a2
        fr = read_frames(video, picks)
        sheet = None
        if fr:
            imgs = [fr[i] for i in sorted(fr)]
            hh, ww = imgs[0].shape[:2]
            sc = 240 / max(hh, ww)
            tw, th = int(ww * sc), int(hh * sc)
            cols = 5
            rows = (len(imgs) + cols - 1) // cols
            sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
            for m, im in enumerate(imgs):
                sheet[(m // cols) * th:(m // cols + 1) * th, (m % cols) * tw:(m % cols + 1) * tw] = cv2.resize(im, (tw, th))
            cv2.imwrite(str(ddir / f"set_{g['id']}.jpg"), sheet)
        bg = fg = None
        for attempt in range(2):  # one retry: replies are sometimes cut off or invalid
            try:
                log(f"DeepSeek is drawing scene {g['id'] + 1}/{len(groups)} (background + table/props)" + (" - retry" if attempt else ""))
                bg, fg = design.draw_set(frames[g["frame"]], sheet, vw, vh, desc, log, style=opts.get("_set_style"))
            except Exception as e:
                log(f"Scene {g['id'] + 1} failed ({e}).")
            if bg:
                break
        if bg and int(config.REVIEW_PASSES or 0) > 0:
            try:
                png = render_set_preview(bg, fg, vw, vh)
                (ddir / f"setreview_{g['id']}.png").write_bytes(png)
                bg, fg = design.review_set(frames[g["frame"]], png, bg, fg, vw, vh, log, style=opts.get("_set_style"))
            except Exception as e:
                log(f"Scene {g['id'] + 1} review failed ({e}); keeping the first drawing.")
        if bg:
            path.write_text(bg, encoding="utf-8")
        if fg:
            fpath.write_text(fg, encoding="utf-8")
        return bg

    if use_ai:
        jobs = [("c", i) for i in range(count)] + [("b", g) for g in groups]
        done = [0]

        def work(item):
            kind, x = item
            out = make_character(x) if kind == "c" else make_background(x)
            done[0] += 1
            cb(done[0] / len(jobs), f"DeepSeek drawing {done[0]}/{len(jobs)}")
            return out
        results = design.run_parallel(work, jobs, workers=4)
        for (kind, x), res in zip(jobs, results):
            if kind == "c":
                rigs[x] = res or {"parts": {}, "colors": {}}
                styles[x] = _apply_colors(styles[x], rigs[x].get("colors", {}))
        borrow = {}
        bg_svgs = {g["id"]: (ddir / f"bg_{g['id']}.svg").read_text(encoding="utf-8")
                   for g in groups if (ddir / f"bg_{g['id']}.svg").exists()}
        # a scene whose background failed borrows the most similar finished background
        if bg_svgs:
            done_ids = [g["id"] for g in groups if g["id"] in bg_svgs]
            for g, gsig in zip(groups, sigs):
                if g["id"] not in bg_svgs:
                    best = max(done_ids, key=lambda k: float(np.minimum(gsig, sigs[k]).sum()))
                    borrow[g["id"]] = best
                    log(f"Scene {g['id'] + 1} uses the background of scene {best + 1}.")
            shot_group = [borrow.get(gid, gid) for gid in shot_group]
    else:
        bg_svgs = {}

    # character sheet for the web page
    try:
        sheet = []
        for i in range(count):
            sheet.append(render_preview(styles[i], rigs[i].get("parts", {}), 480))
            (ddir / f"character_{i}.png").write_bytes(sheet[-1])
    except Exception as e:
        log(f"Preview failed: {e}")
    job.update(design={"characters": count, "backgrounds": len(bg_svgs), "groups": len(groups)})
    fg_svgs = {g["id"]: (ddir / f"fg_{g['id']}.svg").read_text(encoding="utf-8")
               for g in groups if (ddir / f"fg_{g['id']}.svg").exists()}
    for gid, src in (borrow.items() if use_ai and bg_svgs else []):
        if src in fg_svgs:
            fg_svgs[gid] = fg_svgs[src]
    return styles, rigs, bg_svgs, shot_group, fg_svgs


def direct_shots(d, video, feats, people, shots, fps, W, H, rigs, cb, log):
    """DeepSeek looks at 3 frames of every shot and blocks it (cached per job)."""
    from . import design
    path = d / "design" / "shot_scripts.json"
    (d / "design").mkdir(exist_ok=True)
    cached = {int(k): v for k, v in (_load_json(path) or {}).items()}
    if not providers.configured():
        log("No DeepSeek key: shots get automatic blocking (talking gestures only).")
        return cached
    layouts = choreo.shot_layout(feats, people, shots, fps, W, H)
    todo = [k for k, ((a, b), lay) in enumerate(zip(shots, layouts))
            if lay and (b - a) >= fps * 0.8 and k not in cached][:150]
    if not todo:
        return cached
    want = []
    for k in todo:
        a, b = shots[k]
        want += [a + (b - a) // 6, (a + b) // 2, b - 1 - (b - a) // 6]
    frames = read_frames(video, want)

    def describe(c):
        note = (rigs[c["pid"]].get("analysis") or "")[:160] if c["pid"] < len(rigs) else ""
        return f'- id "{c["pid"]}": at {c["hx"] / W:.0%} from the left. {note}'

    done = [0]

    def work(k):
        a, b = shots[k]
        imgs = [frames[i] for i in (a + (b - a) // 6, (a + b) // 2, b - 1 - (b - a) // 6) if i in frames]
        try:
            res = design.script_shot(imgs, (b - a) / fps, "\n".join(describe(c) for c in layouts[k]),
                                     choreo.BASE_POSES, list(choreo.MOODS), choreo.GESTURES)
        except Exception as e:
            log(f"Shot {k + 1}: directing failed ({e}); automatic blocking used.")
            res = None
        done[0] += 1
        cb(done[0] / len(todo), f"DeepSeek directing shot {done[0]}/{len(todo)}")
        return res

    for k, res in zip(todo, design.run_parallel(work, todo, workers=4)):
        if res:
            cached[k] = res
    _save_json(path, {str(k): v for k, v in cached.items()})
    acts = sum(len(v.get("actions", [])) for v in cached.values())
    log(f"DeepSeek directed {len(cached)} shots ({acts} gestures)")
    return cached


def run_job(job):
    d = Path(job.dir)
    opts = job.options
    log = job.log
    prog = Progress(job)
    t0 = time.time()

    # story mode: a topic / script / YouTube link -> a brand-new original video
    if opts.get("mode") == "story":
        from . import story
        story.run(job, d, opts, prog, log)
        return

    # 1. source + normalized video
    prog.stage("prepare")
    src = next((p for p in d.glob("source.*") if p.suffix != ".part"), None)
    if opts.get("youtube_url") and src is None:
        log(f"Downloading {opts['youtube_url']}")
        src, title = media.download_youtube(opts["youtube_url"], d, prog.cb("prepare"))
        log(f"Downloaded: {title}")
    if src is None:
        raise RuntimeError("No input video.")
    info = media.probe(src)
    fps = min(info["fps"], 60.0)
    start = opts.get("start") or None
    end = opts.get("end") or None
    video = d / "video.mp4"
    if not video.exists():
        job.update(message="Normalizing video")
        media.normalize(src, video, fps, start, end)
    log(f"Video: {info['duration']:.1f}s at {fps:.2f} fps")

    # 2. audio (original voice)
    prog.stage("audio")
    wav = d / "audio.wav"
    if not wav.exists():
        media.extract_audio(src, wav, start, end, info["has_audio"], info["duration"])
    data, sr = lipsync.read_wav(wav)
    duration = len(data) / sr

    # 3. transcript
    prog.stage("transcribe")
    tpath = d / "transcript.json"
    if opts.get("transcribe", True) and info["has_audio"]:
        if not tpath.exists():
            _save_json(tpath, transcribe(wav, duration, prog.cb("transcribe")))
        transcript = _load_json(tpath)
        log(f"Transcript: {len(transcript['segments'])} lines ({transcript.get('language')})")
    else:
        transcript = {"segments": []}
    segments = transcript["segments"]

    # animator mode: no face tracking - DeepSeek finds the characters and writes the
    # animation itself (works for cartoons, full-body action, any kind of video)
    if opts.get("motion") == "animator":
        from . import animator
        animator.run(job, d, video, wav, info, opts, prog, log)
        return

    # 4. tracking every frame: people + cuts
    prog.stage("track")
    npz, tjson = d / "tracking.npz", d / "tracking.json"
    cached = _load_json(tjson) if tjson.exists() else None
    if npz.exists() and cached and cached.get("version") == TRACK_VERSION:
        z = np.load(npz)
        feats, tracked = z["feats"], z["tracked"]
        rows_by, k = {}, 0
        for tid, length in z["tl_index"]:
            rows_by[int(tid)] = z["tl_rows"][k:k + length]
            k += length
        track = cached
    else:
        if npz.exists():
            # saved by an older version of the app: track again, and forget drawings tied to
            # the old people / shots (they may be numbered differently now)
            log("Tracking was saved by an older version - tracking again (better face + cut detection).")
            ddir = d / "design"
            if ddir.exists():
                for f in ddir.iterdir():
                    if f.name.startswith(("rig_", "style_", "person_", "character_", "review_", "casting", "cast_",
                                          "shot_watch", "shot_scripts", "fg_", "set",
                                          "bg_", "shot_", "scenes")):
                        f.unlink(missing_ok=True)
            (d / "output.mp4").unlink(missing_ok=True)
        step = 2 if opts.get("quality") == "fast" else 1
        res = track_video(video, step=step, progress=prog.cb("track"),
                          max_visible=4, max_characters=int(opts.get("max_people") or 8))
        feats = res.pop("feats")
        rows_by, tracked = res.pop("rows_by"), res.pop("tracked")
        res["version"] = TRACK_VERSION
        tl_index = np.array([[tid, len(r)] for tid, r in rows_by.items()], np.int64).reshape(-1, 2)
        tl_rows = np.concatenate(list(rows_by.values())) if rows_by else np.zeros((0, feats.shape[2]), np.float32)
        np.savez_compressed(npz, feats=feats, tracked=tracked, tl_index=tl_index, tl_rows=tl_rows)
        _save_json(tjson, res)
        track = res
    people = track["people"]
    n_video = feats.shape[0]
    log(f"Tracked {n_video} frames: {len(track.get('shots') or [])} shots, {len(people)} different people "
        "(" + ", ".join(f"person {p['id'] + 1}: {p['seen']} frames in {len(p.get('shots') or [])} shots"
                        for p in people) + ")")

    fps = track["fps"] or fps
    n = n_video or int(round(duration * fps))  # one cartoon frame per video frame

    # 5. lip-sync from the voice (used where a face is not visible)
    prog.stage("lipsync")
    energy = lipsync.frame_energy(wav, fps, n)
    audio_mouth = None
    solo = len(people) <= 1
    directed = opts.get("motion", "director") == "director" and bool(people)
    face_pct = float(np.nanmean(feats[:, 0, 0] == 1) * 100) if people else 0.0
    if (directed or (solo and face_pct < 97)) and info["has_audio"]:
        exe = rhubarb_exe()
        cues_path = d / "rhubarb.json"
        if exe:
            try:
                if not cues_path.exists():
                    job.update(message="Rhubarb lip-sync from voice")
                    lipsync.run_rhubarb(exe, wav, cues_path)
                cues = _load_json(cues_path)
                cues = cues["mouthCues"] if isinstance(cues, dict) else cues
                audio_mouth = lipsync.cues_to_frames(cues, fps, n)
                log(f"Rhubarb: {len(cues)} mouth shapes")
            except Exception as e:
                log(f"Rhubarb failed ({e}); using loudness-based mouth.")
    if audio_mouth is None:
        audio_mouth = lipsync.energy_to_frames(energy)

    # 6. DeepSeek draws characters + backgrounds (and reviews them)
    prog.stage("design")
    feats, people, notes = cast_stage(d, video, feats, people, track, rows_by, tracked, opts, log)
    track["people"] = people
    styles, rigs, bg_svgs, shot_group, fg_svgs = design_stage(job, d, video, track, people, opts, prog.cb("design"), log, notes)

    H = _even(opts.get("height") or 720)
    W = _even(H * track["w"] / track["h"]) if n_video and track["h"] else _even(H * 16 / 9)
    count = max(len(people), 1)
    shots = track.get("shots") or [[0, n]]

    # 7. directing: DeepSeek blocks every shot (poses, moods, timed gestures), like Claude
    prog.stage("direct")
    scripts, emotions = {}, []
    if directed:
        scripts = watch_shots(d, video, feats, people, shots, notes, energy, fps, W, H, prog.cb("direct"), log)
    else:
        epath = d / "emotions.json"
        emotions = _load_json(epath)
        if emotions is None:
            emotions = director.plan_emotions(segments, log, prog.cb("direct"))
            _save_json(epath, emotions)

    # 8. render every frame
    prog.stage("render")
    from .sprites import background_surface, build_rig
    if directed:
        speech = lipsync.speech_mask(segments, fps, n)
        log(f"Speech (words) in {speech.mean():.0%} of frames - mouths stay closed during music and silence")
        states, layouts, who = choreo.build_states(feats, people, shots, scripts, energy, audio_mouth, fps, n, W, H,
                                                   speech=speech)
        log("Director mode: " + ", ".join(f"shot {k + 1}: {len(l)} on screen" for k, l in enumerate(layouts[:12]))
            + (" ..." if len(layouts) > 12 else ""))
    else:
        states = []
        for i in range(count):
            F = feats[:, i, :] if people else None
            cuts = [a for a, _ in shots]
            states.append(animate.build(F, n, fps, W, H, energy, audio_mouth, segments, emotions, solo=solo, cuts=cuts))
    r_max = max(float(np.max(s["r"])) for s in states)
    zoom = min(max(r_max / 100.0 * 1.15, 0.6), 4.0)
    puppets = [(styles[i], build_rig(rigs[i].get("parts", {}), zoom)) for i in range(count)]
    backgrounds, foregrounds = {}, {}
    for g, svg in bg_svgs.items():
        try:
            backgrounds[g] = background_surface(svg, W, H)
            if g in fg_svgs:
                foregrounds[g] = background_surface(fg_svgs[g], W, H)
        except Exception as e:
            log(f"Background {g} could not be rendered: {e}")
    painter = Painter(W, H, puppets, backgrounds, bg_style=styles[0], foregrounds=foregrounds)

    shot_of = np.zeros(n, dtype=np.int64)
    push_in = np.zeros(n)
    for si, (a, b) in enumerate(track.get("shots") or []):
        shot_of[a:min(b, n)] = si
        push_in[a:min(b, n)] = choreo.ease(np.linspace(0, 1, max(min(b, n) - a, 1)))[:max(min(b, n) - a, 0)]
    subs = opts.get("subtitles", False)
    caps = captions(segments) if subs else []
    cap_idx = np.full(n, -1, dtype=np.int64)
    for ci, cp in enumerate(caps):
        cap_idx[max(int(cp["start"] * fps), 0):min(int(cp["end"] * fps) + 1, n)] = ci

    out = d / "output.tmp.mp4"
    enc = media.open_encoder(out, W, H, fps, wav, n)
    keys = [[k for k in s if k != "seg"] for s in states]
    cb = prog.cb("render")
    try:
        for i in range(n):
            if job.cancelled:
                raise RuntimeError("Cancelled")
            frame_states = [{k: float(s[k][i]) for k in ks} for s, ks in zip(states, keys)]
            group = shot_group[shot_of[i]] if shot_group else None
            ci = int(cap_idx[i])
            push = 1.0 + 0.035 * push_in[i] if directed else 1.0  # slow camera push-in per shot
            enc.stdin.write(painter.draw(frame_states, group, caps[ci]["text"] if ci >= 0 else None, zoom=push))
            if i % 30 == 0:
                cb(i / n, f"Drawing frame {i}/{n}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore")
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed: " + err[-1500:])
    except BaseException:
        enc.kill()
        raise
    shutil.move(out, d / "output.mp4")
    log(f"Done: {n} frames, {W}x{H}, {count} characters, {len(backgrounds)} backgrounds, in {time.time() - t0:.0f}s")
    job.update(progress=100, stage="done", message="Finished")
