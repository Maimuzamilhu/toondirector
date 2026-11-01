"""DeepSeek as the artist, the way Claude Design works: it looks at the video frames,
writes SVG code for each character (as separate puppet parts) and for each scene's
background, then reviews a rendered test frame against the original and fixes it.

Our code animates the parts with the per-frame tracking, so the drawings only
need to be still poses in fixed coordinates.
"""
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import config, providers

# Every part is drawn in "head units": head radius = 100, head centre = (0, 0).
PARTS = {
    "hair_back": ("-170 -170 340 520",
                  "Hair or headwear BEHIND the head and shoulders (long hair falling down, back of a "
                  "hijab/turban/hood, ponytail). Empty <svg/> if the person has short hair."),
    "head": ("-130 -130 260 260",
             "Face skin shape + ears + neck top. Skull top at y=-100, chin at y=+100, cheeks at "
             "x=+-86. Soft shading on one side. NO eyes, NO eyebrows, NO mouth (they are animated "
             "separately)."),
    "face_details": ("-130 -130 260 260",
                     "Things drawn ON the face over the eyes: nose (around 0,22), cheek lines, "
                     "freckles, moles, wrinkles, beard / moustache / stubble, glasses frames. Keep the "
                     "mouth area (ellipse centre 0,52, rx 26, ry 20) EMPTY - the animated mouth goes "
                     "there. Keep the eye areas (+-33,-5, rx 18, ry 22) clear except glasses frames."),
    "hair_front": ("-170 -190 340 330",
                   "Hair on top of the head and fringe/bangs, caps, hats, the front of a hijab or "
                   "turban. Must leave the eyes and eyebrows visible: nothing inside the box "
                   "x=-62..62, y=-48..20."),
    "torso": ("-240 -80 480 560",
              "Neckline/collar centred at (0,0), shoulders at (+-115, 25), upper body down to "
              "y=470. Clothing with collar, buttons, pockets, logos, patterns, shading, a mic or "
              "lanyard if they wear one. NO arms (sleeves are separate). Leave the neck itself "
              "out: a skin-coloured neck is drawn under the collar."),
    "upper_arm": ("-55 -40 110 220",
                  "One upper arm hanging straight down: shoulder joint at (0,0), elbow at (0,140), "
                  "about 70 wide. Usually the sleeve of the shirt. Round ends so joints overlap."),
    "forearm": ("-50 -30 100 190",
                "One forearm hanging straight down: elbow at (0,0), wrist at (0,125), about 60 wide. "
                "Sleeve or bare skin, with a cuff if the person has one."),
    "hand": ("-50 -20 100 110",
             "One relaxed open hand hanging down from the wrist at (0,0), palm towards the viewer, "
             "fingers ending around y=70. Skin colour, simple cartoon fingers, watch/bracelet if any."),
    # full-body characters (animator mode): legs hang from the hips at (+-55, 430) of the torso
    "thigh": ("-60 -30 120 260",
              "One upper leg hanging straight down: hip joint at (0,0), knee at (0,200), about 85 wide. "
              "Trousers/shorts/skirt or bare skin. Round ends so the joints overlap."),
    "shin": ("-55 -25 110 240",
             "One lower leg hanging straight down: knee at (0,0), ankle at (0,190), about 70 wide. "
             "Trouser leg, socks or skin."),
    "foot": ("-50 -25 160 90",
             "One foot/shoe seen from the side, ankle at (0,0), toes pointing to +x (image right), "
             "sole around y=45, about 120 long."),
}
FULL_ONLY = ("thigh", "shin", "foot")

ART_STYLE = (
    "Modern flat 2D cartoon (explainer-video / animated sitcom style): bold clean shapes, "
    "dark outlines (stroke-width 3 to 4, stroke-linejoin round), 2-3 tones per colour for "
    "shading and a highlight, slightly exaggerated caricature that is clearly recognisable "
    "as the real person (hair shape, beard, glasses, face shape, skin tone, clothes)."
)

CHARACTER_PROMPT = """You are an expert vector illustrator who builds cut-out animation puppets in hand-written SVG code (like Toon Boom / Adobe Character Animator rigs).

Look at the photo of the person and draw them as a cartoon puppet: {style}
{extra}
The puppet is made of separate parts. Our engine moves them (head turns, arm bends, blinks, lip-sync), so each part is a still drawing in FIXED coordinates. Use exactly this viewBox for each part and keep every shape inside it:

{parts}

Eyes, eyebrows and the mouth are NOT drawn by you - the engine draws them animated: eyes at (+-33, -5) about 32 wide and 40 tall, eyebrows at y=-38, mouth centred at (0, 50) about 40 wide. Shape the face so these fit naturally.

Rules:
- Plain SVG 1.1 only: path, circle, ellipse, rect, polygon, polyline, line, g, linearGradient, radialGradient, clipPath. No text, no images, no filters, no scripts, no CSS.
- Every part is its own complete <svg xmlns="http://www.w3.org/2000/svg" viewBox="..."> ... </svg>.
- Transparent background (never fill the whole viewBox).
- Rich detail: hair strands/clumps, clothing folds, seams, shading, highlights.

Reply in exactly this format and nothing else:
<analysis>2-3 sentences: face shape, hair, skin tone, beard, glasses, clothes, accessories</analysis>
<colors>{{"skin": "#rrggbb", "iris": "#rrggbb", "brow": "#rrggbb", "lips": "#rrggbb"}}</colors>
{part_tags}
"""

REVIEW_PROMPT = """You drew a cartoon puppet of the person in the FIRST image. The SECOND image is how the puppet renders now (eyes, brows and mouth are drawn by the engine).
Compare them like an art director. Check: likeness (hair shape, beard, glasses, face shape, skin tone), clothes and colours, proportions, parts that look broken, misplaced, cut off or empty.
{style}
Coordinates and rules are the same as before:
{parts}

Reply in exactly this format:
<fixes>up to 6 short bullet points of what you change</fixes>
then ONLY the parts you changed, each as <part name="..."><svg ...>...</svg></part>. If nothing needs fixing reply <fixes>none</fixes>.
"""

BACKGROUND_PROMPT = """You are a background artist for a 2D cartoon. Redraw the scene in this video frame as a detailed flat-vector cartoon background in hand-written SVG. {style}
{extra}
- viewBox="0 0 {w} {h}" (same framing and camera angle as the photo; keep big shapes where they are).
- REMOVE all people: draw what would be behind them (walls, curtains, furniture), because animated characters are placed on top later.
- Keep the setting recognisable: walls, windows, shelves, lights, desk, stage, logos as simple shapes, colour mood and lighting. Add depth with 2-3 tones and soft shadows. No text.
- Plain SVG only (path, rect, circle, ellipse, polygon, line, g, gradients). No images, filters, scripts or CSS.
Reply with only the <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">...</svg> code."""

# used when no face is visible in the video: invent the character from the description
NARRATOR_PROMPT = CHARACTER_PROMPT.replace(
    "Look at the photo of the person and draw them as a cartoon puppet:",
    'Design an original character from this description: "{desc}". Draw them as a cartoon puppet:')


HEAD_PARTS = ["hair_back", "head", "face_details", "hair_front"]
BODY_PARTS = ["torso", "upper_arm", "forearm", "hand"]
LEG_PARTS = ["thigh", "shin", "foot"]

# used when the source video is already a drawing/cartoon: copy its look
STYLE_MATCH = (
    "MATCH THE ART STYLE OF THE SOURCE IMAGE exactly if it is a drawing or cartoon: its "
    "proportions (e.g. huge caricature heads, skinny limbs), line quality (sketchy pencil "
    "outlines, hatching), colours and shading. If it is a real photo, make an expressive "
    "caricature of the person: exaggerated head and features, clean outlines, 2-3 tones per colour."
)


def _parts_text(names=None):
    names = names or list(PARTS)
    return "\n".join(f'- {k}: viewBox="{PARTS[k][0]}". {PARTS[k][1]}' for k in names)


def _part_tags(names):
    return "\n".join(f'<part name="{k}"><svg ...>...</svg></part>' for k in names)


def _jpeg_data_url(img_bgr, max_side=768):
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _png_data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _chat(content, max_tokens=16000, temperature=0.6, think=False):
    """One AI call. Thinking is off by default: all output tokens go to the SVG code."""
    text, finish, _ = providers.chat(content, max_tokens=max_tokens, temperature=temperature, think=think)
    if finish == "length":
        text += "\n<!-- truncated -->"
    return text


def _ask(text, images=(), **kw):
    content = [{"type": "text", "text": text}]
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return _chat(content if images else text, **kw)


# ---------------- parsing / cleaning LLM SVG ----------------
_BAD = [
    (re.compile(r"<script\b.*?</script>", re.S | re.I), ""),
    (re.compile(r"<foreignObject\b.*?</foreignObject>", re.S | re.I), ""),
    (re.compile(r"<image\b.*?(/>|</image>)", re.S | re.I), ""),  # no external pictures
    (re.compile(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", re.I), ""),
]
# text (shop signs) and filters (glow / soft blur) are allowed: resvg renders them and
# nothing in an SVG drawn as an image can run code


def clean_svg(svg, viewbox=None):
    svg = svg.strip()
    for rx, rep in _BAD:
        svg = rx.sub(rep, svg)
    m = re.match(r"<svg\b([^>]*)>", svg, re.S)
    if not m:
        return None
    attrs = m.group(1)
    attrs = re.sub(r'\s(width|height|viewBox|preserveAspectRatio|xmlns)\s*=\s*("[^"]*"|\'[^\']*\')', "", attrs)
    if viewbox:
        attrs += f' viewBox="{viewbox}"'
    else:
        vb = re.search(r'viewBox\s*=\s*"([^"]*)"', m.group(0))
        attrs += f' viewBox="{vb.group(1)}"' if vb else ""
    head = f'<svg xmlns="http://www.w3.org/2000/svg"{attrs}>'
    if not svg.endswith("</svg>"):
        svg += "</svg>"  # truncated reply: close it so the finished shapes still render
    return head + svg[m.end():]


def parse_parts(text):
    parts = {}
    for name, body in re.findall(r'<part\s+name="([a-z_]+)"\s*>\s*(.*?)\s*</part>', text, re.S):
        if name in PARTS:
            svg = clean_svg(body, PARTS[name][0])
            if svg:
                parts[name] = svg
    return parts


def parse_colors(text):
    m = re.search(r"<colors>\s*(\{.*?\})\s*</colors>", text, re.S)
    try:
        raw = json.loads(m.group(1)) if m else {}
    except Exception:
        raw = {}
    return {k: v for k, v in raw.items() if isinstance(v, str) and re.match(r"^#[0-9a-fA-F]{6}$", v)}


def validate(parts):
    """Drops parts resvg can't render or that render empty."""
    from .sprites import render_svg
    ok = {}
    for name, svg in parts.items():
        try:
            surf = render_svg(svg, 0.5)
            if surf is not None:
                ok[name] = svg
        except Exception:
            pass
    return ok


# ---------------- public steps ----------------
def draw_character(person_img, description="", log=print, note="", others=(), full_body=False, style=None):
    """Returns {"parts": {name: svg}, "colors": {...}, "analysis": str}.

    Head parts and body parts are requested in parallel (two shorter replies).
    note: casting note for this person; others: notes of the rest of the cast, so the
    characters come out clearly different from each other."""
    extra = []
    if note:
        extra.append(f"The person to draw is circled in yellow. Casting note: {note}")
    if others:
        extra.append("Other characters in this cartoon (yours must look clearly DIFFERENT from them - "
                     "exaggerate this person's own distinctive features):\n" + "\n".join(f"- {o}" for o in others))
    if description:
        extra.append(f"Extra direction from the user: {description}")
    extra = "\n".join(extra)
    images = [_jpeg_data_url(_ready_crop(person_img))] if person_img is not None else []
    base = CHARACTER_PROMPT if images else NARRATOR_PROMPT

    def one(names):
        prompt = base.format(style=style or ART_STYLE, extra=extra, parts=_parts_text(names),
                             part_tags=_part_tags(names), desc=description or "friendly young presenter")
        who = "upper body and arm" if names is BODY_PARTS else "leg and foot" if names is LEG_PARTS else "head"
        prompt += (f"\nThis request is ONLY for the {who} parts listed above (other artists draw the rest of "
                   "the same character; keep the same style and colours).")
        if names is LEG_PARTS:
            prompt += "\nThe legs hang from the hips at (+-55, 430) of the torso drawing (torso neckline at 0,0)."
        return _ask(prompt, images)

    groups = [HEAD_PARTS, BODY_PARTS] + ([LEG_PARTS] if full_body else [])
    texts = run_parallel(one, groups, workers=3)
    head_text, body_text = texts[0], texts[1]
    merged = {}
    for t in texts:
        merged.update(parse_parts(t))
    parts = validate(merged)
    colors = parse_colors(head_text) or parse_colors(body_text)
    analysis = re.search(r"<analysis>(.*?)</analysis>", head_text, re.S)
    note = analysis.group(1).strip() if analysis else ""
    wanted = [k for k in PARTS if full_body or k not in FULL_ONLY]
    log(f"DeepSeek drew {len(parts)}/{len(wanted)} parts" + (f": {note[:200]}" if note else ""))
    missing = [k for k in wanted if k not in parts and k != "hair_back"]
    if missing:
        log("Missing parts (built-in drawing used instead): " + ", ".join(missing))
    return {"parts": parts, "colors": colors, "analysis": note}


def review_character(person_img, design, preview_png, log=print):
    """One Claude-style review pass: compare original vs render, return improved design."""
    prompt = REVIEW_PROMPT.format(style=ART_STYLE, parts=_parts_text())
    text = _ask(prompt, [_jpeg_data_url(person_img), _png_data_url(preview_png)], think=True, max_tokens=32000)
    fixes = re.search(r"<fixes>(.*?)</fixes>", text, re.S)
    fixed = validate(parse_parts(text))
    log("Review: " + (fixes.group(1).strip().replace("\n", " ")[:300] if fixes else "no notes")
        + f" ({len(fixed)} parts redrawn)")
    if fixed:
        design = dict(design, parts={**design["parts"], **fixed})
    return design


SET_PROMPT = """You are the set designer of a 2D cartoon remake of a real video. Image 1 is one frame of a scene; image 2 is a sheet of up to 10 frames from the same camera angle (so you can see parts of the set that people hide in one frame).
Redraw the SET as flat-vector cartoon art in hand-written SVG, in TWO layers. {style}
{extra}
Layer 1 <background>: everything BEHIND the people - walls, curtains, stage, lights, screens, logos as simple shapes. No people. Fill the whole viewBox.
Layer 2 <foreground>: things IN FRONT of the people that hide part of their bodies - e.g. the table/desk top and front panel, plates of food, cups, bottles, phones, mic stands, a podium. Transparent everywhere else (no full-size rectangles). If nothing is in front of the people, an empty <svg/>.
Both layers: viewBox="0 0 {w} {h}", same framing and camera angle as the video, keep big shapes exactly where they are in the frames (the table edge must be at the same height as in the video). Match the source's colour palette and BRIGHTNESS closely. 2-3 tones per colour, soft shadows. Allowed: path, rect, circle, ellipse, polygon, line, g, linearGradient, radialGradient, <text> for real signs/labels you can read in the video (font-family="sans-serif"), and <filter> with feGaussianBlur for soft glow around lamps/windows. No images, scripts or CSS.

Reply in exactly this format:
<background><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">...</svg></background>
<foreground><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">...</svg></foreground>"""

SET_REVIEW_PROMPT = """Image 1 is a real video frame. Image 2 is your cartoon redraw of its set (people are added later by the engine, so they are not in image 2).
Compare like an art director: are the walls/curtains/stage, colours, lighting, the table position and height, and the objects in front of the people right? Fix what is wrong or missing. {style}
Same rules as before: <background> fills the viewBox; <foreground> only has objects in front of people, transparent elsewhere; viewBox="0 0 {w} {h}".
Reply: <fixes>short bullets</fixes> then the corrected <background>...</background> and/or <foreground>...</foreground> (only the layers you changed). If all is right reply <fixes>none</fixes>."""


def _layer(text, tag, w, h):
    m = re.search(rf"<{tag}>\s*(<svg\b.*?(?:</svg>|$))", text, re.S)
    if not m:
        return None
    svg = clean_svg(m.group(1), f"0 0 {w} {h}")
    return svg if svg and validate({tag: svg}) else None


EXPLAINER_STYLE = (
    "Polished modern explainer-animation illustration (flat vector, but rich and cinematic): every "
    "surface uses soft linearGradient/radialGradient shading instead of flat fills; warm glowing "
    "windows, lamps and light pools (radial gradients that fade to transparent); atmospheric depth "
    "(far layers lighter, hazier and bluer); lots of small believable details (bricks, window frames "
    "with curtains, plants, signs, shelves, cables, street furniture); soft long shadows; NO black "
    "outlines - shapes are separated by colour and light. Harmonious palette (dusk purples and warm "
    "oranges, or fresh daylight greens and blues). Leave open floor/ground space where characters stand."
)


def draw_set(frame_bgr, sheet_bgr, w, h, description="", log=print, style=None):
    """Returns (background_svg, foreground_svg) for one scene."""
    extra = f"Extra direction from the user: {description}" if description else ""
    images = [_jpeg_data_url(frame_bgr, 1024)]
    if sheet_bgr is not None:
        images.append(_jpeg_data_url(sheet_bgr, 1200))
    text = _ask(SET_PROMPT.format(style=style or ART_STYLE, extra=extra, w=w, h=h), images, max_tokens=24000)
    bg, fg = _layer(text, "background", w, h), _layer(text, "foreground", w, h)
    if not bg:  # older reply style: a single svg
        m = re.search(r"<svg\b.*?(</svg>|$)", text, re.S)
        svg = clean_svg(m.group(0), f"0 0 {w} {h}") if m else None
        bg = svg if svg and validate({"bg": svg}) else None
    if not bg:
        log("Background drawing failed for one scene.")
    return bg, fg


def review_set(frame_bgr, render_png, bg, fg, w, h, log=print, style=None):
    """Claude-style check: compare the cartoon set with the real frame and fix it."""
    text = _ask(SET_REVIEW_PROMPT.format(style=style or ART_STYLE, w=w, h=h),
                [_jpeg_data_url(frame_bgr, 1024), _png_data_url(render_png)], think=True, max_tokens=32000)
    fixes = re.search(r"<fixes>(.*?)</fixes>", text, re.S)
    nb, nf = _layer(text, "background", w, h), _layer(text, "foreground", w, h)
    log("Set review: " + (fixes.group(1).strip().replace("\n", " ")[:220] if fixes else "no notes"))
    return nb or bg, nf or fg


TEXT_SET_PROMPT = """You are the set designer of an original 2D animated YouTube video. Design this set from its description and draw it in hand-written SVG, in TWO layers. {style}
Set: {desc}
{extra}
Layer 1 <background>: everything behind the characters. Fill the whole viewBox. The floor/ground where characters stand is the band around y = {floor} (keep it open and walkable, no big objects on it).
Layer 2 <foreground>: things in front of the characters (table tops, counters, plants at the very front, railings), transparent everywhere else; an empty <svg/> if nothing belongs in front.
viewBox="0 0 {w} {h}". Rich, believable detail, 2-3 tones per colour, soft shadows, glow around lights. Allowed: path, rect, circle, ellipse, polygon, line, g, linearGradient, radialGradient, <text> for signs (font-family="sans-serif"), <filter> with feGaussianBlur. No images, scripts or CSS.
Reply in exactly this format:
<background><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">...</svg></background>
<foreground><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">...</svg></foreground>"""

TEXT_SET_REVIEW_PROMPT = """This is your render of the set "{desc}" for a polished 2D explainer video. Judge it like an art director: is it rich, well lit, clearly that place, with an open floor band around y = {floor}? Fix weak or empty areas, flat colours, wrong proportions, clutter on the floor. {style}
Reply: <fixes>short bullets</fixes> then the improved <background>...</background> and/or <foreground>...</foreground> (same viewBox 0 0 {w} {h}), or <fixes>none</fixes>."""


def draw_set_text(desc, w, h, extra="", log=print, style=None, review=True):
    """Background + foreground for an original video, from a text description."""
    from .sprites import background_surface
    floor = int(h * 0.86)
    text = _ask(TEXT_SET_PROMPT.format(style=style or EXPLAINER_STYLE, desc=desc, extra=extra, w=w, h=h, floor=floor),
                max_tokens=24000)
    bg, fg = _layer(text, "background", w, h), _layer(text, "foreground", w, h)
    if bg and review:
        import io

        import cairo
        surf = cairo.ImageSurface(cairo.FORMAT_RGB24, w // 2, h // 2)
        c = cairo.Context(surf)
        c.set_source_surface(background_surface(bg, w // 2, h // 2), 0, 0)
        c.paint()
        if fg:
            c.set_source_surface(background_surface(fg, w // 2, h // 2), 0, 0)
            c.paint()
        buf = io.BytesIO()
        surf.write_to_png(buf)
        try:
            t2 = _ask(TEXT_SET_REVIEW_PROMPT.format(desc=desc, floor=floor, w=w, h=h, style=style or EXPLAINER_STYLE),
                      [_png_data_url(buf.getvalue())], think=True, max_tokens=32000)
            fixes = re.search(r"<fixes>(.*?)</fixes>", t2, re.S)
            log("Set review: " + (fixes.group(1).strip().replace("\n", " ")[:160] if fixes else "no notes"))
            bg = _layer(t2, "background", w, h) or bg
            fg = _layer(t2, "foreground", w, h) or fg
        except Exception as e:
            log(f"Set review skipped ({e})")
    return bg, fg


# kept for compatibility
def draw_background(frame_bgr, w, h, description="", log=print):
    return draw_set(frame_bgr, None, w, h, description, log)[0]


SHOT_PROMPT = """You are the director of a 2D cartoon remake of a real video, like an animator blocking a scene.
These {k} images are the start, middle and end of ONE shot ({dur:.1f} seconds long).
The cartoon characters in this shot are (left to right as seen in the image):
{people}

For each character choose a base pose for the whole shot and list the clear, visible actions (with times in seconds from the start of the shot). Only list actions you can actually see; keep it simple and believable. Talking hand movement is added automatically for whoever speaks, so do not list "talk".

base_pose: one of {bases} (hold_mic_left/right = holding a microphone in the hand on that side of the image)
mood: one of {moods}
gesture: one of {gestures}

Reply with JSON only:
{{"people": {{"<id>": {{"base_pose": "...", "mood": "..."}}}},
 "actions": [{{"who": "<id>", "gesture": "...", "start": 0.0, "end": 1.5}}]}}"""


def script_shot(frames, dur, people_desc, bases, moods, gestures):
    """DeepSeek watches a shot (3 frames) and blocks it: base poses, moods, timed gestures."""
    prompt = SHOT_PROMPT.format(k=len(frames), dur=dur, people=people_desc, bases=bases,
                                moods=moods, gestures=gestures)
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": _jpeg_data_url(f, 640)}} for f in frames]
    data = _json_call(content, max_tokens=2000, temperature=0.3)
    out = {"people": {}, "actions": []}
    for pid, info in (data.get("people") or {}).items():
        out["people"][str(pid)] = {
            "base_pose": info.get("base_pose") if info.get("base_pose") in bases else "idle",
            "mood": info.get("mood") if info.get("mood") in moods else "neutral"}
    for a in data.get("actions") or []:
        if a.get("gesture") in gestures:
            try:
                st = max(float(a.get("start", 0)), 0.0)
                en = min(float(a.get("end", st + 1.5)), dur + 0.5)
                if en > st:
                    out["actions"].append({"who": str(a.get("who")), "gesture": a["gesture"], "start": st, "end": en})
            except (TypeError, ValueError):
                pass
    return out


CAST_PROMPT = """You are casting a 2D cartoon remake of a video. These {k} photos show the people found in it (photo 1 = "{first}", in the order listed below). The target person in each photo is circled in yellow.
{ids}

1. If two photos show the SAME real person (same face, hair, clothes), mark the later one as same_as the earlier one.
2. For every distinct person, write a short visual note a cartoonist needs to draw them so they are instantly recognisable AND clearly different from everyone else: face shape, skin tone, hair shape and colour, facial hair, glasses/sunglasses, clothes (colour, pattern, type), accessories, body build. Put the features that make them different from the others first.

Reply with JSON only: {{"people": {{"<id>": {{"same_as": null or "<id>", "note": "..."}}}}}}"""


def _ready_crop(img):
    """Small video crops are upscaled so the model can see details."""
    h, w = img.shape[:2]
    if min(h, w) < 384:
        s = 384 / min(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return img


def _json_call(content, max_tokens=3000, temperature=0.3):
    """AI call that must return JSON. A broken or cut-off reply is asked again, a little
    cooler each time, and the last try goes to another provider if there is one."""
    last, skip = None, ()
    for attempt in range(3):
        text, _, used = providers.chat(content, max_tokens=max_tokens, temperature=max(temperature - 0.2 * attempt, 0.1),
                                       json_mode=True, skip=skip)
        try:
            return providers.parse_json(text)
        except ValueError as e:
            last = e
            if attempt == 1 and len(providers.ready()) > 1:
                skip = (used,)
    raise RuntimeError(f"The AI returned broken data 3 times ({last})")


def cast_people(crops):
    """crops: list of (id, image). Returns {id: {"same_as": id|None, "note": str}}."""
    ids = "\n".join(f'- photo {k + 1}: id "{pid}"' for k, (pid, _) in enumerate(crops))
    content = [{"type": "text", "text": CAST_PROMPT.format(k=len(crops), first=crops[0][0], ids=ids)}]
    content += [{"type": "image_url", "image_url": {"url": _jpeg_data_url(_ready_crop(img), 512)}} for _, img in crops]
    data = _json_call(content)
    out = {}
    valid = {str(pid) for pid, _ in crops}
    for pid, info in (data.get("people") or {}).items():
        if str(pid) in valid and isinstance(info, dict):
            same = info.get("same_as")
            same = str(same) if same is not None and str(same) in valid and str(same) != str(pid) else None
            out[str(pid)] = {"same_as": same, "note": str(info.get("note", ""))[:600]}
    return out


GROUP_PROMPT = """You are casting a 2D cartoon remake of a video. A face tracker cut the video into {k} face tracks; image N shows track "T{{N}}" - one or two moments of that track side by side, the tracked face circled in yellow (other people may be partly visible at the edges: ignore them). The same real person usually appears in several tracks (different shots, angles, sunglasses on/off, side views).
Group the tracks into REAL PEOPLE. Use face, hair, beard, glasses/sunglasses, skin tone, and especially CLOTHES (people do not change clothes during the video).
Hard rule - these pairs are on screen at the SAME time in different places, so they are NOT the same person:
{conflicts}
If a track does not show a real face (a hand, a microphone, an object, a blur), list it in not_person.
For each person write a short note for the cartoonist: what makes them recognisable and different from the others (put the differences first).

Reply with JSON only:
{{"people": [{{"tracks": ["T1", "T4"], "note": "..."}}], "not_person": ["T7"]}}"""


def group_tracks(crops, conflicts):
    """crops: list of images (track N = crops[N-1]); conflicts: list of (i, j) 1-based pairs.
    Returns (groups as lists of 0-based indices, notes, not_person indices)."""
    ctext = ", ".join(f"T{i}-T{j}" for i, j in conflicts[:200]) or "none"
    content = [{"type": "text", "text": GROUP_PROMPT.format(k=len(crops), conflicts=ctext)}]
    content += [{"type": "image_url", "image_url": {"url": _jpeg_data_url(_ready_crop(img), 320)}} for img in crops]
    data = _json_call(content, max_tokens=4000)

    def idx(x):
        try:
            k = int(str(x).strip().lstrip("Tt")) - 1
            return k if 0 <= k < len(crops) else None
        except ValueError:
            return None
    not_person = {i for i in (idx(x) for x in data.get("not_person") or []) if i is not None}
    groups, notes, used = [], [], set()
    for p in data.get("people") or []:
        members = [i for i in (idx(x) for x in p.get("tracks") or []) if i is not None
                   and i not in used and i not in not_person]
        if members:
            used.update(members)
            groups.append(members)
            notes.append(str(p.get("note", ""))[:600])
    return groups, notes, not_person


SAME_PROMPT = """These {k} photos (photo 1 to {k}, in order) each show one character from the same video, circled in yellow. Some may be the SAME real person photographed from a different angle, in profile, with or without sunglasses, or partly hidden by a microphone or hand.
Compare face, hair, beard, skin tone and especially clothes (people do not change clothes during the video).
Which photos show the same person? Only list pairs you are confident about.
Reply with JSON only: {{"same": [[1, 4]]}} (an empty list if all are different people)."""


def same_people(crops):
    """Returns 0-based index pairs of photos that show the same person."""
    content = [{"type": "text", "text": SAME_PROMPT.format(k=len(crops))}]
    content += [{"type": "image_url", "image_url": {"url": _jpeg_data_url(_ready_crop(img), 384)}} for img in crops]
    data = _json_call(content, max_tokens=500)
    out = []
    for pr in data.get("same") or []:
        try:
            i, j = int(pr[0]) - 1, int(pr[1]) - 1
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= i < len(crops) and 0 <= j < len(crops) and i != j:
            out.append([min(i, j), max(i, j)])
    return out


WATCH_PROMPT = """You are the director of a 2D cartoon remake of a real video. You are watching ONE shot at 10 frames per second: each image is one second (10 frames, left-to-right, top-to-bottom, each labelled with its time from the start of this part). Tracked people have a coloured box labelled P1, P2, ...
This part covers {t0:.1f}s to {t1:.1f}s of the shot. The voice track has sound at: {voiced}.
Characters here:
{people}

Report what each character does, with times in seconds from the start of THIS PART (0.1 s precision):
- talking: when each person is the one speaking (mouth moving, speaking into a mic). Use the voice timing to help. Several people can talk at once.
- actions: visible gestures, from: {gestures}
- looks: where each person looks: another person's id, "camera", "left" or "right".
- base_pose for the whole part, one of {bases} (hold_mic_left/right = holding a microphone in the hand on that side of the image) and mood, one of {moods}.

Reply with JSON only:
{{"people": {{"<id>": {{"base_pose": "...", "mood": "..."}}}},
 "talking": [{{"who": "<id>", "start": 0.0, "end": 1.2}}],
 "actions": [{{"who": "<id>", "gesture": "...", "start": 0.0, "end": 1.0}}],
 "looks": [{{"who": "<id>", "at": "<id>|camera|left|right", "start": 0.0, "end": 2.0}}]}}"""


def watch_window(sheets, t0, t1, people_desc, voiced_text, bases, moods, gestures):
    """DeepSeek watches one window (<= ~8 s) of a shot at 10 fps. Times returned are
    relative to the window start."""
    prompt = WATCH_PROMPT.format(t0=t0, t1=t1, people=people_desc, voiced=voiced_text or "none",
                                 bases=bases, moods=moods, gestures=gestures)
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": _jpeg_data_url(img, 1000)}} for _, img in sheets]
    data = _json_call(content, max_tokens=3000)
    dur = t1 - t0

    def spans(key, field=None, allowed=None):
        res = []
        for a in data.get(key) or []:
            try:
                st = max(float(a.get("start", 0)), 0.0)
                en = min(float(a.get("end", st + 1)), dur + 0.2)
            except (TypeError, ValueError):
                continue
            if en <= st:
                continue
            item = {"who": str(a.get("who")), "start": st, "end": en}
            if field:
                val = a.get(field)
                if allowed is not None and val not in allowed:
                    continue
                item[field] = str(val)
            res.append(item)
        return res

    out = {"people": {}, "talking": spans("talking"), "actions": spans("actions", "gesture", gestures),
           "looks": spans("looks", "at")}
    for pid, info in (data.get("people") or {}).items():
        if isinstance(info, dict):
            out["people"][str(pid)] = {
                "base_pose": info.get("base_pose") if info.get("base_pose") in bases else "idle",
                "mood": info.get("mood") if info.get("mood") in moods else "neutral"}
    return out


def run_parallel(fn, items, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
