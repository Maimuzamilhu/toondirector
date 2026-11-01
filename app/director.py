"""DeepSeek as the 'director': designs the character and picks an emotion per sentence.

Works without an API key too (uses defaults + simple text rules).
"""
import json
import re

from . import config, providers

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "thinking", "excited"]
HAIR_STYLES = ["short", "long", "spiky", "bun", "curly", "bald", "hijab"]

DEFAULT_STYLE = {
    "skin": "#f2c29b",
    "hair_style": "short",
    "hair_color": "#3b2a20",
    "eye_color": "#4a3426",
    "shirt_color": "#3d7cc9",
    "beard": False,
    "beard_color": "#3b2a20",
    "glasses": False,
    "bg_top": "#bfe3ff",
    "bg_bottom": "#f4f9ff",
    "outline": "#2b1d17",
}

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _ask_json(system, user):
    text, _, _ = providers.chat(system + "\n\n" + user, max_tokens=1500, temperature=0.4, json_mode=True)
    return providers.parse_json(text)


def _clean_style(raw, base):
    style = dict(base)
    for k, v in (raw or {}).items():
        if k not in DEFAULT_STYLE:
            continue
        if isinstance(DEFAULT_STYLE[k], bool):
            style[k] = bool(v)
        elif k == "hair_style":
            if v in HAIR_STYLES:
                style[k] = v
        elif isinstance(v, str) and HEX.match(v):
            style[k] = v.lower()
    return style


def design_character(description, sampled_colors, log=print):
    """Returns a style dict. Colours sampled from the video are the starting point."""
    base = dict(DEFAULT_STYLE)
    base.update({"skin": sampled_colors.get("skin", base["skin"]),
                 "shirt_color": sampled_colors.get("shirt", base["shirt_color"]),
                 "hair_color": sampled_colors.get("hair", base["hair_color"])})
    if sampled_colors.get("beard"):
        base.update(beard=True, beard_color=sampled_colors["beard"])
    if not description.strip() or not providers.configured():
        if description.strip():
            log("No DEEPSEEK_API_KEY set - using default character.")
        return base
    system = (
        "You design a simple 2D cartoon character for a vector renderer. Reply with JSON only, "
        "keys: skin, hair_style, hair_color, eye_color, shirt_color, beard (bool), beard_color, "
        "glasses (bool), bg_top, bg_bottom, outline. Colours are #rrggbb. "
        f"hair_style is one of {HAIR_STYLES} (hijab = head scarf, hair_color is its colour). "
        "bg_top/bg_bottom are a soft background gradient that fits the description. "
        "Keep the sampled colours unless the description says otherwise."
    )
    user = f"Description: {description}\nColours sampled from the video: {json.dumps(sampled_colors)}"
    try:
        style = _clean_style(_ask_json(system, user), base)
        log(f"DeepSeek designed the character: {style}")
        return style
    except Exception as e:  # network / key / JSON problems should not kill the job
        log(f"DeepSeek design failed ({e}); using default character.")
        return base


def _rule_emotion(text):
    t = text.lower()
    if "?" in t and any(w in t for w in ("why", "how", "what", "hmm")):
        return "thinking"
    if "!" in t:
        return "excited"
    if any(w in t for w in ("sad", "sorry", "unfortunately", "miss")):
        return "sad"
    if any(w in t for w in ("great", "love", "happy", "awesome", "thanks", "thank you")):
        return "happy"
    return "neutral"


def plan_emotions(segments, log=print, progress=None, chunk_chars=3500):
    """One emotion per transcript segment. Long transcripts are sent in chunks."""
    if not segments:
        return []
    if not providers.configured():
        log("No DEEPSEEK_API_KEY set - using simple text rules for emotions.")
        return [_rule_emotion(s["text"]) for s in segments]

    system = (
        "You are an animation director. For each numbered line of a video transcript, choose the "
        f"facial emotion the cartoon speaker should show, one of {EMOTIONS}. Consider context. "
        'Reply with JSON: {"emotions": [{"i": <number>, "e": "<emotion>"}, ...]} covering every line.'
    )
    result = [None] * len(segments)
    chunks, cur, size = [], [], 0
    for i, s in enumerate(segments):
        line = f"{i}: {s['text']}"
        if cur and size + len(line) > chunk_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append(cur)

    for ci, chunk in enumerate(chunks):
        try:
            data = _ask_json(system, "\n".join(chunk))
            for item in data.get("emotions", []):
                i, e = int(item.get("i", -1)), item.get("e")
                if 0 <= i < len(result) and e in EMOTIONS:
                    result[i] = e
        except Exception as e:
            log(f"DeepSeek emotion chunk {ci + 1} failed ({e}); using rules for it.")
        if progress:
            progress((ci + 1) / len(chunks), f"DeepSeek directing part {ci + 1}/{len(chunks)}")
    return [r or _rule_emotion(s["text"]) for r, s in zip(result, segments)]
