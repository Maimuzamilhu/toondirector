"""Story mode: a topic, a script or a YouTube link -> a brand-new, original 2D video.

Nothing is copied from a source video. A YouTube link is only read for its transcript,
as research about the topic; DeepSeek then writes a NEW script (own structure, wording
and examples). Every picture is drawn by the app:

  1. SCRIPT  DeepSeek writes (or, for your own script, splits) a scene-by-scene plan:
             a fixed CAST and fixed SETTINGS that scenes reuse, narration, actions,
             badges, titles, camera moves, sound effects and the music mood
  2. SETS    each setting is drawn once (background + foreground), then self-reviewed
  3. VOICE   neural narration per scene (edge-tts) with the time of every word
  4. ACTION  DeepSeek animates every scene: keyframes for each character, timed to the
             narration's words
  5. SOUND   generated background music (ducked under the voice) + sound effects
  6. EDIT    transitions, camera moves, word-by-word captions, title card, badges,
             glow/vignette finish -> MP4 (16:9 YouTube or 9:16 Shorts)
"""
import asyncio
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import providers, research
from . import config

LANGS = {"en": "English", "ur": "Urdu", "hi": "Hindi", "ar": "Arabic", "es": "Spanish", "fr": "French",
         "de": "German", "tr": "Turkish", "id": "Indonesian", "pt": "Portuguese", "bn": "Bengali"}
VOICES = {
    ("en", "male"): "en-US-AndrewNeural", ("en", "female"): "en-US-AvaNeural",
    ("ur", "male"): "ur-PK-AsadNeural", ("ur", "female"): "ur-PK-UzmaNeural",
    ("hi", "male"): "hi-IN-MadhurNeural", ("hi", "female"): "hi-IN-SwaraNeural",
    ("ar", "male"): "ar-SA-HamedNeural", ("ar", "female"): "ar-SA-ZariyahNeural",
    ("es", "male"): "es-ES-AlvaroNeural", ("es", "female"): "es-ES-ElviraNeural",
    ("fr", "male"): "fr-FR-HenriNeural", ("fr", "female"): "fr-FR-DeniseNeural",
    ("de", "male"): "de-DE-ConradNeural", ("de", "female"): "de-DE-KatjaNeural",
    ("tr", "male"): "tr-TR-AhmetNeural", ("tr", "female"): "tr-TR-EmelNeural",
    ("id", "male"): "id-ID-ArdiNeural", ("id", "female"): "id-ID-GadisNeural",
    ("pt", "male"): "pt-BR-AntonioNeural", ("pt", "female"): "pt-BR-FranciscaNeural",
    ("bn", "male"): "bn-IN-BashkarNeural", ("bn", "female"): "bn-IN-TanishaaNeural",
}
POSES = ["stand", "sit", "walk", "run", "jump", "crouch", "lie", "wave", "point", "cheer", "think", "open_arms",
         "hips", "shrug", "salute", "bow", "fist", "read", "write", "kneel"]
ACCESSORIES = ["none", "cap", "beanie", "bun", "long_hair", "ponytail", "hat", "crown", "turban", "helmet", "glasses",
               "scarf", "dress", "robe", "beard"]
SFX = ["whoosh", "pop", "ding", "click", "thud", "swoosh_up"]
MOODS = ["curious", "upbeat", "calm", "dramatic", "happy", "mysterious"]


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def _save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------- 0. research from a YouTube link
def _is_rate_limited(e):
    t = str(e)
    return "429" in t or "Too Many Requests" in t or "rate" in t.lower()


def _vtt_text(path):
    lines, last = [], ""
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if not ln or "-->" in ln or ln.startswith(("WEBVTT", "Kind:", "Language:")) or ln == last:
            continue
        lines.append(ln)
        last = ln
    return " ".join(lines)


def video_title(url):
    """A public video's title, from YouTube's official oEmbed address. Nothing is downloaded."""
    import json as _json
    import urllib.parse
    import urllib.request
    query = urllib.parse.urlencode({"url": url, "format": "json"})
    try:
        with urllib.request.urlopen(f"https://www.youtube.com/oembed?{query}", timeout=15) as r:
            return _json.load(r).get("title", "")
    except (OSError, ValueError):
        return ""


def youtube_material(url, d, log):
    """The video's transcript, used only as research. Tries subtitles first (one language
    at a time, with pauses); if YouTube blocks or has none, downloads only the audio and
    transcribes it locally with Whisper."""
    import yt_dlp
    from .config import FFMPEG
    base = {"quiet": True, "noplaylist": True, "retries": 3, "sleep_interval_requests": 1,
            "sleep_interval_subtitles": 3}
    title = ""
    # which languages does this video actually have subtitles in? (one light request)
    tries = []
    try:
        with yt_dlp.YoutubeDL(dict(base, skip_download=True)) as ydl:
            meta = ydl.extract_info(url, download=False)
        title = meta.get("title", "") or ""
        orig = (meta.get("language") or "").split("-")[0]
        manual = list((meta.get("subtitles") or {}).keys())
        auto = list((meta.get("automatic_captions") or {}).keys())
        # best first: human subtitles in the video's own language, then its auto captions,
        # then any human subtitles, then English
        for code in ([c for c in manual if c.split("-")[0] == orig] +
                     [c for c in auto if c in (f"{orig}-orig", orig)] + manual[:2] + ["en"]):
            if code and [code] not in tries:
                tries.append([code])
        if orig:
            log(f"Source video language: {orig} - reading its own subtitles")
    except Exception as e:
        if _is_rate_limited(e):
            log("YouTube is limiting requests right now - waiting a moment...")
            time.sleep(8)
    if not tries:
        tries = [["en"], ["en-orig", "en.*"], ["ur"], ["hi"]]
    # 1. subtitles (cheap) - one language request at a time, so one refusal does not end it
    for langs in tries[:4]:
        try:
            with yt_dlp.YoutubeDL(dict(base, skip_download=True, writesubtitles=True, writeautomaticsub=True,
                                       subtitlesformat="vtt", subtitleslangs=langs,
                                       outtmpl=str(d / "ref.%(ext)s"))) as ydl:
                info = ydl.extract_info(url, download=True)
            title = info.get("title", "") or title
        except Exception as e:
            if _is_rate_limited(e):
                log("YouTube is limiting subtitle downloads right now - waiting a moment...")
                time.sleep(8)
            continue
        subs = sorted(d.glob("ref*.vtt"))
        if subs:
            text = _vtt_text(subs[0])
            if len(text) > 200:
                log(f"Read the subtitles of: {title}")
                return title, text[:30000]
    # 2. audio only + Whisper on this PC
    log("No subtitles available - downloading only the audio to transcribe it here (research only)...")
    aopts = dict(base, format="bestaudio/best", outtmpl=str(d / "ref_audio.%(ext)s"),
                 ffmpeg_location=str(Path(FFMPEG).parent) if Path(FFMPEG).exists() else None)
    last_err = None
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(aopts) as ydl:
                info = ydl.extract_info(url, download=True)
            title = info.get("title", "") or title
            last_err = None
            break
        except Exception as e:
            last_err = e
            if _is_rate_limited(e):
                log(f"YouTube is limiting downloads (try {attempt + 1}/3) - waiting {20 * (attempt + 1)} s...")
                time.sleep(20 * (attempt + 1))
            else:
                break
    if last_err is not None:
        raise RuntimeError("YouTube blocked the download (too many requests from this internet connection). "
                           "Wait 10-30 minutes and press Retry, or paste the video's transcript into the "
                           "'My script' tab instead.")
    src = next(p for p in d.glob("ref_audio.*") if p.suffix != ".wav")
    wav = d / "ref_audio.wav"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    from .lipsync import read_wav
    from .transcribe import transcribe
    data, sr = read_wav(wav)
    tr = transcribe(wav, len(data) / sr)
    return title, " ".join(s["text"] for s in tr["segments"])[:30000]


# ----------------------------------------------------------------- 1. script
SCRIPT_PROMPT = """You are the head writer and director of an animated YouTube channel (polished 2D explainer style: simple characters with round black heads and white dot eyes, rich illustrated sets).
{task}
Language of the narration, captions, badges and titles: {lang}.
Target length: about {minutes} minutes ({words} words of narration in total, spoken at a natural pace).
{notes}

Plan the video as JSON:
- "title": catchy YouTube title. "hook": the first sentence must make people keep watching.
- "cast": EVERY character, defined ONCE and reused in all scenes (never invent a second copy of the same person). Each: id (c1, c2, ...), name, description, kind (human, crowd, bird, seal, animal), look = {{"body": "#rrggbb" clothing colour, "accent": "#rrggbb", "accessory": one of {acc}, "height": 0.6-1.2}}. Give every character a different colour/accessory so viewers can tell them apart. Keep the cast small (1-5) unless the story needs more.
- "settings": every place, defined ONCE and reused: id (s1, s2, ...), description (a rich visual description for an illustrator: the place, time of day, lighting, colours, props).
- "scenes": 6-14 seconds of narration each. Each scene: setting (a settings id), narration (the exact words the narrator says), speaker (the cast id whose mouth moves, or null for an off-screen narrator), characters (list of {{"id", "x": 0.1-0.9 where they stand, "pose": one of {poses}, "action": what they do in this scene}}), badges (list of short on-screen texts like "75 YEARS" or "POPULATION 128", may be empty), title (big on-screen title text or ""), camera (static, push_in, pan_left, pan_right), transition_in (cut, fade, whip), sfx (list of {{"name": one of {sfx}, "at": seconds into the scene}}).
- "music_mood": one of {moods}.
EVERY scene must show at least one character on screen (a host/narrator character presents when no story character is there; add crowds, soldiers or people of the time where they fit), and the speaker should usually be someone on screen.

Write like a top YouTube channel: strong hook, clear structure, surprising facts, simple language, a satisfying ending with a call to action. Make the visuals tell the story too.
Reply with JSON only."""


OUTLINE_PROMPT = """You are the head writer of an animated YouTube channel (polished 2D explainer style).
{task}
Language of the narration, captions, badges and titles: {lang}.
This is a LONG video: about {minutes} minutes. Plan it as {chapters} chapters of about {per_ch} words of narration each.
{notes}
Reply with JSON only:
{{"title": "catchy YouTube title", "hook": "the first sentence of the video",
 "cast": [every character, defined ONCE for the whole video: {{"id": "c1", "name", "description", "kind": human|crowd|bird|seal|animal, "look": {{"body": "#rrggbb", "accent": "#rrggbb", "accessory": one of {acc}, "height": 0.6-1.2}}}}],
 "settings": [every place, defined ONCE: {{"id": "s1", "description": "rich visual description: place, time of day, lighting, colours, props"}}],
 "music_mood": one of {moods},
 "chapters": [{{"title": "...", "summary": "what this chapter covers, its facts, examples and story beats", "settings": ["s1"], "characters": ["c1"]}}]}}
Keep the cast small (1-6) and give each character a different colour/accessory. A strong hook, a clear arc across the chapters, a satisfying ending with a call to action."""

CHAPTER_PROMPT = """You are writing chapter {n} of {total} of the YouTube video "{title}". Language: {lang}.
Chapter: {ch_title} - {summary}
Story so far (previous chapter summaries): {before}
{notes}
Write {words} words of narration for THIS chapter - between {lo} and {hi} words, count them (the video must hit its length: not shorter, not longer), split into scenes of 6-14 seconds (about 15-35 words each). Continue naturally from the previous chapter; do not re-introduce the video {extra}.
EVERY scene must show at least one character on screen (a host/narrator character presents when no story character is there; add crowds, soldiers or people of the time where they fit), and the speaker should usually be someone on screen.
Use ONLY these characters (by id): {cast}
and ONLY these settings (by id): {settings}
Each scene: setting, narration, speaker (a character id whose mouth moves, or null for an off-screen narrator), characters ([{{"id", "x": 0.1-0.9, "pose": one of {poses}, "action"}}]), badges (short on-screen texts, may be empty), title (on-screen title, usually "" except at the start of the chapter), camera (static, push_in, pan_left, pan_right), transition_in (cut, fade, whip), sfx ([{{"name": one of {sfx}, "at": seconds}}]).
Reply with JSON only: {{"scenes": [ ... ]}}"""

ADD_PROMPT = """Chapter {n} ("{ch_title}": {summary}) of the YouTube video "{title}" needs about {need} MORE words of narration. Write NEW scenes that continue right after its last scene (add explanation, examples, surprising facts, small story moments - no repetition). Language: {lang}.
Use ONLY these characters: {cast}
and ONLY these settings: {settings}
Same scene format: setting, narration, speaker, characters ([{{"id", "x", "pose": one of {poses}, "action"}}]), badges, title, camera, transition_in, sfx ([{{"name": one of {sfx}, "at"}}]).
The chapter so far (JSON): {scenes}
Reply with JSON only: {{"scenes": [ only the NEW scenes ]}}"""

SHORTEN_PROMPT = """This chapter of the YouTube video "{title}" is too long ({have} words; it must be about {want}). Rewrite it SHORTER in {lang}: keep the best facts, examples and story beats, cut repetition and filler, keep the same characters and settings. Same scene format.
Current chapter scenes (JSON): {scenes}
Reply with JSON only: {{"scenes": [ ... ]}}"""

EXPAND_PROMPT = """This chapter of the YouTube video "{title}" is too short ({have} words; it needs about {want}). Rewrite it LONGER in {lang}: add more explanation, examples, surprising facts and small story moments, keeping the same characters and settings. Same scene format as before.
Current chapter scenes (JSON): {scenes}
Reply with JSON only: {{"scenes": [ ... ]}}"""


def _task_text(kind, material):
    if kind == "topic":
        return f"Write an ORIGINAL video about this topic: {material}\nResearch it from your own knowledge; be accurate."
    if kind == "script":
        return ("Turn THIS SCRIPT into a video. Keep its meaning and wording (translate only if it is in another "
                f"language), and split it into scenes:\n---\n{material[:20000]}\n---")
    return ("Below is the transcript of an existing YouTube video. Use it ONLY as research about the topic. "
            "Write a completely NEW, ORIGINAL video: your own structure, your own wording, your own examples "
            "and jokes, your own characters and settings. Never copy sentences from it.\n---\n"
            f"{material[:20000]}\n---")


def _words(scenes):
    return sum(len(str(sc.get("narration", "")).split()) for sc in scenes)


def write_script(kind, material, lang, minutes, notes, log):
    """Short videos: one plan. Long videos (and when a plan comes out short): an outline,
    then every chapter written on its own with a word target, then expanded until the
    narration is long enough for the minutes asked for."""
    from . import design
    langname = LANGS.get(lang, lang)
    target = int(minutes * 140)  # words at a natural narration pace
    task = _task_text(kind, material)
    nt = f"Extra direction: {notes}" if notes else ""

    if minutes <= 3 or kind == "script":
        prompt = SCRIPT_PROMPT.format(task=task, lang=langname, minutes=minutes, words=target, notes=nt,
                                      acc=ACCESSORIES, poses=POSES, sfx=SFX, moods=MOODS)
        data = design._json_call([{"type": "text", "text": prompt}], max_tokens=24000, temperature=0.8)
        if kind == "script" or _words(data.get("scenes") or []) >= target * 0.8:
            return clean_script(data, log)
        log(f"The first plan was too short ({_words(data.get('scenes') or [])} of ~{target} words) - "
            "writing it chapter by chapter instead...")

    # --- long video: outline -> chapters (in parallel) -> expand short ones
    chapters_n = max(2, int(round(minutes / 1.5)))
    per_ch = int(target / chapters_n)
    log(f"Writing a {minutes:g}-minute video: outline + {chapters_n} chapters of ~{per_ch} words")
    outline = design._json_call([{"type": "text", "text": OUTLINE_PROMPT.format(
        task=task, lang=langname, minutes=minutes, chapters=chapters_n, per_ch=per_ch, notes=nt,
        acc=ACCESSORIES, moods=MOODS)}], max_tokens=8000, temperature=0.8)
    chs = (outline.get("chapters") or [])[:max(chapters_n + 4, 4)] or [{"title": "Main", "summary": str(material)[:300]}]
    cast_txt = "; ".join(f'{c.get("id")} = {c.get("name")}' for c in outline.get("cast") or [])
    set_txt = "; ".join(f'{x.get("id")} = {str(x.get("description", ""))[:80]}' for x in outline.get("settings") or [])

    def write_ch(k):
        ch = chs[k]
        before = " | ".join(c.get("summary", "")[:160] for c in chs[:k]) or "(this is the first chapter)"
        extra = "" if k else "- this FIRST chapter opens with the hook: " + str(outline.get("hook", ""))
        if k == len(chs) - 1:
            extra += " - this is the LAST chapter: end with a satisfying wrap-up and a call to action"
        data = design._json_call([{"type": "text", "text": CHAPTER_PROMPT.format(
            n=k + 1, total=len(chs), title=outline.get("title", ""), lang=langname, ch_title=ch.get("title", ""),
            summary=ch.get("summary", ""), before=before, notes=nt, words=per_ch, lo=int(per_ch * 0.9),
            hi=int(per_ch * 1.1), extra=extra, cast=cast_txt,
            settings=set_txt, poses=POSES, sfx=SFX)}], max_tokens=12000, temperature=0.8)
        scenes = data.get("scenes") or []
        for _ in range(2):  # fix the length: expand a short chapter, tighten a long one
            have = _words(scenes)
            if per_ch * 0.85 <= have <= per_ch * 1.2:
                break
            tmpl = EXPAND_PROMPT if have < per_ch * 0.85 else SHORTEN_PROMPT
            fixed = design._json_call([{"type": "text", "text": tmpl.format(
                title=outline.get("title", ""), have=have, want=per_ch, lang=langname,
                scenes=json.dumps(scenes, ensure_ascii=False)[:14000])}], max_tokens=12000, temperature=0.6)
            new = fixed.get("scenes") or []
            if new and abs(_words(new) - per_ch) < abs(have - per_ch):
                scenes = new
        return scenes

    parts = [sc or [] for sc in design.run_parallel(write_ch, list(range(len(chs))), workers=4)]
    for k, sc in enumerate(parts):
        log(f"Chapter {k + 1}/{len(chs)}: {_words(sc)} words")

    # top up: short chapters get EXTRA scenes appended (more reliable than rewriting)
    for _ in range(3):
        total = sum(_words(x) for x in parts)
        if total >= target * 0.92:
            break
        short = sorted(range(len(parts)), key=lambda k: _words(parts[k]) - per_ch)
        short = [k for k in short if _words(parts[k]) < per_ch][:4]
        if not short:
            break
        log(f"Script is {total} of ~{target} words - adding scenes to {len(short)} short chapter(s)...")

        def add(k):
            need = max(per_ch - _words(parts[k]), 40)
            data = design._json_call([{"type": "text", "text": ADD_PROMPT.format(
                title=outline.get("title", ""), n=k + 1, ch_title=chs[k].get("title", ""),
                summary=chs[k].get("summary", ""), lang=langname, need=need, cast=cast_txt, settings=set_txt,
                poses=POSES, sfx=SFX, scenes=json.dumps(parts[k], ensure_ascii=False)[:10000])}],
                max_tokens=8000, temperature=0.7)
            return data.get("scenes") or []
        for k, extra_sc in zip(short, design.run_parallel(add, short, workers=4)):
            parts[k] = parts[k] + extra_sc
    all_scenes = [sc for part in parts for sc in part]
    data = {"title": outline.get("title", ""), "cast": outline.get("cast") or [],
            "settings": outline.get("settings") or [], "music_mood": outline.get("music_mood"),
            "scenes": all_scenes}
    got = _words(all_scenes)
    log(f"Script length: {got} words (~{got / 140:.1f} min) for {minutes:g} minutes asked")
    return clean_script(data, log)


def ensure_on_screen(cast, scenes, log):
    """Every scene shows at least one character. Empty scenes get the host/narrator as an
    on-screen presenter whose mouth follows the narration."""
    host = next((c for c in cast if any(w in c["name"].lower() for w in
                                        ("narrator", "host", "presenter", "guide", "teacher"))), None)
    if host is None:
        humans = [c for c in cast if c["kind"] == "human"]
        host = humans[0] if humans else None
    if host is None:  # no usable character at all: add a host
        host = {"id": "host", "name": "Host", "description": "the friendly presenter of the channel",
                "kind": "human", "look": {"body": "#2a9d8f", "accent": "#e9c46a", "accessory": "glasses", "height": 1.0}}
        cast.insert(0, host)
    added = 0
    for k, sc in enumerate(scenes):
        if not sc["characters"]:
            sc["characters"].append({"id": host["id"], "x": 0.2 if k % 2 == 0 else 0.8, "pose": "stand",
                                     "action": "presents this part to the viewer, gesturing and pointing at "
                                               "what is being explained"})
            if not sc.get("speaker"):
                sc["speaker"] = host["id"]
            added += 1
    if added:
        log(f"{added} scene(s) had nobody on screen - {host['name']} presents them")


def _distinct_colours(cast):
    """No two people in nearly the same clothes colour, or viewers cannot tell them apart."""
    palette = ["#c0392b", "#2a9d8f", "#e0a42b", "#7b4bc4", "#3f7fd1", "#d96a9a", "#5aa05a", "#c9783a", "#4bb3c4"]
    seen = []

    def rgb_(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    def far(c):
        return all(sum(abs(a - b) for a, b in zip(rgb_(c), rgb_(o))) > 90 for o in seen)

    for c in cast:
        if c["kind"] != "human":
            continue
        if not far(c["look"]["body"]):
            c["look"]["body"] = next((p for p in palette if far(p)), c["look"]["body"])
        seen.append(c["look"]["body"])


def clean_script(data, log):
    """Makes the plan safe to use: known ids only, sane numbers, no duplicate characters."""
    from .animator import _clean_look
    cast, seen_names = [], set()
    for i, c in enumerate(data.get("cast") or []):
        name = str(c.get("name", f"Character {i + 1}"))[:40]
        if name.lower() in seen_names:  # the same person twice -> keep one
            continue
        seen_names.add(name.lower())
        cast.append({"id": str(c.get("id") or f"c{i + 1}"), "name": name,
                     "description": str(c.get("description", ""))[:300],
                     "kind": c.get("kind") if c.get("kind") in ("human", "crowd", "bird", "seal", "animal") else "human",
                     "look": _clean_look(c.get("look") if isinstance(c.get("look"), dict) else {})})
    settings = [{"id": str(s.get("id") or f"s{i + 1}"), "description": str(s.get("description", ""))[:700]}
                for i, s in enumerate(data.get("settings") or [])]
    if not settings:
        settings = [{"id": "s1", "description": "a cosy, warmly lit studio with shelves, plants and a big window"}]
    cids, sids = {c["id"] for c in cast}, {s["id"] for s in settings}
    scenes = []
    for sc in data.get("scenes") or []:
        narr = str(sc.get("narration", "")).strip()
        if not narr:
            continue
        chars = []
        for ch in sc.get("characters") or []:
            if isinstance(ch, dict) and str(ch.get("id")) in cids and str(ch.get("id")) not in [x["id"] for x in chars]:
                try:
                    x = min(max(float(ch.get("x", 0.5)), 0.05), 0.95)
                except (TypeError, ValueError):
                    x = 0.5
                chars.append({"id": str(ch["id"]), "x": x,
                              "pose": ch.get("pose") if ch.get("pose") in POSES else "stand",
                              "action": str(ch.get("action", ""))[:200]})
        sfx = []
        for e in sc.get("sfx") or []:
            if isinstance(e, dict) and e.get("name") in SFX:
                try:
                    sfx.append({"name": e["name"], "at": max(float(e.get("at", 0)), 0.0)})
                except (TypeError, ValueError):
                    pass
        scenes.append({
            "setting": str(sc.get("setting")) if str(sc.get("setting")) in sids else settings[0]["id"],
            "narration": narr[:1200],
            "speaker": str(sc.get("speaker")) if str(sc.get("speaker")) in cids else None,
            "characters": chars,
            "badges": [re.sub(r"^\+(?=\d+\s*(BC|BCE|CE|AD)\b)", "", str(b).strip())[:40]
                       for b in (sc.get("badges") or []) if str(b).strip()][:3],
            "title": str(sc.get("title") or "")[:60],
            "camera": sc.get("camera") if sc.get("camera") in ("static", "push_in", "pan_left", "pan_right") else "push_in",
            "transition_in": sc.get("transition_in") if sc.get("transition_in") in ("cut", "fade", "whip") else "fade",
            "sfx": sfx,
        })
    if not scenes:
        raise RuntimeError("DeepSeek did not return any scenes. Try again or simplify the topic.")
    ensure_on_screen(cast, scenes, log)
    _distinct_colours(cast)
    out = {"title": str(data.get("title", ""))[:100], "cast": cast, "settings": settings, "scenes": scenes,
           "music_mood": data.get("music_mood") if data.get("music_mood") in MOODS else "curious"}
    log(f"Script: \"{out['title']}\" - {len(scenes)} scenes, {len(cast)} characters, {len(settings)} settings")
    return out


DIRECT_PROMPT = """You are the art director of a polished animated YouTube channel. The narration of the video "{title}" is final; plan what the viewer SEES for every scene.
Narration, scene by scene (number: words):
{lines}
Existing cast (id = name): {cast}
Reply with JSON only:
{{"settings": [{n_min}-{n_max} different places: {{"id": "s1", "description": "rich description for an illustrator: the exact place and era, time of day, one distinct colour palette and light, what is in the foreground, middle and far background, strong depth. No people. An open clear floor across the lower part."}}],
 "new_cast": [up to 6 extra characters this story really needs (soldiers, a merchant, a scribe, a crowd, a named ruler): {{"id": "n1", "name", "description", "kind": human|crowd|bird|seal|animal, "look": {{"body": "#rrggbb" bright clothes colour, never black or dark grey, "accent": "#rrggbb", "accessory": one of {acc}, "height": 0.8-1.2}}}}],
 "scenes": [one entry per narration line, in order: {{"i": number, "setting": "s1", "camera": static|push_in|pan_left|pan_right, "speaker": the on-screen host's id or null, "characters": [{{"id", "x": 0.12-0.88, "pose": one of {poses}, "action": what they visibly do}}]}}]}}
Rules:
- Show what the words talk about: a named person or group appears on screen, and the place matches the subject (a line about a railway happens at a railway, a battle happens on that battlefield).
- Every setting looks clearly different from the others in colour and time of day. Use one setting for at most {max_use} scenes, and never more than 3 scenes in a row.
- At least half of the scenes have 2 to 4 characters interacting, spread at least 0.2 apart in x. Never an empty scene.
- speaker is only the narrator or host, when on screen: other characters do not speak the narrator's words.
- Vary the poses across scenes; do not make everyone point."""


DIRECT_PART_PROMPT = """You are the art director of an animated YouTube video. Plan what the viewer SEES for these scenes.
Narration (number: words):
{lines}
Use ONLY these places: {places}
and ONLY these characters: {cast}
Reply with JSON only: {{"scenes": [{{"i": number, "setting": place id, "camera": static|push_in|pan_left|pan_right, "speaker": the on-screen host's id or null, "characters": [{{"id", "x": 0.12-0.88, "pose": one of {poses}, "action"}}]}}]}}
Show what the words talk about. Change place often, with never more than 3 scenes in a row in one place. At least half of the scenes have 2 to 4 characters interacting, 0.2 apart in x. Never an empty scene. Vary the poses."""


def direct_scenes(script, log):
    """Art director pass: more places and characters, matched to the narration, which stays
    word for word (so a recorded voice is kept). Replaces the sets, so they are redrawn.
    Long videos are planned part by part with the same places and people."""
    from . import design
    scenes = script["scenes"]
    n = len(scenes)
    n_max = max(4, min(16, n // 4))
    lines = "\n".join(f"{i}: {sc['narration']}" for i, sc in enumerate(scenes))
    cast_txt = "; ".join(f'{c["id"]} = {c["name"]}' for c in script["cast"])
    log(f"Art direction: planning up to {n_max} places and the characters of each scene...")
    size = 50
    first = range(0, min(size, n))
    data = design._json_call([{"type": "text", "text": DIRECT_PROMPT.format(
        title=script["title"], lines=lines[:40000], cast=cast_txt, n_min=max(3, n_max // 2), n_max=n_max,
        max_use=max(4, n // max(n_max - 2, 1)), acc=ACCESSORIES, poses=POSES) + (
            f"\nThis video is long: in \"scenes\" give only scenes {first[0]} to {first[-1]}." if n > size else "")}],
        max_tokens=24000, temperature=0.7)
    settings = [x for x in data.get("settings") or [] if isinstance(x, dict) and x.get("id") and x.get("description")]
    plan = {}

    def take(d_):
        for e in d_.get("scenes") or []:
            try:
                plan[int(e["i"])] = e
            except (KeyError, TypeError, ValueError):
                pass

    take(data)
    have = {c["id"] for c in script["cast"]}
    extra = [c for c in data.get("new_cast") or [] if isinstance(c, dict) and c.get("id") and c["id"] not in have][:6]
    if n > size and len(settings) >= 2:
        all_cast = "; ".join(f'{c["id"]} = {c["name"]}' for c in script["cast"] + extra)
        places = "; ".join(f'{x["id"]} = {str(x["description"])[:90]}' for x in settings)

        def part(a0):
            rows = "\n".join(f"{i}: {scenes[i]['narration']}" for i in range(a0, min(a0 + size, n)))
            return design._json_call([{"type": "text", "text": DIRECT_PART_PROMPT.format(
                lines=rows, cast=all_cast, places=places, poses=POSES)}], max_tokens=16000, temperature=0.7)

        for d_ in design.run_parallel(part, list(range(size, n, size)), workers=4):
            take(d_)
    if len(settings) < 2 or len(plan) < n * 0.7:
        log("Art direction came back incomplete - keeping the original plan")
        return dict(script, directed=True, keep_sets=True)
    merged = []
    for i, sc in enumerate(scenes):
        e = plan.get(i)
        if e:
            sc = dict(sc, setting=e.get("setting"), camera=e.get("camera", sc["camera"]), speaker=e.get("speaker"),
                      characters=(e.get("characters") or [])[:4])
        else:  # a scene the AI skipped keeps its people, in the first new place
            sc = dict(sc, setting=settings[0]["id"])
        merged.append(sc)
    out = clean_script({"title": script["title"], "cast": script["cast"] + extra, "settings": settings,
                        "scenes": merged, "music_mood": script.get("music_mood")}, log)
    out["directed"] = True
    return out


# ----------------------------------------------------------------- 3. voice
async def _tts_one(text, voice, out_mp3):
    import edge_tts
    comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    words = []
    with open(out_mp3, "wb") as f:
        async for ch in comm.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"s": ch["offset"] / 1e7, "e": (ch["offset"] + ch["duration"]) / 1e7, "w": ch["text"]})
    return words


VOICE_PY = config.DATA_DIR / ".venv-voice" / "Scripts" / "python.exe"
OPENVOICE_CKPT = config.MODELS_DIR / "openvoice_v2"


def clone_available():
    return VOICE_PY.exists() and (OPENVOICE_CKPT / "converter" / "checkpoint.pth").exists()


def to_my_voice(d, n_scenes, ref, log, cb=None):
    """Converts every scene's narration into the user's voice (OpenVoice V2, CPU).
    Timing is unchanged, so word timings and lip-sync stay correct. Cached."""
    from .config import FFMPEG
    vdir, cdir = d / "voice", d / "voice_mine"
    cdir.mkdir(exist_ok=True)
    ref_wav = d / "my_voice_22k.wav"
    if not ref_wav.exists():  # any audio/video file -> mono 22 kHz wav, first 30 seconds
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(ref), "-t", "30", "-vn", "-ac", "1",
                        "-ar", "22050", str(ref_wav)], check=True)
    todo = []
    for i in range(n_scenes):
        src22 = cdir / f"scene_{i}_src.wav"
        if not (cdir / f"scene_{i}_src.wav").exists() or not (cdir / "out" / f"scene_{i}_src.wav").exists():
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(vdir / f"scene_{i}.wav"), "-ac", "1",
                            "-ar", "22050", str(src22)], check=True)
            todo.append(str(src22))
    if todo:
        log(f"Turning the narration into your voice ({len(todo)} scenes, OpenVoice on CPU)...")
        # stream its progress lines ("converted 3/17") into the progress bar
        p = subprocess.Popen([str(VOICE_PY), str(config.ROOT / "app" / "voice_clone.py"), "--ref", str(ref_wav),
                              "--ckpt", str(OPENVOICE_CKPT), "--out", str(cdir / "out"), *todo],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in p.stdout:
            m = re.match(r"converted (\d+)/(\d+)", line.strip())
            if m and cb:
                k, total = int(m.group(1)), int(m.group(2))
                cb(k / total, f"Your voice: scene {k}/{total}")
        err = p.stderr.read()
        if p.wait() != 0:
            raise RuntimeError("Voice cloning failed: " + err[-800:])
    return [cdir / "out" / f"scene_{i}_src.wav" for i in range(n_scenes)]


def narrate(scenes, voice, d, log, my_voice=None, cb=None):
    """One audio file per scene + word timings. Cached. With my_voice (a recording of the
    user), the narration is converted into that voice, keeping the same timing."""
    from . import audio_fx
    from .config import FFMPEG
    vdir = d / "voice"
    vdir.mkdir(exist_ok=True)
    out = []
    for i, sc in enumerate(scenes):
        wav = vdir / f"scene_{i}.wav"
        wj = vdir / f"scene_{i}.json"
        if not (wav.exists() and wj.exists()):
            mp3 = vdir / f"scene_{i}.mp3"
            for attempt in range(3):
                try:
                    words = asyncio.run(_tts_one(sc["narration"], voice, mp3))
                    break
                except Exception as e:
                    if attempt == 2:
                        raise RuntimeError(f"Voice failed for scene {i + 1}: {e}")
                    time.sleep(2)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1", "-ar", str(audio_fx.SR),
                            str(wav)], check=True)
            _save(wj, words)
        out.append({"wav": wav, "words": _load(wj)})
    if my_voice:
        if not clone_available():
            log("'Use my voice' needs the voice tools (.venv-voice + models/openvoice_v2) - using the normal voice.")
        else:
            mine = to_my_voice(d, len(scenes), my_voice, log, cb)
            for o, m in zip(out, mine):
                o["wav"] = m
    for o in out:
        x = audio_fx.read_wav(o["wav"])  # resampled to the mix rate
        o["audio"], o["dur"] = x, len(x) / audio_fx.SR
    log(f"Narration recorded: {sum(o['dur'] for o in out):.0f} seconds" + (" (in your voice)" if my_voice else ""))
    return out


# ----------------------------------------------------------------- 4. action
ACT_PROMPT = """You are a 2D animator. Animate ONE scene of an original explainer video.
Setting: {setting}
Scene length: {dur:.1f} seconds. Narration (with the time each word is spoken): {words}
Characters in the scene (simple stick-style figures):
{chars}
The floor is at y = 0.86 of the image height. A standing character of height h has its HIP at y = 0.86 - 0.34*h. Typical h: 0.45 (wide shot) to 0.7 (close).
Write KEYFRAMES that act out the scene and match the words, like a good animator: a new pose on key words about every 1.5-2.5 seconds, reacting to each other, turning to face whoever matters. Do not hold one pose, and do not repeat the same pose twice in a row. Use "point" only when there is something to point at.
In scenes longer than 6 seconds, at least one character really travels: give two keys with different x, 1-3 seconds apart, and pose "walk" (or "run"). The app animates the legs from the travel, so walk only when x changes, and face the way they go. Characters stay at least 0.15 apart.
Each key: {{"t", "x" (0.08-0.92, the feet), "h" (body size: 0.62 normal, up to 0.78 for the main character of a close moment), "facing" (1 right / -1 left), "head" (tilt in degrees, -12 to 12), "pose" (one of {poses}), "mood" (neutral, happy, laughing, angry, sad, surprised, smug, scared)}}. The limbs and feet on the floor are handled for you.
Do not draw lines, pointers or beams.
Reply with JSON only: {{"keys": {{"<character id>": [ ... ]}}}}"""


def _check_credit(e):
    """When no provider can answer the job stops with a clear message: without this it
    would render a whole video with blank sets and characters that stand still."""
    if isinstance(e, providers.AllProvidersFailed):
        raise e


def _clean_key(kf, kind):
    """Limbs always come from the clean pose library, never from raw angles."""
    from . import animator
    if kind != "human":
        return animator.apply_pose(kf)
    kf = dict(kf)
    pose = str(kf.get("pose", "stand"))
    f = 1 if float(kf.get("facing", 1) or 1) >= 0 else -1
    kf.update({k: list(v) for k, v in animator.pose_limbs(pose, f).items()})
    if pose == "lie":  # arms rest along the body, not through the floor
        kf.update(arm_l=[f * 82, 0], arm_r=[f * 78, 0])
    kf["lean"] = {"bow": -f * 34, "lie": 90 * f, "kneel": -f * 6, "read": -f * 4, "write": -f * 6}.get(pose, 0)
    return kf


def _spring(store, key, target, ts, dt):
    """Limbs follow their pose through a soft spring: a small overshoot, then they settle."""
    st = store.get(key)
    if st is None:
        store[key] = [target, 0.0, ts]
        return target
    for _ in range(min(int(round((ts - st[2]) / dt)), 8)):
        st[1] += (17.0 ** 2 * (target - st[0]) - 2 * 0.58 * 17.0 * st[1]) * dt
        st[0] += st[1] * dt
    st[2] = ts
    return st[0]


def animate_scene(sc, setting_desc, cast_by_id, dur, words):
    from . import animator, design
    chars = "\n".join(f'- id "{c["id"]}" {cast_by_id[c["id"]]["name"]} ({cast_by_id[c["id"]]["kind"]}): '
                      f'starts around x={c["x"]:.2f}, pose {c["pose"]}; action: {c["action"]}' for c in sc["characters"])
    wtxt = " ".join(f'{w["w"]}({w["s"]:.1f})' for w in words)[:3000]
    data = design._json_call([{"type": "text", "text": ACT_PROMPT.format(
        setting=setting_desc[:300], dur=dur, words=wtxt, chars=chars or "(none)", poses=POSES)}],
        max_tokens=6000, temperature=0.4)
    keys = {}
    for cid, ks in (data.get("keys") or {}).items():
        if str(cid) not in cast_by_id or str(cid) not in [c["id"] for c in sc["characters"]]:
            continue
        good = []
        for kf in ks or []:
            if isinstance(kf, dict):
                try:
                    kf["t"] = min(max(float(kf.get("t", 0)), 0.0), dur)
                    kf["x"] = min(max(float(kf.get("x", 0.5)), 0.08), 0.92)
                    good.append(_clean_key(kf, cast_by_id[str(cid)]["kind"]))
                except (TypeError, ValueError):
                    pass
        if good:
            good.sort(key=lambda k: k["t"])
            for k0, k1 in zip(good, good[1:]):  # nobody teleports: limit the walking speed
                reach = max(k1["t"] - k0["t"], 0.05) * 0.3
                k1["x"] = min(max(k1["x"], k0["x"] - reach), k0["x"] + reach)
            keys[str(cid)] = _liven(good, dur, cast_by_id[str(cid)]["kind"], hash(str(cid)) % 5)
    # characters DeepSeek skipped still appear, standing where the script put them
    for c in sc["characters"]:
        if c["id"] not in keys:
            keys[c["id"]] = [_clean_key({"t": 0, "x": c["x"], "h": 0.62, "facing": 1, "pose": c["pose"],
                                         "mood": "neutral"}, cast_by_id[c["id"]]["kind"])]
    _ensure_travel(keys, sc, cast_by_id, dur)
    return {"keys": keys}


GESTURES = ["open_arms", "think", "shrug", "wave", "hips", "fist", "point"]


def _liven(keys, dur, kind, seed):
    """Nobody freezes: long holds get a natural gesture in the middle."""
    if kind != "human":
        return keys
    out, n = [], seed
    timeline = keys + [dict(keys[-1], t=dur)]
    for a, b in zip(timeline, timeline[1:]):
        out.append(a)
        t = a["t"] + 2.2
        while t < b["t"] - 1.2:
            pose = GESTURES[n % len(GESTURES)]
            n += 1
            out.append(_clean_key(dict(a, t=t, pose=pose), kind))
            t += 2.2
    return out


def _ensure_travel(keys, sc, cast_by_id, dur):
    """In longer scenes someone actually walks, so the shot is never just people standing."""
    if dur < 6 or any(max(k["x"] for k in q) - min(k["x"] for k in q) > 0.08 for q in keys.values()):
        return
    who = next((c for c in sc["characters"] if cast_by_id[c["id"]]["kind"] == "human" and c["id"] in keys), None)
    if not who:
        return
    q = keys[who["id"]]
    x0 = q[0]["x"]
    step = 0.2 if x0 < 0.5 else -0.2
    f = 1 if step > 0 else -1
    t0 = min(dur * 0.3, 2.5)
    before = [k for k in q if k["t"] < t0]
    after = [dict(k, x=x0 + step) for k in q if k["t"] > t0 + 2.0]
    q[:] = before + [_clean_key(dict(q[0], t=t0, x=x0, pose="walk", facing=f), "human"),
                     _clean_key(dict(q[0], t=t0 + 2.0, x=x0 + step, pose="stand", facing=f), "human")] + after


# ----------------------------------------------------------------- text (any language) via resvg
_TEXT_CACHE = {}


def text_surface(spans, size, weight="bold", stroke=True):
    """spans: list of (text, '#rrggbb'). Renders with resvg so Urdu/Hindi/Arabic are shaped
    correctly. Returns a cairo surface (cached)."""
    import html

    import cairo
    import resvg_py
    key = (tuple(spans), size, weight, stroke)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    est_w = int(sum(len(t) for t, _ in spans) * size * 0.75 + size * 2)
    h = int(size * 1.9)
    tspans = "".join(f'<tspan fill="{c}">{html.escape(t)}</tspan>' for t, c in spans)
    st = f' stroke="#101014" stroke-width="{size * 0.16:.1f}" paint-order="stroke" stroke-linejoin="round"' if stroke else ""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{est_w}" height="{h}" viewBox="0 0 {est_w} {h}">'
           f'<text x="{est_w / 2}" y="{size * 1.3:.1f}" text-anchor="middle" font-family="Segoe UI, Nirmala UI, '
           f'Arial, sans-serif" font-size="{size}" font-weight="{weight}"{st}>{tspans}</text></svg>')
    import io
    png = resvg_py.svg_to_bytes(svg_string=svg)
    surf = cairo.ImageSurface.create_from_png(io.BytesIO(png))
    if len(_TEXT_CACHE) > 400:
        _TEXT_CACHE.clear()
    _TEXT_CACHE[key] = surf
    return surf


def paint_text(ctx, surf, cx, cy, alpha=1.0, scale=1.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(scale, scale)
    ctx.set_source_surface(surf, -surf.get_width() / 2, -surf.get_height() / 2)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# ----------------------------------------------------------------- the whole mode
def title_for_search(material):
    return material.split("\n", 1)[0].replace("Original video title:", "").strip()[:120]


def run(job, d, opts, prog, log):
    import cairo

    from . import animator, audio_fx, design, explainer, media
    from .sprites import background_surface
    t_start = time.time()
    if not providers.configured():
        raise RuntimeError("Add an AI provider in Settings first.")
    lang = opts.get("language", "en")
    voice = VOICES.get((lang, opts.get("voice", "male")), VOICES[("en", "male")])
    minutes = float(opts.get("minutes") or 2)
    shorts = opts.get("aspect") == "9:16"
    H = int(opts.get("height") or 1080)
    W = int(round(H * 16 / 9)) // 2 * 2
    if shorts:
        W, H = int(H * 9 / 16) // 2 * 2, H
    ddir = d / "design"
    ddir.mkdir(exist_ok=True)

    # 1. script
    prog.stage("prepare")
    polish = False
    spath = d / "script.json"
    if opts.get("extend_to") and spath.exists():  # "Add more scenes" on a finished video
        want = float(opts["extend_to"])
        log(f"Adding scenes so the video is about {want:g} minutes...")
        for _ in range(3):  # each pass gets closer; stop once it is long enough
            try:
                if not extend_job(d, want, log):
                    break
            except Exception as e:
                _check_credit(e)
                raise
            if _words(_load(spath)["scenes"]) >= want * 140 * 0.92:
                break
        opts.pop("extend_to", None)
        opts["minutes"] = want
        job.update(options=opts)
        polish = not _load(spath).get("directed")
    script = _load(spath)
    if script is None:
        kind = opts.get("source", "topic")
        material = opts.get("topic") or opts.get("script_text") or ""
        if kind == "youtube":
            log("Reading the YouTube video's transcript (research only - nothing is copied)...")
            if config.STORE:
                title, material = video_title(opts["youtube_url"]), opts.get("transcript", "")
                if not title and not material:
                    raise RuntimeError("Could not read that video's title. Check the link, or paste its transcript.")
            else:
                title, material = youtube_material(opts["youtube_url"], d, log)
            material = f"Original video title: {title}\n{material}"
        if kind != "script":
            log("Researching the topic...")
            found, sources = research.gather(opts.get("topic") or title_for_search(material), lang, log)
            if found:
                material += ("\n\nVERIFIED RESEARCH. Base every fact, date and name on this; leave out anything it "
                             "does not support:\n" + found)
                _save(d / "sources.json", sources)
        log("The AI is writing the script...")
        try:
            script = write_script(kind, material, lang, minutes, opts.get("description", ""), log)
        except Exception as e:
            _check_credit(e)
            raise
        polish = True
    if polish:
        try:
            script = direct_scenes(script, log)
        except Exception as e:
            _check_credit(e)
            raise
        if not script.pop("keep_sets", False):
            for f in list(ddir.glob("set_*.svg")) + list(ddir.glob("setpreview_*.png")):
                f.unlink()
        (d / "scene_anim.json").unlink(missing_ok=True)
    _save(spath, script)
    job.update(script_title=script["title"])
    cast_by_id = {c["id"]: c for c in script["cast"]}
    set_by_id = {s["id"]: s for s in script["settings"]}
    scenes = script["scenes"]
    air = {x["id"]: explainer.ambient_kind(x["description"]) for x in script["settings"]}

    # 2. sets (each drawn once, reused by every scene that happens there)
    prog.stage("design")
    used = sorted({sc["setting"] for sc in scenes})

    def make_set(sid):
        bp, fp = ddir / f"set_{sid}_bg.svg", ddir / f"set_{sid}_fg.svg"
        bg, fg = (bp.read_text(encoding="utf-8") if bp.exists() else None), None
        for attempt in range(0 if bg else 2):
            try:
                bg, fg = design.draw_set_text(set_by_id[sid]["description"], W, H, opts.get("description", ""), log,
                                              review=int(config.REVIEW_PASSES or 0) > 0)
            except Exception as e:
                _check_credit(e)
                log(f"Set {sid} failed ({e})")
            if bg:
                break
        if bg:
            bp.write_text(bg, encoding="utf-8")
        if fg:
            fp.write_text(fg, encoding="utf-8")
        if bg:
            small = background_surface(bg, W // 3, H // 3)
            small.write_to_png(str(ddir / f"setpreview_{used.index(sid)}.png"))
        sets_done[0] += 1
        design_cb(sets_done[0] / len(used), f"Drawing the places: {sets_done[0]} of {len(used)} finished")

    sets_done = [0]
    design_cb = prog.cb("design")
    design_cb(0, f"Drawing the places: 0 of {len(used)} finished")
    design.run_parallel(make_set, used, workers=4)
    bgs, fgs = {}, {}
    for sid in used:
        bp, fp = ddir / f"set_{sid}_bg.svg", ddir / f"set_{sid}_fg.svg"
        if bp.exists():
            bgs[sid] = background_surface(bp.read_text(encoding="utf-8"), W, H)
        if fp.exists():
            fgs[sid] = background_surface(fp.read_text(encoding="utf-8"), W, H)
    for i, c in enumerate(script["cast"]):
        (ddir / f"character_{i}.png").write_bytes(animator.stick_preview(c["look"]))
    log(f"{len(bgs)} sets drawn, {len(script['cast'])} characters (each defined once and reused)")

    # 3. voice
    prog.stage("audio")
    my_voice = (d / opts["voice_ref"]) if opts.get("voice_ref") else None
    log("Recording the narration")
    voiced = narrate(scenes, voice, d, log, my_voice=my_voice, cb=prog.cb("audio"))

    # timeline: each scene = its narration + a little breathing room
    gap = 0.45
    starts, t = [], 0.0
    for v in voiced:
        starts.append(t)
        t += v["dur"] + gap
    total = t + 0.6

    # 4. action
    prog.stage("direct")
    apath = d / "scene_anim.json"
    anim = {int(k): v for k, v in (_load(apath) or {}).items()}
    todo = [i for i in range(len(scenes)) if i not in anim]
    done = [0]

    def work(i):
        try:
            res = animate_scene(scenes[i], set_by_id[scenes[i]["setting"]]["description"], cast_by_id,
                                voiced[i]["dur"] + gap, voiced[i]["words"])
        except Exception as e:
            _check_credit(e)
            log(f"Scene {i + 1}: animation failed ({e}); characters stand in place.")
            res = animate_scene_fallback(scenes[i])
        done[0] += 1
        prog.cb("direct")(done[0] / max(len(todo), 1), f"Acting out scene {done[0]} of {len(todo)}")
        return res

    for i, res in zip(todo, design.run_parallel(work, todo, workers=4)):
        anim[i] = res
    _save(apath, {str(k): v for k, v in anim.items()})

    # 5. sound: narration + generated music + effects
    prog.stage("lipsync")
    sr = audio_fx.SR
    voice_track = np.zeros(int(total * sr) + sr, np.float32)
    for st, v in zip(starts, voiced):
        a = int(st * sr)
        seg = voice_track[a:a + len(v["audio"])]
        seg += v["audio"][:len(seg)]
    effects = []
    for i, (st, sc) in enumerate(zip(starts, scenes)):
        if i > 0 and sc["transition_in"] in ("whip", "fade"):
            effects.append((max(st - 0.2, 0), audio_fx.sfx("whoosh")))
        for k in range(len(sc["badges"])):
            effects.append((st + 0.5 + k * 0.35, audio_fx.sfx("pop")))
        if sc["title"]:
            effects.append((st + 0.1, audio_fx.sfx("swoosh_up")))
        for e in sc["sfx"]:
            effects.append((st + e["at"], audio_fx.sfx(e["name"])))
    music_track = audio_fx.music(script["music_mood"], total + 1)
    mixed = audio_fx.mix(voice_track, music_track, effects,
                         music_level=float(opts.get("music_level", 0.5)))
    wav = d / "final_audio.wav"
    audio_fx.write_wav(wav, mixed)
    log(f"Sound: narration + '{script['music_mood']}' music + {len(effects)} sound effects")

    # per-frame helpers: which scene, mouth (spoken words), captions
    fps = 30.0
    n = int(total * fps)
    from .lipsync import frame_energy
    import wave as _wave
    tmp16 = d / "voice16.wav"
    with _wave.open(str(tmp16), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(voice_track, -1, 1) * 32767).astype(np.int16).tobytes())
    energy = frame_energy(tmp16, fps, n)
    words_abs = []
    for i, (st, v) in enumerate(zip(starts, voiced)):
        for w in v["words"]:
            words_abs.append((st + w["s"], st + max(w["e"], w["s"] + 0.08), w["w"], i))
    captions = make_captions(words_abs)

    # 6. edit + render
    prog.stage("render")
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    other = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    octx = cairo.Context(other)
    sky = cairo.LinearGradient(0, 0, 0, H)
    sky.add_color_stop_rgb(0, 0.55, 0.6, 0.78)
    sky.add_color_stop_rgb(1, 0.9, 0.82, 0.75)
    fs = H / 1080.0 if not shorts else W / 1080.0
    springs = {}

    def draw_scene(c, i, ts, t_abs, f):
        sc = scenes[i]
        dur = voiced[i]["dur"] + gap
        u = animator._ease(min(max(ts / dur, 0), 1))
        # every scene frames its set a little differently, so a place never looks the same twice
        z0 = 1.03 + 0.08 * ((i * 0.618) % 1)
        side = (((i * 0.382) % 1) - 0.5) * 2 * (z0 - 1) * 0.42 * W
        z, dx = z0 + 0.012 * u, side
        if sc["camera"] == "push_in":
            z = z0 + 0.07 * u
        elif sc["camera"] == "pan_left":
            z, dx = max(z0, 1.08), (0.5 - u) * 0.06 * W
        elif sc["camera"] == "pan_right":
            z, dx = max(z0, 1.08), (u - 0.5) * 0.06 * W
        c.save()
        c.translate(W / 2 + dx, H * 0.62)
        c.scale(z, z)
        c.translate(-W / 2, -H * 0.62)
        bg = bgs.get(sc["setting"])
        if bg is not None:
            c.set_source_surface(bg, 0, 0)
        else:
            c.set_source(sky)
        c.paint()
        people = []
        talking_now = any(a <= t_abs <= b for a, b, _, si in words_abs if si == i)
        for cid, ks in anim.get(i, {}).get("keys", {}).items():
            ch = cast_by_id.get(cid)
            if not ch or not ks:
                continue
            kv = animator.sample(ks, ts)
            hh = min(max(kv["h"], 0.56), 0.8) * H
            seed = sum(ord(ch_) for ch_ in cid) % 17
            # walking comes from real travel: the feet stride with the distance covered
            vel = (animator.sample(ks, ts + 0.06)["x"] - animator.sample(ks, ts - 0.06)["x"]) / 0.12 * W / hh
            move = min(abs(vel) / 0.3, 1.0)
            facing = kv["facing"]
            arms = kv["pose"] not in ("sit", "crouch", "kneel", "lie", "bow")
            if move > 0.02 and kv["pose"] not in ("sit", "lie"):
                if move > 0.4:
                    facing = 1 if vel > 0 else -1
                amp, stride = (40, 0.8) if abs(vel) > 1.3 else (26, 0.6)
                phase = 2 * math.pi * kv["x"] * W / (2 * stride * hh)
                swing, lift = math.sin(phase) * move, math.cos(phase)
                kv["leg_l"] = (amp * swing, -facing * 38 * max(lift, 0) * move)
                kv["leg_r"] = (-amp * swing, -facing * 38 * max(-lift, 0) * move)
                kv["lean"] -= facing * (2 + 3 * min(abs(vel), 1.5)) * move
                if kv["pose"] in ("stand", "walk", "run"):
                    kv["arm_l"] = (kv["arm_l"][0] - amp * 0.7 * swing, kv["arm_l"][1] + facing * 18 * move)
                    kv["arm_r"] = (kv["arm_r"][0] + amp * 0.7 * swing, kv["arm_r"][1] + facing * 18 * move)
            is_speaker = sc["speaker"] == cid and talking_now
            talk = min(float(energy[min(f, n - 1)]) * 1.3, 1.0) if is_speaker else 0.0
            blink_v = animator.blink(t_abs, seed % 7)
            # life: breathing, a slow sway, nods and hand beats while speaking
            sway = math.sin(t_abs * 1.55 + seed)
            kv["lean"] += 0.9 * math.sin(t_abs * 0.7 + seed)
            kv["head"] += 1.8 * math.sin(t_abs * 0.9 + seed * 2) + 4.5 * talk * math.sin(t_abs * 6.7)
            if arms:
                beat = 6 * talk * math.sin(t_abs * 4.3 + seed)
                kv["arm_l"] = (kv["arm_l"][0] - 1.8 * sway - beat, kv["arm_l"][1] - beat * 0.7)
                kv["arm_r"] = (kv["arm_r"][0] + 1.8 * sway + beat, kv["arm_r"][1] + beat * 0.7)
            dt_ = 1.0 / fps
            lim = {}
            for name in ("arm_l", "arm_r"):
                lim[name] = tuple(_spring(springs, (i, cid, name, j), kv[name][j], ts, dt_) for j in (0, 1))
            lean_, head_ = (_spring(springs, (i, cid, k_), kv[k_], ts, dt_) for k_ in ("lean", "head"))
            ground = H * 0.86
            hop = hh * 0.2 * abs(math.sin(t_abs * 3.4)) if kv["pose"] == "jump" else 0.0
            people.append((hh, dict(ch["look"], kind=ch["kind"], name=ch["name"]), dict(
                x=kv["x"] * W, y=ground - hop, ground=ground, size=hh, facing=facing, lean=lean_,
                head=head_, arm_l=lim["arm_l"], arm_r=lim["arm_r"], leg_l=kv["leg_l"], leg_r=kv["leg_r"],
                blink=blink_v, talk=talk, smile=kv["smile"], brow_down=kv["brow_down"], brow_up=kv["brow_up"],
                look_x=0.35 * facing, seat=kv["pose"] == "sit", wide=explainer.is_beast(ch["name"]))))
        # people never stand inside each other
        row = sorted((p[2] for p in people if p[1].get("kind") != "crowd"), key=lambda st_: st_["x"])
        for a_, b_ in zip(row, row[1:]):
            gap_ = (0.62 if a_["wide"] or b_["wide"] else 0.4) * max(a_["size"], b_["size"]) - (b_["x"] - a_["x"])
            if gap_ > 0:
                a_["x"] -= gap_ / 2
                b_["x"] += gap_ / 2
        # a crowd stands behind everyone, then big animals; small animals are animal-sized
        for _, look, st in sorted(people, key=lambda p: (p[1].get("kind") != "crowd", not p[2]["wide"], p[0])):
            kind = look.get("kind", "human")
            if kind == "crowd":
                size = st["size"] * 0.56
                reach = size * 1.25
                explainer.draw_crowd(c, dict(st, size=size, y=H * 0.8, ground=H * 0.8,
                                             x=min(max(st["x"], reach), W - reach)), look)
            elif explainer.is_beast(look.get("name")):
                explainer.draw_beast(c, dict(st, size=st["size"] * 0.82), look)
            elif kind != "human":
                explainer.draw_animal(c, kind, dict(st, size=st["size"] * 0.42), look)
            else:
                explainer.StickCharacter(c, look).draw(st)
        fg = fgs.get(sc["setting"])
        if fg is not None:
            c.save()
            c.translate(dx * 0.35, 0)  # the foreground slides a little more: depth
            c.set_source_surface(fg, 0, 0)
            c.paint()
            c.restore()
        explainer.ambient(c, air.get(sc["setting"], "motes"), t_abs, W, H)
        c.restore()
        # graphics on top: badges (pop in), title card
        for k, b in enumerate(sc["badges"]):
            t0 = 0.5 + k * 0.35
            if ts >= t0:
                p = min((ts - t0) / 0.25, 1.0)
                sc_ = 0.6 + 0.4 * animator._ease(p) + 0.08 * math.sin(min(p, 1) * math.pi)
                surf_b = text_surface([(b, "#ffffff")], int(44 * fs))
                bx = W - 40 * fs - surf_b.get_width() * 0.5
                by = 70 * fs + k * 80 * fs
                # card behind the text
                c.save()
                c.translate(bx, by)
                c.scale(sc_, sc_)
                wb, hb = surf_b.get_width() * 0.9, 64 * fs
                c.rectangle(-wb / 2, -hb / 2, wb, hb)
                c.set_source_rgba(0.95, 0.35, 0.28, 0.95)
                c.fill()
                c.restore()
                paint_text(c, surf_b, bx, by, alpha=p, scale=sc_)
        if sc["title"] and ts < 3.2:
            a = min(ts / 0.3, 1.0) * min((3.2 - ts) / 0.4, 1.0)
            paint_text(c, text_surface([(sc["title"], "#ffffff")], int(84 * fs)), W / 2, H * 0.13, alpha=a,
                       scale=0.9 + 0.1 * animator._ease(min(ts / 0.4, 1)))

    out = d / "output.tmp.mp4"
    enc = media.open_encoder(out, W, H, fps, wav, n)
    trans = 0.35
    cb = prog.cb("render")
    try:
        for f in range(n):
            if job.cancelled:
                raise RuntimeError("Cancelled")
            t_abs = f / fps
            i = max(0, int(np.searchsorted(starts, t_abs, side="right")) - 1)
            ts = t_abs - starts[i]
            draw_scene(ctx, i, ts, t_abs, f)
            # transition from the previous scene
            if i > 0 and ts < trans and scenes[i]["transition_in"] != "cut":
                draw_scene(octx, i - 1, t_abs - starts[i - 1], t_abs, f)
                p = animator._ease(ts / trans)
                if scenes[i]["transition_in"] == "whip":
                    ctx.save()
                    ctx.set_source_surface(other, -p * W, 0)
                    ctx.paint()
                    ctx.restore()
                else:
                    ctx.set_source_surface(other, 0, 0)
                    ctx.paint_with_alpha(1 - p)
            # captions: the current phrase, the spoken word highlighted
            cap = next((cp for cp in captions if cp["start"] <= t_abs <= cp["end"]), None)
            if cap and opts.get("captions", True):
                spans = []
                for j, (a, b, w) in enumerate(cap["words"]):
                    spans.append((w + " ", "#ffd84a" if a <= t_abs <= b + 0.05 else "#ffffff"))
                cs = text_surface(spans, int(54 * fs))
                paint_text(ctx, cs, W / 2, H * (0.94 if not shorts else 0.78))
            explainer.post_process(surf)
            surf.flush()
            enc.stdin.write(surf.get_data())
            if f % 30 == 0:
                cb(f / max(n, 1), f"Drawing frame {f}/{n}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="ignore")
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed: " + err[-1500:])
    except BaseException:
        enc.kill()
        raise
    shutil.move(out, d / "output.mp4")
    sources = _load(d / "sources.json") or []
    refs = "\n\nSources:\n" + "\n".join(f"{x['title']}: {x['url']}" for x in sources) if sources else ""
    (d / "youtube.txt").write_text(f"{script['title']}\n\n" + "\n\n".join(s["narration"] for s in scenes) + refs,
                                   encoding="utf-8")
    log(f"Done: \"{script['title']}\" - {total:.0f}s, {n} frames, {W}x{H}, in {time.time() - t_start:.0f}s")
    job.update(progress=100, stage="done", message="Finished")


def animate_scene_fallback(sc):
    from . import animator
    keys = {}
    for c in sc["characters"]:
        keys[c["id"]] = [animator.apply_pose({"t": 0, "x": c["x"], "y": 0.69, "h": 0.62, "facing": 1,
                                              "pose": c["pose"], "mood": "happy"})]
    return {"keys": keys, "effects": []}


def make_captions(words_abs, max_words=5):
    caps, cur = [], []
    for a, b, w, si in words_abs:
        if cur and (len(cur) >= max_words or si != cur[-1][3] or a - cur[-1][1] > 0.7):
            caps.append(cur)
            cur = []
        cur.append((a, b, w, si))
    if cur:
        caps.append(cur)
    out = []
    for k, cp in enumerate(caps):
        end = cp[-1][1] + 0.3
        if k + 1 < len(caps):
            end = min(end, caps[k + 1][0][0])
        out.append({"start": cp[0][0], "end": end, "words": [(a, b, w) for a, b, w, _ in cp]})
    return out


# ----------------------------------------------------------------- make an existing video longer
INSERT_PROMPT = """You are expanding the YouTube video "{title}" (language: {lang}). It is too short; it needs about {need} MORE words of narration.
Here is part {part} of {parts} of the current script, scene by scene (number: narration):
{scenes}
Write NEW scenes to INSERT inside this part: more history, explanation, examples, surprising facts and small story moments that make it richer. Do not repeat what existing scenes say. Each new scene says after which existing scene number it goes ("after").
Use ONLY these characters (by id): {cast}
and ONLY these settings (by id): {settings}
EVERY new scene must show at least one character on screen.
Scene format: after, setting, narration (15-35 words), speaker (a character id on screen), characters ([{{"id", "x": 0.1-0.9, "pose": one of {poses}, "action"}}]), badges (short on-screen texts), title (usually ""), camera (static, push_in, pan_left, pan_right), transition_in (cut, fade, whip), sfx ([{{"name": one of {sfx}, "at"}}]).
Reply with JSON only: {{"scenes": [ ... ]}} totalling about {part_need} words."""


def extend_job(d, minutes, log=print):
    """Inserts new scenes into a finished story job until it is about `minutes` long.
    Existing scenes keep their recorded narration (files are renumbered, not redone)."""
    from . import design
    d = Path(d)
    script = _load(d / "script.json")
    old = script["scenes"]
    have = sum(len(sc["narration"].split()) for sc in old)
    target = int(minutes * 140)
    need = target - have
    if need < 60:
        log(f"Already about {have / 140:.1f} minutes - nothing to add")
        return 0
    lang = LANGS.get(_load(d / "job.json")["options"].get("language", "en"), "English")
    cast_txt = "; ".join(f'{c["id"]} = {c["name"]}' for c in script["cast"])
    set_txt = "; ".join(f'{x["id"]} = {x["description"][:80]}' for x in script["settings"])
    parts = max(2, min(24, round(need / 300)))
    bounds = [round(k * len(old) / parts) for k in range(parts + 1)]

    def write(k):
        a, b = bounds[k], bounds[k + 1]
        lines = "\n".join(f"{i}: {old[i]['narration']}" for i in range(a, b))
        data = design._json_call([{"type": "text", "text": INSERT_PROMPT.format(
            title=script["title"], lang=lang, need=need, part=k + 1, parts=parts, scenes=lines,
            cast=cast_txt, settings=set_txt, poses=POSES, sfx=SFX, part_need=need // parts)}],
            max_tokens=12000, temperature=0.75)
        out = []
        for sc in data.get("scenes") or []:
            try:
                after = int(sc.get("after", b - 1))
            except (TypeError, ValueError):
                after = b - 1
            out.append((min(max(after, a), b - 1), sc))
        return out

    new = [x for chunk in design.run_parallel(write, list(range(parts)), workers=4) for x in chunk]
    cleaned = clean_script({"cast": script["cast"], "settings": script["settings"],
                            "scenes": [sc for _, sc in new], "title": script["title"]}, lambda *_: None)["scenes"]
    # clean_script keeps order but may drop empty scenes: match them back by narration
    by_text = {sc["narration"]: sc for sc in cleaned}
    inserts = {}
    for after, sc in new:
        c = by_text.get(str(sc.get("narration", "")).strip()[:1200])
        if c:
            inserts.setdefault(after, []).append(c)
    merged, origin = [], []  # origin[i] = old index or None for a new scene
    for i, sc in enumerate(old):
        merged.append(sc)
        origin.append(i)
        for c in inserts.get(i, []):
            merged.append(c)
            origin.append(None)

    # renumber cached narration files so old scenes keep their recording
    moves = []
    for new_i, old_i in enumerate(origin):
        if old_i is not None and old_i != new_i:
            for sub, pat in (("voice", "scene_{}.wav"), ("voice", "scene_{}.json"), ("voice", "scene_{}.mp3"),
                             ("voice_mine", "scene_{}_src.wav"), ("voice_mine/out", "scene_{}_src.wav")):
                src = d / sub / pat.format(old_i)
                if src.exists():
                    moves.append((src, d / sub / pat.format(new_i)))
    for src, dst in moves:  # two steps so files never overwrite each other
        src.rename(src.with_name(src.name + ".move"))
    for src, dst in moves:
        src.with_name(src.name + ".move").rename(dst)
    for new_i, old_i in enumerate(origin):  # stale files where a NEW scene now sits
        if old_i is None:
            for sub, pat in (("voice", "scene_{}.wav"), ("voice", "scene_{}.json"), ("voice", "scene_{}.mp3"),
                             ("voice_mine", "scene_{}_src.wav"), ("voice_mine/out", "scene_{}_src.wav")):
                f = d / sub / pat.format(new_i)
                if f.exists() and not any(dst == f for _, dst in moves):
                    f.unlink()
    script["scenes"] = merged
    ensure_on_screen(script["cast"], script["scenes"], log)
    _save(d / "script.json", script)
    for f in ("scene_anim.json", "final_audio.wav", "voice16.wav"):
        (d / f).unlink(missing_ok=True)
    words = sum(len(sc["narration"].split()) for sc in merged)
    log(f"Added {len(merged) - len(old)} new scenes: {len(merged)} scenes, {words} words (~{words / 140:.1f} min)")
    return len(merged) - len(old)
