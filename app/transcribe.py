"""Speech-to-text with timestamps (faster-whisper, CPU int8)."""
from . import config, hardware

_model = None
_model_name = None


def transcribe(wav_path, duration, progress=None):
    global _model, _model_name
    from faster_whisper import WhisperModel

    name = config.WHISPER_MODEL or "base"
    if _model is None or _model_name != name:
        if progress:
            progress(0.0, f"Loading Whisper '{name}' (first run downloads it)")
        _model = None
        if config.USE_GPU != "off" and hardware.nvidia():
            try:
                _model = WhisperModel(name, device="cuda", compute_type="float16")
            except Exception:
                pass  # no CUDA libraries on this PC: the CPU is fine
        _model = _model or WhisperModel(name, device="cpu", compute_type="int8")
        _model_name = name
    from .lipsync import read_wav
    audio, _ = read_wav(wav_path)  # 16 kHz mono float32; avoids the PyAV decoder
    segments, info = _model.transcribe(audio, vad_filter=True, beam_size=1, word_timestamps=True)
    out = []
    for s in segments:
        words = [{"s": round(w.start, 2), "e": round(w.end, 2), "w": w.word.strip()} for w in (s.words or [])]
        out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})
        if progress and duration:
            progress(min(s.end / duration, 1.0), f"Transcribing {s.end:.0f}s / {duration:.0f}s")
    return {"language": info.language, "segments": out}


def release():
    """Frees the speech model's memory between jobs."""
    global _model, _model_name
    _model = _model_name = None


def captions(segments, max_words=7):
    """Short on-screen captions (a few words at a time) from word timestamps."""
    caps = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            caps.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        cur = []
        for w in words:
            cur.append(w)
            if len(cur) >= max_words or w["w"][-1:] in ".?!,":
                caps.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
                cur = []
        if cur:
            caps.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
    # keep each caption on screen until the next one starts (max +0.6s)
    for a, b in zip(caps, caps[1:]):
        a["end"] = min(b["start"], a["end"] + 0.6)
    return caps
