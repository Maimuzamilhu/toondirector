"""Every AI provider the app can use, tried in your order.

All of them speak the same "OpenAI-compatible" protocol, so a provider is just an address,
a key and a model. When one runs out of quota or credit the next one takes over, and the
job keeps going. If every provider is only limited for a minute, the job waits.
"""
import json
import re
import threading
import time

from . import config

PROVIDERS = {
    "deepseek": dict(name="DeepSeek", group="paid", url="https://api.deepseek.com", key_url="https://platform.deepseek.com/api_keys",
                     model="deepseek-flash", note="Reliable and low cost. Pay as you go."),
    "groq": dict(name="Groq", group="free", url="https://api.groq.com/openai/v1", key_url="https://console.groq.com/keys",
                 prefer=["llama-3.3-70b", "llama", "qwen", "gpt-oss"], max_out=8000, note="Free tier with a daily limit. Very fast."),
    "gemini": dict(name="Google Gemini", group="free", url="https://generativelanguage.googleapis.com/v1beta/openai",
                   key_url="https://aistudio.google.com/apikey", prefer=["flash-lite", "flash"], max_out=8000,
                   note="Free tier. Limits change often. Can see images."),
    "openrouter": dict(name="OpenRouter", group="free", url="https://openrouter.ai/api/v1", key_url="https://openrouter.ai/keys",
                       prefer=[":free"], note="Many free models. About 50 requests a day."),
    "cerebras": dict(name="Cerebras", group="free", url="https://api.cerebras.ai/v1", key_url="https://cloud.cerebras.ai",
                     prefer=["llama", "gpt-oss", "qwen"], max_out=8000, note="Free tier with short replies."),
    "mistral": dict(name="Mistral", group="free", url="https://api.mistral.ai/v1", key_url="https://console.mistral.ai/api-keys",
                    prefer=["mistral-small", "mistral-medium", "mistral-large"], note="Free experiment plan."),
    "sambanova": dict(name="SambaNova", group="free", url="https://api.sambanova.ai/v1", key_url="https://cloud.sambanova.ai/apis",
                      prefer=["llama", "deepseek"], max_out=8000, note="Free tier."),
    "nvidia": dict(name="NVIDIA NIM", group="free", url="https://integrate.api.nvidia.com/v1", key_url="https://build.nvidia.com",
                   prefer=["llama", "deepseek", "qwen"], note="Free for trying out models."),
    "cohere": dict(name="Cohere", group="free", url="https://api.cohere.ai/compatibility/v1", key_url="https://dashboard.cohere.com/api-keys",
                   prefer=["command-a", "command"], note="Free trial key."),
    "together": dict(name="Together AI", group="paid", url="https://api.together.xyz/v1", key_url="https://api.together.ai/settings/api-keys",
                     prefer=["llama", "deepseek"], note="Pay as you go."),
    "openai": dict(name="OpenAI", group="paid", url="https://api.openai.com/v1", key_url="https://platform.openai.com/api-keys",
                   prefer=["mini", "gpt"], note="Pay as you go. Can see images."),
    "anthropic": dict(name="Anthropic Claude", group="paid", url="https://api.anthropic.com/v1", key_url="https://console.anthropic.com/settings/keys",
                      prefer=["sonnet", "haiku", "opus"], note="Pay as you go. Can see images."),
    "xai": dict(name="xAI Grok", group="paid", url="https://api.x.ai/v1", key_url="https://console.x.ai",
                prefer=["grok"], note="Pay as you go."),
    "ollama": dict(name="Ollama", group="local", url="http://localhost:11434/v1", key_url="https://ollama.com/download",
                   nokey=True, note="Runs on this PC. Free and private. Best with a GPU."),
    "lmstudio": dict(name="LM Studio", group="local", url="http://localhost:1234/v1", key_url="https://lmstudio.ai",
                     nokey=True, note="Runs on this PC. Free and private. Best with a GPU."),
    "custom": dict(name="Custom server", group="local", url="", key_url="", note="Any OpenAI-compatible address."),
}
_PATH = config.DATA_DIR / "ai_providers.json"
_lock = threading.Lock()
_cool = {}     # provider id -> time it can be tried again
_used = {}     # (provider id, date) -> requests sent
_models = {}   # provider id -> model chosen automatically
on_switch = []  # callbacks(from_name, to_name, reason)
_last = [None]


class AllProvidersFailed(RuntimeError):
    pass


def load():
    try:
        st = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    st.setdefault("order", list(PROVIDERS))
    st.setdefault("items", {})
    for pid in PROVIDERS:
        if pid not in st["order"]:
            st["order"].append(pid)
        st["items"].setdefault(pid, {"key": "", "model": "", "url": "", "on": False})
    if not st.get("migrated"):  # the key from the first versions of the app
        st["migrated"] = True
        if config.DEEPSEEK_API_KEY:
            st["items"]["deepseek"].update(key=config.DEEPSEEK_API_KEY, on=True,
                                           model=config.DEEPSEEK_VISION_MODEL or "deepseek-flash")
            save(st)
    return st


def save(st):
    with _lock:
        _PATH.write_text(json.dumps(st, indent=1), encoding="utf-8")
    _models.clear()


def update(pid, **kw):
    st = load()
    st["items"][pid].update(kw)
    save(st)
    _cool.pop(pid, None)


def set_order(order):
    st = load()
    st["order"] = [p for p in order if p in PROVIDERS] + [p for p in st["order"] if p not in order]
    save(st)


def _info(pid, st=None):
    st = st or load()
    item, base = st["items"][pid], PROVIDERS[pid]
    return dict(base, id=pid, key=item.get("key", ""), model=item.get("model", ""), on=bool(item.get("on")),
                url=(item.get("url") or base["url"]).rstrip("/"))


def ready(st=None):
    """Providers switched on and usable, in your order."""
    st = st or load()
    out = []
    for pid in st["order"]:
        p = _info(pid, st)
        if p["on"] and p["url"] and (p["key"] or p.get("nokey")):
            out.append(p)
    return out


def configured():
    return bool(ready())


def used_today(pid):
    return _used.get((pid, time.strftime("%Y-%m-%d")), 0)


def _client(p, timeout=300):
    from openai import OpenAI
    return OpenAI(api_key=p["key"] or "local", base_url=p["url"], timeout=timeout, max_retries=0)


def list_models(pid):
    p = _info(pid)
    return sorted(m.id for m in _client(p, 30).models.list().data)


def _model(p):
    if p["model"]:
        return p["model"]
    if p["id"] in _models:
        return _models[p["id"]]
    if p.get("model") == "" and PROVIDERS[p["id"]].get("model"):
        return PROVIDERS[p["id"]]["model"]
    ids = list_models(p["id"])
    pick = next((m for pat in p.get("prefer", []) for m in ids if pat in m.lower()), ids[0] if ids else "")
    _models[p["id"]] = pick
    return pick


def test(pid):
    """A tiny request, so you see straight away that a key works. Returns (model, reply)."""
    p = _info(pid)
    model = _model(p)
    r = _client(p, 40).chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20)
    return model, (r.choices[0].message.content or "").strip()


def _status(e):
    return getattr(e, "status_code", None)


def _describe(e):
    code = _status(e)
    return {401: "the key was refused", 402: "out of credit", 403: "no access", 429: "limit reached"}.get(code) or \
        ("is not reachable" if code is None else f"error {code}")


def _call(p, content, max_tokens, temperature, json_mode, think):
    model = _model(p)
    kw = dict(model=model, messages=[{"role": "user", "content": content}], temperature=temperature,
              max_tokens=min(max_tokens, p.get("max_out", max_tokens)))
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    if p["id"] == "deepseek":
        kw["extra_body"] = {"thinking": {"type": "enabled" if think else "disabled"}}
    client = _client(p, 600 if think else 300)
    try:
        resp = client.chat.completions.create(**kw)
    except Exception as e:
        if _status(e) == 400 and "response_format" in kw:  # not every model has a JSON mode
            del kw["response_format"]
            resp = client.chat.completions.create(**kw)
        else:
            raise
    day = (p["id"], time.strftime("%Y-%m-%d"))
    _used[day] = _used.get(day, 0) + 1
    return resp.choices[0]


def chat(content, max_tokens=4000, temperature=0.6, json_mode=False, think=False, skip=()):
    """One request. content is a string or the usual list with text and images.
    Returns (text, finish_reason, provider id)."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    images = any(c.get("type") == "image_url" for c in content)
    tried, started = {}, time.time()
    while True:
        waiting = []
        for p in ready():
            if p["id"] in skip:
                continue
            until = _cool.get(p["id"], 0)
            if until > time.time():
                waiting.append(until)
                continue
            for attempt in range(3):
                try:
                    c = _call(p, content, max_tokens, temperature, json_mode, think)
                    if _last[0] and _last[0] != p["name"]:
                        for cb in on_switch:
                            cb(_last[0], p["name"], tried.get(_last[0], "was unavailable"))
                    _last[0] = p["name"]
                    return c.message.content or "", c.finish_reason, p["id"]
                except Exception as e:
                    code = _status(e)
                    tried[p["name"]] = _describe(e)
                    if code == 429:
                        msg = str(e).lower()
                        long_wait = "day" in msg or "quota" in msg or "credit" in msg
                        _cool[p["id"]] = time.time() + (3600 if long_wait else 45)
                        break
                    if code in (401, 402, 403, 404):
                        _cool[p["id"]] = time.time() + 1800
                        break
                    if code == 400:
                        if images:  # this model cannot see images: not worth asking again
                            _cool[p["id"]] = time.time() + 600
                        break
                    if attempt == 2:  # network trouble or a server error
                        _cool[p["id"]] = time.time() + 60
                    else:
                        time.sleep(2 + attempt * 3)
        soon = [u for u in waiting if u - time.time() < 90]
        if soon and time.time() - started < 900:
            time.sleep(max(min(soon) - time.time(), 1) + 0.5)
            continue
        if not tried and not waiting and not ready():
            raise AllProvidersFailed("No AI provider is set up. Open Settings, then AI providers, and add a key.")
        why = "; ".join(f"{k}: {v}" for k, v in tried.items()) or "all are resting after reaching a limit"
        raise AllProvidersFailed("All your AI providers stopped (" + why + "). Add credit or another provider "
                                 "in Settings, then press Resume. Nothing done so far is lost.")


def parse_json(text):
    """Finds the JSON in a reply, even inside a code fence or after a sentence."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a:b + 1])
    raise ValueError("no JSON in the reply")
