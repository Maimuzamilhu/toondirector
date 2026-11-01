"""Looks a topic up on the web before the script is written, so facts come from real sources.

Wikipedia works without any key. With a free Tavily key the whole web is searched as well.
"""
import json
import re
import urllib.parse
import urllib.request

from . import config, providers

UA = {"User-Agent": "ToonDirector/1.0 (desktop app)"}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def wikipedia(query, lang="en", n=3):
    base = f"https://{lang}.wikipedia.org/w/api.php?"
    found = _get(base + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                                "srlimit": n, "format": "json"}))
    titles = [x["title"] for x in found["query"]["search"]]
    out = []
    for t in titles:
        page = _get(base + urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                                   "exchars": 3500, "titles": t, "format": "json"}))
        for v in page["query"]["pages"].values():
            if v.get("extract"):
                out.append({"title": t, "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                            "text": re.sub(r"\n{2,}", "\n", v["extract"])})
    return out


def tavily(query):
    body = json.dumps({"query": query, "search_depth": "basic", "max_results": 6}).encode()
    req = urllib.request.Request("https://api.tavily.com/search", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + config.TAVILY_API_KEY, **UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "text": x.get("content", "")} for x in data.get("results", [])]


def _search_terms(topic):
    """A full sentence finds the wrong pages, so the AI names the encyclopedia articles to read."""
    try:
        ask = (f'A video is about: "{topic[:300]}". Name the 3 Wikipedia article titles most useful to research it, '
               'in English. Reply with JSON only: {"titles": ["...", "...", "..."]}')
        text, _, _ = providers.chat(ask, max_tokens=200, temperature=0.2, json_mode=True)
        titles = [str(t) for t in providers.parse_json(text).get("titles", [])][:3]
        if titles:
            return titles
    except Exception:
        pass
    return [topic]


def gather(topic, lang, log):
    """Returns (text for the writer, list of sources). Never stops a job: research is a bonus."""
    sources, seen = [], set()
    try:
        for q in _search_terms(topic):
            for s in wikipedia(q, "en", 1):  # English has the most depth; the script is written in your language
                if s["title"] not in seen:
                    seen.add(s["title"])
                    sources.append(s)
    except Exception as e:
        log(f"Wikipedia could not be reached ({str(e)[:80]})")
    if config.TAVILY_API_KEY:
        try:
            sources += tavily(topic)
        except Exception as e:
            log(f"Web search could not be reached ({str(e)[:80]})")
    if not sources:
        return "", []
    log(f"Research: read {len(sources)} sources")
    text = "\n\n".join(f"Source {i + 1}: {s['title']}\n{s['text'][:3000]}" for i, s in enumerate(sources))
    return text[:18000], [{"title": s["title"], "url": s["url"]} for s in sources]
