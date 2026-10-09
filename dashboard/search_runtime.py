"""Reliable, dependency-free Wikipedia retrieval for the local dashboard."""

from collections import OrderedDict
import json
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
CACHE_TTL_SECONDS = 10 * 60
CACHE_MAX_ITEMS = 96
MAX_ATTEMPTS = 3
_CACHE = OrderedDict()
_LOCK = threading.Lock()
_STATS = {"requests": 0, "network_requests": 0, "cache_hits": 0, "retries": 0,
          "failures": 0, "last_error": None, "last_success_at": None}


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _cache_get(key, now):
    with _LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        saved_at, sources = entry
        if now - saved_at > CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        _CACHE.move_to_end(key)
        _STATS["cache_hits"] += 1
        return [dict(source) for source in sources]


def _cache_put(key, sources, now):
    with _LOCK:
        _CACHE[key] = (now, [dict(source) for source in sources])
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX_ITEMS:
            _CACHE.popitem(last=False)


def _fetch_wikipedia(query, limit, timeout):
    params = {"action": "query", "generator": "search", "gsrsearch": query,
              "gsrlimit": max(1, min(int(limit), 8)), "prop": "extracts|info",
              "exintro": "1", "explaintext": "1", "inprop": "url",
              "format": "json", "formatversion": "2"}
    request = Request(f"{WIKIPEDIA_API}?{urlencode(params)}",
                      headers={"User-Agent": "GroundedResearchDashboard/2.0 (local research project)"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_sources(payload, query, limit):
    query_tokens = _tokens(query)
    ranked = []
    for page in payload.get("query", {}).get("pages", []):
        extract = " ".join(str(page.get("extract", "")).split())
        if not extract:
            continue
        title = str(page.get("title", "Untitled source"))
        title_overlap = len(query_tokens & _tokens(title)) / max(len(query_tokens), 1)
        snippet_overlap = len(query_tokens & _tokens(extract[:600])) / max(len(query_tokens), 1)
        score = 0.7 * title_overlap + 0.3 * snippet_overlap
        ranked.append((score, {"title": title, "url": page.get("fullurl", ""),
                               "snippet": extract[:2200], "retrieval_score": round(score, 4),
                               "provider": "Wikipedia", "retrieved_for": query}))
    ranked.sort(key=lambda item: (-item[0], item[1]["title"].lower()))
    seen, result = set(), []
    for _, source in ranked:
        identity = source["url"] or source["title"].lower()
        if identity in seen:
            continue
        seen.add(identity)
        result.append(source)
        if len(result) >= limit:
            break
    return result


def search_web_detailed(query, limit=4, timeout=12, attempts=MAX_ATTEMPTS):
    """Retrieve ranked passages with bounded retries, cache and diagnostics."""
    normalized_query = " ".join(str(query).split()).strip()
    if not normalized_query:
        return [], {"status": "empty-query", "query": "", "attempts": 0, "cached": False}
    limit = max(1, min(int(limit), 8))
    key, now = (normalized_query.casefold(), limit), time.time()
    with _LOCK:
        _STATS["requests"] += 1
    cached = _cache_get(key, now)
    if cached is not None:
        return cached, {"status": "ok", "query": normalized_query, "attempts": 0,
                        "cached": True, "provider": "Wikipedia", "source_count": len(cached)}
    last_error = None
    attempts = max(1, min(int(attempts), MAX_ATTEMPTS))
    for attempt in range(1, attempts + 1):
        try:
            with _LOCK:
                _STATS["network_requests"] += 1
            sources = _normalize_sources(_fetch_wikipedia(normalized_query, limit, timeout),
                                         normalized_query, limit)
            _cache_put(key, sources, time.time())
            with _LOCK:
                _STATS["last_error"] = None
                _STATS["last_success_at"] = int(time.time())
            return sources, {"status": "ok" if sources else "no-results", "query": normalized_query,
                             "attempts": attempt, "cached": False, "provider": "Wikipedia",
                             "source_count": len(sources)}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                with _LOCK:
                    _STATS["retries"] += 1
                time.sleep(0.15 * attempt)
    with _LOCK:
        _STATS["failures"] += 1
        _STATS["last_error"] = last_error
    return [], {"status": "error", "query": normalized_query, "attempts": attempts,
                "cached": False, "provider": "Wikipedia", "source_count": 0, "error": last_error}


def search_web(query, limit=4, timeout=12):
    sources, metadata = search_web_detailed(query, limit, timeout)
    if metadata["status"] == "error":
        raise RuntimeError(metadata["error"])
    return sources


def retrieval_health():
    with _LOCK:
        return {**_STATS, "provider": "Wikipedia", "cache_items": len(_CACHE),
                "cache_ttl_seconds": CACHE_TTL_SECONDS,
                "status": "degraded" if _STATS["last_error"] else "ready"}


def clear_retrieval_cache():
    with _LOCK:
        _CACHE.clear()


def sources_to_context(sources):
    return "\n\n".join(f"[{index}] {source['title']}\n{source['snippet']}"
                         for index, source in enumerate(sources, start=1))
