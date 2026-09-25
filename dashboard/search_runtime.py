"""Small, dependency-free retrieval client for the local chat dashboard."""

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def search_web(query, limit=4, timeout=15):
    """Retrieve concise, attributable passages from Wikipedia's search API."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": max(1, min(int(limit), 6)),
        "prop": "extracts|info",
        "exintro": "1",
        "explaintext": "1",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    }
    request = Request(
        f"{WIKIPEDIA_API}?{urlencode(params)}",
        headers={"User-Agent": "GroundedResearchDashboard/1.0 (local research project)"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", [])
    sources = []
    for page in pages:
        extract = " ".join(str(page.get("extract", "")).split())
        if not extract:
            continue
        sources.append({
            "title": page.get("title", "Untitled source"),
            "url": page.get("fullurl", ""),
            "snippet": extract[:1800],
        })
    return sources


def sources_to_context(sources):
    return "\n\n".join(
        f"[{index}] {source['title']}\n{source['snippet']}"
        for index, source in enumerate(sources, start=1)
    )
