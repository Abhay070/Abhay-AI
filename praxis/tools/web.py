"""
Web search and page fetching.

Search has four interchangeable backends, chosen by SEARCH_BACKEND:

  duckduckgo  (default) scrapes the HTML endpoint. No key, no account, free.
              Fragile by nature — it is scraping — and some networks block it.
  brave       Brave Search API. Free tier, needs BRAVE_API_KEY.
  tavily      Tavily. Built for LLM use, returns clean summaries.
              Free tier, needs TAVILY_API_KEY.
  searxng     Your own SearXNG instance. Set SEARXNG_URL.

Multiple backends is not over-engineering here: search is the tool most likely
to be blocked by a corporate proxy, a firewall, or rate limiting, and an
assistant whose search silently dies is worse than one with no search at all.
Each backend reports its own failure clearly so you know which layer broke.
"""

from __future__ import annotations

import html
import os
import re
import urllib.parse

import httpx

from . import Tool, ToolResult, register

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

MAX_CHARS = 6000

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
                        re.DOTALL | re.IGNORECASE)
_BLANKS_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINES_RE = re.compile(r"\n\s*\n\s*\n+")
_DDG_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'(?:<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?',
    re.DOTALL | re.IGNORECASE,
)


def _text(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment or "")).strip()


def _tidy(text: str) -> str:
    """Collapse the whitespace wreckage that stripping tags leaves behind."""
    lines = [_BLANKS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def _format(query: str, results: list[dict], backend: str) -> ToolResult:
    if not results:
        return ToolResult(False, f"No results for '{query}' via {backend}.")
    lines = [f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
             for i, r in enumerate(results, 1)]
    return ToolResult(
        True,
        f"Search results for '{query}' (via {backend}):\n\n" + "\n\n".join(lines) +
        "\n\nThese are snippets, not the pages. Use web_fetch before relying on detail.",
        {"query": query, "results": results, "backend": backend},
    )


# -- backends ---------------------------------------------------------------

def _duckduckgo(query: str, count: int) -> ToolResult:
    with httpx.Client(timeout=20, follow_redirects=True,
                      headers={"User-Agent": UA}) as client:
        resp = client.post("https://html.duckduckgo.com/html/",
                           data={"q": query, "kl": "wt-wt"})
        resp.raise_for_status()

    results = []
    for match in _DDG_RE.finditer(resp.text):
        href = match.group(1)
        if "duckduckgo.com/l/" in href:
            parsed = urllib.parse.urlparse("https:" + href if href.startswith("//") else href)
            target = urllib.parse.parse_qs(parsed.query).get("uddg")
            href = urllib.parse.unquote(target[0]) if target else href
        title = _text(match.group(2))
        if not title or not href.startswith("http"):
            continue
        results.append({"title": title, "url": href,
                        "snippet": _text(match.group(3) or "")[:300]})
        if len(results) >= count:
            break
    return _format(query, results, "duckduckgo")


def _brave(query: str, count: int) -> ToolResult:
    key = os.getenv("BRAVE_API_KEY", "")
    if not key:
        return ToolResult(False, "BRAVE_API_KEY is not set. Free key at "
                                 "https://brave.com/search/api/")
    with httpx.Client(timeout=20) as client:
        resp = client.get("https://api.search.brave.com/res/v1/web/search",
                          params={"q": query, "count": count},
                          headers={"X-Subscription-Token": key,
                                   "Accept": "application/json"})
        resp.raise_for_status()
        payload = resp.json()
    results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "snippet": _text(r.get("description", ""))[:300]}
               for r in payload.get("web", {}).get("results", [])[:count]]
    return _format(query, results, "brave")


def _tavily(query: str, count: int) -> ToolResult:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return ToolResult(False, "TAVILY_API_KEY is not set. Free key at "
                                 "https://tavily.com")
    with httpx.Client(timeout=25) as client:
        resp = client.post("https://api.tavily.com/search",
                           json={"api_key": key, "query": query,
                                 "max_results": count,
                                 "include_answer": True})
        resp.raise_for_status()
        payload = resp.json()
    results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:300]}
               for r in payload.get("results", [])[:count]]
    out = _format(query, results, "tavily")
    if out.ok and payload.get("answer"):
        out.output = f"Summary: {payload['answer']}\n\n{out.output}"
    return out


def _searxng(query: str, count: int) -> ToolResult:
    base = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not base:
        return ToolResult(False, "SEARXNG_URL is not set.")
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        resp = client.get(f"{base}/search",
                          params={"q": query, "format": "json"})
        resp.raise_for_status()
        payload = resp.json()
    results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:300]}
               for r in payload.get("results", [])[:count]]
    return _format(query, results, "searxng")


BACKENDS = {"duckduckgo": _duckduckgo, "brave": _brave,
            "tavily": _tavily, "searxng": _searxng}


def web_search(query: str, count: int = 5) -> ToolResult:
    query = str(query).strip()
    if not query:
        return ToolResult(False, "Empty query.")
    try:
        count = max(1, min(int(count), 10))
    except (TypeError, ValueError):
        count = 5

    name = os.getenv("SEARCH_BACKEND", "duckduckgo").lower()
    backend = BACKENDS.get(name)
    if backend is None:
        return ToolResult(False, f"Unknown SEARCH_BACKEND '{name}'. "
                                 f"Options: {', '.join(BACKENDS)}")
    try:
        return backend(query, count)
    except httpx.HTTPStatusError as e:
        return ToolResult(False, f"{name} returned HTTP {e.response.status_code}.")
    except httpx.HTTPError as e:
        return ToolResult(
            False,
            f"Search via {name} failed ({type(e).__name__}). The host may be blocked by "
            "a proxy or firewall. Try another SEARCH_BACKEND, or use web_fetch on a "
            "specific URL instead.")


def web_fetch(url: str) -> ToolResult:
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        return ToolResult(False, "URL must start with http:// or https://")
    try:
        with httpx.Client(timeout=25, follow_redirects=True,
                          headers={"User-Agent": UA}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            raw = resp.text
    except httpx.HTTPStatusError as e:
        return ToolResult(False, f"HTTP {e.response.status_code} fetching {url}")
    except httpx.HTTPError as e:
        return ToolResult(False, f"Could not fetch {url}: {type(e).__name__}")

    if "html" in ctype:
        match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.DOTALL | re.IGNORECASE)
        title = _text(match.group(1)) if match else url
        body = _SCRIPT_RE.sub(" ", raw)
        body = re.sub(r"</(p|div|h[1-6]|li|tr|section|article)>", "\n", body, flags=re.I)
        body = re.sub(r"<br[^>]*>", "\n", body, flags=re.I)
        text = _tidy(_text(body))
    else:
        title = url
        text = _tidy(raw) if "json" not in ctype else raw

    truncated = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    note = f"\n\n[truncated at {MAX_CHARS} characters]" if truncated else ""
    return ToolResult(True, f"# {title}\nSource: {url}\n\n{text}{note}",
                      {"url": url, "title": title, "truncated": truncated})


register(Tool(
    name="web_search",
    description="Search the web. Use for anything current, anything after your training "
                "cutoff, prices, news, versions, or facts you are not certain about",
    args={"query": "search terms", "count": "results to return, 1-10 (default 5)"},
    run=web_search,
    icon="⌕",
    requires="enable_web",
))

register(Tool(
    name="web_fetch",
    description="Fetch and read a web page as text. Use after web_search when a snippet "
                "is not enough, or when the user gives you a URL",
    args={"url": "the full URL"},
    run=web_fetch,
    icon="⇲",
    requires="enable_web",
))
