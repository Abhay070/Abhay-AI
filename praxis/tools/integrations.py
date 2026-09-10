"""
Integrations with public services that need no API key.

Every service here is free and unauthenticated, which is a deliberate
constraint: a tool that needs an account is a tool most people will never
switch on, and a personal assistant should be useful the moment it starts.

Structure worth noting. Each tool is split in two — a `_parse_*` function that
turns a payload into text, and a thin caller that does the HTTP. The parser is
where the bugs live (a renamed field, a missing key, an empty list) and it is
pure, so it is tested against recorded fixtures rather than the live internet.
The HTTP half is four lines and fails loudly.

Services:
    wikipedia   article summaries — the fastest cure for a hallucinated fact
    dictionary  definitions, phonetics, usage
    exchange    live currency conversion (Frankfurter, ECB data)
    github      public repository facts: stars, language, licence, activity
    hackernews  what the technical world is reading right now
    arxiv       preprint search, for anything research-shaped
"""

from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from . import Tool, ToolResult, register

TIMEOUT = 20
UA = "Praxis/1.0 (personal assistant; +https://github.com/Abhay070/Abhay-AI)"


def _get(url: str, headers: dict | None = None) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": UA, **(headers or {})}) as client:
        response = client.get(url)
        response.raise_for_status()
        return response


def _fail(service: str, exc: Exception) -> ToolResult:
    """One honest failure message shape for every service."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 404:
            return ToolResult(False, f"{service}: not found.")
        if code == 429:
            return ToolResult(False, f"{service}: rate limited. Try again shortly.")
        return ToolResult(False, f"{service}: HTTP {code}.")
    return ToolResult(False, f"{service} is unreachable ({type(exc).__name__}). "
                             "The network may be blocked or offline.")


# --- wikipedia -------------------------------------------------------------

def _parse_wikipedia(payload: dict) -> ToolResult:
    title = payload.get("title", "")
    extract = (payload.get("extract") or "").strip()
    if not extract:
        return ToolResult(False, f"No summary available for '{title}'.")
    if payload.get("type") == "disambiguation":
        return ToolResult(True, f"'{title}' is ambiguous — it could refer to "
                                f"several things. {extract}",
                          {"disambiguation": True})
    url = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
    described = payload.get("description", "")
    head = f"# {title}" + (f"\n_{described}_" if described else "")
    return ToolResult(True, f"{head}\n\n{extract}" + (f"\n\nSource: {url}" if url else ""),
                      {"title": title, "url": url})


def wikipedia(topic: str) -> ToolResult:
    topic = str(topic).strip()
    if not topic:
        return ToolResult(False, "No topic given.")
    slug = urllib.parse.quote(topic.replace(" ", "_"), safe="")
    try:
        response = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        return _parse_wikipedia(response.json())
    except Exception as e:
        return _fail("Wikipedia", e)


# --- dictionary ------------------------------------------------------------

def _parse_dictionary(payload: list) -> ToolResult:
    if not payload:
        return ToolResult(False, "No definition found.")
    entry = payload[0]
    word = entry.get("word", "")
    phonetic = entry.get("phonetic", "")
    lines = [f"# {word}" + (f"  {phonetic}" if phonetic else "")]
    for meaning in entry.get("meanings", [])[:3]:
        part = meaning.get("partOfSpeech", "")
        lines.append(f"\n**{part}**")
        for i, d in enumerate(meaning.get("definitions", [])[:3], 1):
            lines.append(f"{i}. {d.get('definition', '')}")
            if d.get("example"):
                lines.append(f"   _\"{d['example']}\"_")
    return ToolResult(True, "\n".join(lines), {"word": word})


def dictionary(word: str) -> ToolResult:
    word = str(word).strip()
    if not word:
        return ToolResult(False, "No word given.")
    try:
        response = _get("https://api.dictionaryapi.dev/api/v2/entries/en/"
                        + urllib.parse.quote(word))
        return _parse_dictionary(response.json())
    except Exception as e:
        return _fail("Dictionary", e)


# --- currency --------------------------------------------------------------

def _parse_exchange(payload: dict, amount: float, to: str) -> ToolResult:
    rates = payload.get("rates", {})
    base = payload.get("base", "")
    date = payload.get("date", "")
    target = to.upper()
    if target not in rates:
        available = ", ".join(sorted(rates)[:12])
        return ToolResult(False, f"No rate for {target}. Available include: {available}")
    rate = rates[target]
    total = amount * rate
    return ToolResult(
        True,
        f"{amount:,.2f} {base} = {total:,.2f} {target}\n"
        f"Rate: 1 {base} = {rate:.4f} {target} (ECB reference, {date})",
        {"amount": amount, "from": base, "to": target, "rate": rate,
         "result": round(total, 2), "date": date})


def exchange(amount: str = "1", from_currency: str = "USD",
             to_currency: str = "EUR") -> ToolResult:
    try:
        value = float(str(amount).replace(",", ""))
    except (TypeError, ValueError):
        return ToolResult(False, f"'{amount}' is not a number.")
    src, dst = from_currency.upper().strip(), to_currency.upper().strip()
    if not (len(src) == 3 and len(dst) == 3):
        return ToolResult(False, "Currencies must be 3-letter codes, e.g. USD, INR.")
    try:
        response = _get(f"https://api.frankfurter.app/latest?from={src}&to={dst}")
        return _parse_exchange(response.json(), value, dst)
    except Exception as e:
        return _fail("Exchange rates", e)


# --- github ----------------------------------------------------------------

def _parse_github(payload: dict) -> ToolResult:
    name = payload.get("full_name", "")
    if not name:
        return ToolResult(False, "Repository not found.")
    licence = (payload.get("license") or {}).get("spdx_id") or "no licence"
    lines = [
        f"# {name}",
        payload.get("description") or "_no description_",
        "",
        f"- {payload.get('stargazers_count', 0):,} stars · "
        f"{payload.get('forks_count', 0):,} forks · "
        f"{payload.get('open_issues_count', 0):,} open issues",
        f"- Language: {payload.get('language') or 'unspecified'} · Licence: {licence}",
        f"- Updated: {(payload.get('pushed_at') or '')[:10]} · "
        f"Created: {(payload.get('created_at') or '')[:10]}",
    ]
    if payload.get("archived"):
        lines.append("- **Archived** — no longer maintained.")
    if payload.get("homepage"):
        lines.append(f"- Homepage: {payload['homepage']}")
    lines.append(f"\n{payload.get('html_url', '')}")
    return ToolResult(True, "\n".join(lines),
                      {"stars": payload.get("stargazers_count"),
                       "language": payload.get("language"),
                       "archived": bool(payload.get("archived"))})


def github_repo(repo: str) -> ToolResult:
    repo = str(repo).strip().strip("/")
    if repo.startswith("http"):
        parts = urllib.parse.urlparse(repo).path.strip("/").split("/")
        repo = "/".join(parts[:2]) if len(parts) >= 2 else repo
    if repo.count("/") != 1:
        return ToolResult(False, "Give a repository as owner/name, e.g. python/cpython.")
    try:
        response = _get(f"https://api.github.com/repos/{repo}",
                        {"Accept": "application/vnd.github+json"})
        return _parse_github(response.json())
    except Exception as e:
        return _fail("GitHub", e)


# --- hacker news -----------------------------------------------------------

def _parse_hn(items: list[dict], count: int) -> ToolResult:
    stories = [i for i in items if i and i.get("title")][:count]
    if not stories:
        return ToolResult(False, "No stories returned.")
    lines = []
    for i, s in enumerate(stories, 1):
        url = s.get("url") or f"https://news.ycombinator.com/item?id={s.get('id')}"
        lines.append(f"{i}. {s['title']}\n   {s.get('score', 0)} points · "
                     f"{s.get('descendants', 0)} comments\n   {url}")
    return ToolResult(True, "Top on Hacker News right now:\n\n" + "\n\n".join(lines),
                      {"count": len(stories)})


def hacker_news(count: str = "5") -> ToolResult:
    try:
        n = max(1, min(int(count), 15))
    except (TypeError, ValueError):
        n = 5
    try:
        ids = _get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:n]
        items = []
        for story_id in ids:
            items.append(_get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json").json())
        return _parse_hn(items, n)
    except Exception as e:
        return _fail("Hacker News", e)


# --- arxiv -----------------------------------------------------------------

_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def _parse_arxiv(xml_text: str, count: int) -> ToolResult:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return ToolResult(False, f"arXiv returned unparseable XML: {e.msg}")
    entries = root.findall("a:entry", _ARXIV_NS)[:count]
    if not entries:
        return ToolResult(False, "No papers matched.")
    lines = []
    for i, entry in enumerate(entries, 1):
        def text(tag: str) -> str:
            node = entry.find(f"a:{tag}", _ARXIV_NS)
            return (node.text or "").strip() if node is not None else ""

        authors = [a.text for a in entry.findall("a:author/a:name", _ARXIV_NS)][:3]
        summary = " ".join(text("summary").split())[:280]
        lines.append(
            f"{i}. **{' '.join(text('title').split())}**\n"
            f"   {', '.join(authors)}{' et al.' if len(authors) == 3 else ''} · "
            f"{text('published')[:10]}\n"
            f"   {summary}…\n   {text('id')}")
    return ToolResult(True, "\n\n".join(lines), {"count": len(entries)})


def arxiv(query: str, count: str = "3") -> ToolResult:
    query = str(query).strip()
    if not query:
        return ToolResult(False, "No search query given.")
    try:
        n = max(1, min(int(count), 10))
    except (TypeError, ValueError):
        n = 3
    url = ("https://export.arxiv.org/api/query?search_query=all:"
           + urllib.parse.quote(query)
           + f"&start=0&max_results={n}&sortBy=relevance")
    try:
        return _parse_arxiv(_get(url).text, n)
    except Exception as e:
        return _fail("arXiv", e)


# --- registration ----------------------------------------------------------

register(Tool(
    name="wikipedia",
    description="Look up an encyclopedia summary of a person, place, event or "
                "concept. Faster and more reliable than a general web search for "
                "settled facts — use it before answering from memory",
    args={"topic": "article title, e.g. 'George M. Dallas'"},
    run=wikipedia, icon="📖", requires="enable_web",
))

register(Tool(
    name="dictionary",
    description="Definitions, pronunciation and usage examples for an English word",
    args={"word": "the word to define"},
    run=dictionary, icon="🔤", requires="enable_web",
))

register(Tool(
    name="exchange",
    description="Convert between currencies at today's reference rate. Use for any "
                "money conversion rather than estimating",
    args={"amount": "how much, e.g. '250'", "from_currency": "3-letter code, e.g. USD",
          "to_currency": "3-letter code, e.g. INR"},
    run=exchange, icon="💱", requires="enable_web",
))

register(Tool(
    name="github_repo",
    description="Facts about a public GitHub repository: stars, language, licence, "
                "activity, whether it is archived. Use when asked to evaluate or "
                "compare a library",
    args={"repo": "owner/name, or a GitHub URL"},
    run=github_repo, icon="⑂", requires="enable_web",
))

register(Tool(
    name="hacker_news",
    description="What the technical world is reading right now — top Hacker News "
                "stories with scores and comment counts",
    args={"count": "how many stories, 1-15 (default 5)"},
    run=hacker_news, icon="◫", requires="enable_web",
))

register(Tool(
    name="arxiv",
    description="Search academic preprints. Use for research questions where a "
                "primary paper beats a blog summary",
    args={"query": "search terms", "count": "how many papers, 1-10 (default 3)"},
    run=arxiv, icon="🎓", requires="enable_web",
))
