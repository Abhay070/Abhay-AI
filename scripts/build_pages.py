"""
Build the static showcase site for GitHub Pages.

Pages can serve files; it cannot run Python. So the app itself (streaming,
tools, memory) cannot live there — only the landing page can. This script
produces a version of it that works at a project-page URL.

Three things have to change, and all three are the kind of thing that silently
produces a broken site rather than an error:

  1. Absolute asset paths. The site is served from /<repo>/, so "/static/app.css"
     resolves to the user's root domain and 404s. They must become relative.
  2. The "Open Praxis" buttons. There is no server behind them here, so they
     point at the repository with setup instructions instead of a dead link.
  3. A banner saying plainly that this is a showcase and where the real thing
     runs — better than a visitor clicking a button that does nothing.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
OUT = ROOT / "_site"
REPO = "https://github.com/Abhay070/Abhay-AI"

BANNER = """
<div class="pages-banner">
  <strong>This is the showcase site.</strong>
  Praxis is a Python app that runs on your own machine — GitHub Pages can host
  this page, but not the assistant itself.
  <a href="{repo}#get-it-running">Two commands to run it →</a>
</div>
<style>
.pages-banner {{
  background: var(--bg-raised); border-bottom: 1px solid var(--border);
  padding: 10px 28px; font-size: 13.5px; color: var(--text-dim);
  text-align: center; line-height: 1.5;
}}
.pages-banner strong {{ color: var(--text); }}
.pages-banner a {{ white-space: nowrap; }}
</style>
""".format(repo=REPO)


def build() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    shutil.copytree(WEB / "static", OUT / "static")

    html = (WEB / "index.html").read_text(encoding="utf-8")

    # 1 — absolute asset paths become relative
    html = html.replace('href="/static/', 'href="static/')
    html = html.replace('src="/static/', 'src="static/')
    html = html.replace("url('/static/", "url('static/")

    # 2 — the app links have nothing behind them here
    html = html.replace('href="/chat"', f'href="{REPO}#get-it-running"')
    html = html.replace("Open Praxis →", "Get it running →")
    html = html.replace("Start a conversation", "Get it running")
    html = html.replace(">Open<", ">Get it running<")

    # 3 — say so, above the fold
    html = html.replace("<nav class=\"nav\">", BANNER + "\n<nav class=\"nav\">", 1)

    (OUT / "index.html").write_text(html, encoding="utf-8")

    # The CSS references the logo the same absolute way.
    css_path = OUT / "static" / "app.css"
    css = css_path.read_text(encoding="utf-8")
    css_path.write_text(css.replace("url('/static/logo.svg')", "url('logo.svg')"),
                        encoding="utf-8")

    # Stops Pages running the output through Jekyll, which would eat _underscore
    # directories and any Liquid-looking braces in the CSS.
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    leftovers = re.findall(r'(?:href|src)="/(?!/)[^"]*"', (OUT / "index.html").read_text())
    if leftovers:
        print("FAIL — absolute paths survived:", leftovers, file=sys.stderr)
        return 1

    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print(f"built _site/ with {len(files)} files:")
    for f in files:
        print("  ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
