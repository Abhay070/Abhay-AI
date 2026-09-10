from playwright.sync_api import sync_playwright
S="/tmp/claude-0/-home-user-Abhay-AI/3ead63c1-5449-598d-aa43-d0efe1430187/scratchpad"
errs, http_bad = [], []
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    pg = b.new_page(viewport={"width":1440,"height":920}, device_scale_factor=2)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("response", lambda r: http_bad.append(f"{r.status} {r.url}") if r.status>=400 else None)
    pg.on("dialog", lambda d: d.accept())

    # ---------- LANDING PAGE ----------
    pg.goto("http://127.0.0.1:8000/", wait_until="networkidle"); pg.wait_for_timeout(700)
    nav = pg.locator(".nav-links a")
    # Count is not the invariant — every link resolving is. Hard-coding the
    # number just breaks the test each time a section is added.
    check("landing: nav links present", nav.count() >= 5, f"{nav.count()} links")
    ok_anchor = True
    for i in range(nav.count()):
        href = nav.nth(i).get_attribute("href")
        if not href.startswith("#") or pg.locator(href).count() == 0:
            ok_anchor = False
    check("landing: every nav anchor resolves", ok_anchor)
    ctas = pg.locator('a[href="/chat"]')
    check("landing: CTA buttons point at /chat", ctas.count() >= 3, f"{ctas.count()} CTAs")

    # ---------- CHAT APP ----------
    pg.goto("http://127.0.0.1:8000/chat", wait_until="networkidle"); pg.wait_for_timeout(900)

    check("app: strategy picker rendered", pg.locator("#strategyBtn").count() == 1)
    pg.click("#strategyBtn"); pg.wait_for_timeout(350)
    n_strat = pg.locator("#strategyMenu .mode-item").count()
    check("btn: strategy menu opens", n_strat == 4, f"{n_strat} strategies")
    pg.locator("#strategyMenu .mode-item").nth(1).click(); pg.wait_for_timeout(300)
    check("btn: strategy selectable", pg.locator("#strategyLabel").inner_text() == "Council",
          pg.locator("#strategyLabel").inner_text())
    # put it back to single so the demo backend works normally
    pg.click("#strategyBtn"); pg.wait_for_timeout(250)
    pg.locator("#strategyMenu .mode-item").nth(0).click(); pg.wait_for_timeout(250)

    pg.click("#modeBtn"); pg.wait_for_timeout(300)
    check("btn: mode menu opens", pg.locator("#modeMenu .mode-item").count() == 10)
    pg.locator("#modeMenu .mode-item").nth(2).click(); pg.wait_for_timeout(250)
    check("btn: mode selectable", pg.locator("#modeLabel").inner_text() == "Founder",
          pg.locator("#modeLabel").inner_text())

    theme_before = pg.evaluate("document.documentElement.dataset.theme")
    pg.click("#toggleTheme"); pg.wait_for_timeout(300)
    check("btn: theme toggle", pg.evaluate("document.documentElement.dataset.theme") != theme_before)
    pg.click("#toggleTheme"); pg.wait_for_timeout(200)

    pg.click("#toggleSidebar"); pg.wait_for_timeout(300)
    collapsed = "collapsed" in (pg.locator("#sidebar").get_attribute("class") or "")
    check("btn: sidebar toggle", collapsed)
    pg.click("#toggleSidebar"); pg.wait_for_timeout(250)

    pg.click("#openPalette"); pg.wait_for_timeout(400)
    check("btn: command palette opens", pg.locator(".palette-item").count() > 10,
          f"{pg.locator('.palette-item').count()} commands")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(200)

    pg.click("#openMemory"); pg.wait_for_timeout(800)
    check("btn: memory drawer opens", pg.locator(".drawer").count() == 1)
    pg.keyboard.press("Escape"); pg.wait_for_timeout(250)

    pg.click("#openSettings"); pg.wait_for_timeout(800)
    check("btn: settings drawer opens", pg.locator(".drawer").count() == 1)
    insp = pg.locator("text=Inspect the prompt for this mode")
    if insp.count():
        insp.click(); pg.wait_for_timeout(700)
        check("btn: prompt inspector works", pg.locator("pre.prompt-dump").count() == 1)
    pg.keyboard.press("Escape"); pg.wait_for_timeout(250)

    check("btn: attach labelled", "Attach" in pg.locator("#attachBtn").inner_text())

    # Back to Standard before sending. Founder mode carries a contract, and a
    # scripted demo reply cannot honour it — the resulting rewrite would clear
    # the tool panel and make this check about contracts rather than sending.
    pg.click("#modeBtn"); pg.wait_for_timeout(250)
    pg.locator("#modeMenu .mode-item").nth(0).click(); pg.wait_for_timeout(250)

    # send a real message so message-level buttons exist
    pg.fill("#input", "what is 1920 x 1080")
    pg.click("#sendBtn")
    pg.wait_for_selector(".tool-run", timeout=30000); pg.wait_for_timeout(6000)
    check("flow: message sent + tool ran", pg.locator(".tool-run").count() >= 1)
    check("flow: conversation appears in sidebar", pg.locator(".conv").count() >= 1)

    pg.locator(".msg.assistant").first.hover(); pg.wait_for_timeout(250)
    foot = pg.locator(".msg.assistant .msg-foot button")
    check("btn: message actions present", foot.count() >= 2, f"{foot.count()} buttons")

    pg.fill("#search", "1920"); pg.wait_for_timeout(700)
    check("btn: sidebar search filters", pg.locator(".conv").count() >= 1)
    pg.fill("#search", "zzzznomatch"); pg.wait_for_timeout(700)
    check("btn: search with no match empties list", pg.locator(".conv").count() == 0)
    pg.fill("#search", ""); pg.wait_for_timeout(500)

    pg.locator(".conv").first.hover(); pg.wait_for_timeout(200)
    pins = pg.locator(".conv-actions button")
    check("btn: conversation pin/delete present", pins.count() >= 2)
    # Pin TOGGLES, and the database persists between runs — so assert it
    # changed state, not that it reached a particular state. Asserting
    # "Pinned" passes on a clean database and fails on the second run.
    before = pg.locator(".conv-group").first.inner_text().strip().lower()
    pins.nth(0).click(); pg.wait_for_timeout(800)
    after = pg.locator(".conv-group").first.inner_text().strip().lower()
    check("btn: pin toggles", before != after, f"{before} -> {after}")
    pins_again = pg.locator(".conv").first
    pins_again.hover(); pg.wait_for_timeout(200)
    pg.locator(".conv-actions button").nth(0).click(); pg.wait_for_timeout(700)
    check("btn: pin toggles back",
          pg.locator(".conv-group").first.inner_text().strip().lower() == before,
          pg.locator(".conv-group").first.inner_text())

    pg.click("#newChat"); pg.wait_for_timeout(400)
    check("btn: new chat resets thread", pg.locator(".empty").count() == 1)

    starters = pg.locator(".starter")
    check("btn: starter cards present", starters.count() == 4)

    # ---------- NEW CONTROLS ----------
    pg.click("#helpBtn"); pg.wait_for_timeout(600)
    check("btn: shortcuts sheet opens", pg.locator(".key-row").count() >= 10,
          f"{pg.locator('.key-row').count()} shortcuts")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(250)

    pg.keyboard.press("?"); pg.wait_for_timeout(500)
    check("key: ? opens shortcuts", pg.locator(".key-row").count() >= 10)
    pg.keyboard.press("Escape"); pg.wait_for_timeout(250)

    # slash commands
    pg.click("#input")
    pg.fill("#input", "/")
    pg.dispatch_event("#input", "input"); pg.wait_for_timeout(400)
    n_slash = pg.locator(".slash-item").count()
    check("slash: menu opens on /", n_slash > 0, f"{n_slash} commands")
    pg.fill("#input", "/found")
    pg.dispatch_event("#input", "input"); pg.wait_for_timeout(400)
    check("slash: filters as you type",
          pg.locator(".slash-item").count() == 1 and
          "founder" in pg.locator(".slash-cmd").first.inner_text(),
          pg.locator(".slash-cmd").first.inner_text() if pg.locator(".slash-item").count() else "none")
    pg.locator(".slash-item").first.click(); pg.wait_for_timeout(400)
    check("slash: running a command applies it",
          pg.locator("#modeLabel").inner_text() == "Founder" and
          pg.locator("#input").input_value() == "",
          pg.locator("#modeLabel").inner_text())

    # a slash inside a sentence must NOT open the menu
    pg.fill("#input", "what is 10/2")
    pg.dispatch_event("#input", "input"); pg.wait_for_timeout(300)
    check("slash: mid-sentence slash ignored",
          pg.locator("#slashMenu").is_hidden())
    pg.fill("#input", "")

    check("btn: mic button present", pg.locator("#micBtn").count() == 1)
    check("chip: mode promises shown", pg.locator("#promiseChip").count() == 1)

    # council drawer via slash
    pg.fill("#input", "/council")
    pg.dispatch_event("#input", "input"); pg.wait_for_timeout(350)
    pg.locator(".slash-item").first.click(); pg.wait_for_timeout(1200)
    check("slash: /council opens the model drawer",
          pg.locator(".drawer").count() == 1 and
          "council" in pg.locator(".drawer-head h3").inner_text().lower(),
          pg.locator(".drawer-head h3").inner_text() if pg.locator(".drawer").count() else "none")
    pg.keyboard.press("Escape"); pg.wait_for_timeout(250)

    pg.screenshot(path=f"{S}/audit-final.png")
    # ---------- ATTACHMENTS ----------
    # The failure this pins: uploading a PDF used to paste its text into the
    # user's own message, so the transcript became the document. What the user
    # typed must stay what the user typed, and the file must still reach the
    # model.
    import tempfile as _tempfile, os as _os
    marker = "ZZQUANTUMHERRING"
    doc = _os.path.join(_tempfile.mkdtemp(), "report.txt")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write(f"{marker} project brief.\n" + ("Water conservation. " * 400))

    pg.click("#newChat"); pg.wait_for_timeout(600)
    pg.set_input_files("#fileInput", doc); pg.wait_for_timeout(1500)
    check("attach: chip appears in the composer",
          pg.locator("#attachments .attachment").count() == 1,
          f"{pg.locator('#attachments .attachment').count()} chips")

    pg.fill("#input", "what is this and what does it lack")
    pg.click("#sendBtn"); pg.wait_for_timeout(9000)

    typed = pg.locator(".msg.user .msg-body").last.inner_text()
    check("attach: user's message is only what they typed",
          typed.strip() == "what is this and what does it lack", typed[:70])
    check("attach: the file's text is NOT pasted into the transcript",
          marker not in typed and "Water conservation." not in typed)
    check("attach: the turn shows a file chip",
          pg.locator(".msg.user .msg-file").count() == 1,
          pg.locator(".msg.user .msg-file").first.inner_text()
          if pg.locator(".msg.user .msg-file").count() else "no chip")

    # And the model really did receive it: the prompt inspector shows what was
    # sent. A chip that hides the file from the model would be worse than the
    # bug it replaced.
    import urllib.request as _u, json as _j
    with _u.urlopen("http://127.0.0.1:8000/api/conversations") as r:
        cid = _j.load(r)[0]["id"]
    with _u.urlopen(f"http://127.0.0.1:8000/api/conversations/{cid}") as r:
        convo = _j.load(r)
    user_turn = [m for m in convo["messages"] if m["role"] == "user"][-1]
    check("attach: stored turn keeps the file by reference, not by value",
          marker not in user_turn["content"]
          and bool((user_turn.get("meta") or {}).get("attachments")),
          str(user_turn.get("meta"))[:70])

    pg.screenshot(path=f"{S}/attachment-flow.png", full_page=False)

    b.close()

# The suite drives the real app against the real database, so it must clean
# up the conversations it created. Otherwise every run leaves another identical
# "what is 1920 x 1080" in the sidebar, and after ten runs the app looks broken.
import urllib.request, json as _json
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/conversations") as r:
        for c in _json.load(r):
            if c["title"].startswith(("What is 1920", "Test ", "/", "What is this")):
                req = urllib.request.Request(
                    f"http://127.0.0.1:8000/api/conversations/{c['id']}",
                    method="DELETE")
                urllib.request.urlopen(req).read()
except Exception as e:
    print(f"(cleanup skipped: {type(e).__name__})")

print("=" * 62)
bad = [r for r in results if not r[1]]
for name, ok, detail in results:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"   [{detail}]" if detail else ""))
print("=" * 62)
print(f"{len(results)-len(bad)}/{len(results)} interface checks passed")
print("HTTP errors:", http_bad or "none")
print("JS errors  :", errs or "none")
