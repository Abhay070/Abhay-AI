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
    check("landing: nav links present", nav.count() == 5, f"{nav.count()} links")
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
    pins.nth(0).click(); pg.wait_for_timeout(700)
    check("btn: pin works", pg.locator(".conv-group").first.inner_text().strip().lower() == "pinned",
          pg.locator(".conv-group").first.inner_text())

    pg.click("#newChat"); pg.wait_for_timeout(400)
    check("btn: new chat resets thread", pg.locator(".empty").count() == 1)

    starters = pg.locator(".starter")
    check("btn: starter cards present", starters.count() == 4)

    pg.screenshot(path=f"{S}/audit-final.png")
    b.close()

print("=" * 62)
bad = [r for r in results if not r[1]]
for name, ok, detail in results:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"   [{detail}]" if detail else ""))
print("=" * 62)
print(f"{len(results)-len(bad)}/{len(results)} interface checks passed")
print("HTTP errors:", http_bad or "none")
print("JS errors  :", errs or "none")
