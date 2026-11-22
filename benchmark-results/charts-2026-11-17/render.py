from playwright.sync_api import sync_playwright
NEW = ['GPT-6.2 Astra', 'GPT-6.2 Sol', 'Gemini 4 Carbon', 'Claude Fable 5.2']
HIDE = """() => { for (const el of document.querySelectorAll('body *')) { const p = getComputedStyle(el).position; if ((p === 'fixed' || p === 'sticky') && !el.closest('.prose')) el.style.display = 'none'; } }"""
BOLD = """(names) => {
  const link = document.createElement('link'); link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@700&family=IBM+Plex+Mono:wght@700&display=swap';
  document.head.append(link);
  let n = 0;
  for (const el of document.querySelectorAll('.score-model, text.cost-label')) {
    if (names.includes(el.textContent.trim())) { el.style.fontWeight = '700'; n += 1; }
  }
  return n;
}"""
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome')
    pg = b.new_page(viewport={'width': 1100, 'height': 900}, device_scale_factor=2)
    pg.goto('http://127.0.0.1:8765/blog/shotwell-bench/', wait_until='domcontentloaded')
    pg.wait_for_selector('#score-grid .score-row'); pg.wait_for_selector('#cost-chart svg text.cost-label')
    pg.evaluate(HIDE); n = pg.evaluate(BOLD, NEW)
    pg.evaluate('document.fonts.ready'); pg.wait_for_timeout(2500)
    pg.locator('[data-score-kind="segmentation"]').screenshot(path='/tmp/sb-remake/headline-segmentation.png')
    pg.locator('[data-score-kind="description"]').screenshot(path='/tmp/sb-remake/headline-description.png')
    pg.locator('#cost-chart').locator('xpath=ancestor::figure[1]').screenshot(path='/tmp/sb-remake/headline-pareto-bold.png')
    print('bolded', n, 'rows', pg.locator('#score-grid .score-row').count(), 'updated:', pg.locator('.post-meta').inner_text())
    b.close()
