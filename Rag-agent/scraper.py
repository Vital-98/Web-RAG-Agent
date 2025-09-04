import asyncio, time, re
from urllib.parse import quote_plus
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from playwright_stealth import stealth
from logger import get_logger
from utils import clean_whitespace, ensure_dirs, save_json

logger = get_logger("scraper")

GOOGLE = "https://www.google.com/search?q={q}&num=10&hl=en"
MAX_CHARS = 120_000

async def _gather_google(page, query: str):
    url = GOOGLE.format(q=quote_plus(query))
    logger.info(f"SEARCH → {url}")
    await page.goto(url, timeout=60_000)
    await page.wait_for_selector("#search", timeout=20_000)

    anchors = await page.eval_on_selector_all(
        "#search a[href]",
        "els => els.map(a => ({href:a.href, text:a.textContent||''}))"
    )

    results, seen = [], set()
    for a in anchors:
        href = a["href"]
        if not href.startswith("http"): continue
        if "google." in href and "/url?" not in href: continue
        if href in seen: continue
        seen.add(href)
        title = clean_whitespace(a["text"]) or href
        results.append({"url": href, "title": title})
        if len(results) >= 20: break

    uniq, seen2 = [], set()
    for r in results:
        key = r["url"].split("#")[0]
        if key in seen2: continue
        seen2.add(key)
        uniq.append(r)
        if len(uniq) == 10: break
    logger.info(f"SEARCH → {len(uniq)} candidates kept")
    return uniq

async def _scrape_page(page, url: str):
    try:
        logger.info(f"FETCH → {url}")
        await page.goto(url, timeout=60_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except PWTimeout:
            pass
        title = await page.title()
        try:
            body = await page.inner_text("body", timeout=5_000)
        except Exception:
            html = await page.content()
            body = re.sub(r"<[^>]+>", " ", html)
        text = clean_whitespace(body)[:MAX_CHARS]
        return title or url, text
    except Exception as e:
        logger.error(f"FETCH ERROR → {url} : {e}")
        return "", ""

async def run(query: str, out_path: str = "data/scraped.json", headless: bool = True):
    ensure_dirs()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=["--no-sandbox","--disable-gpu"])
        context = await browser.new_context(user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ))
        page = await context.new_page()

        # Optional stealth (best effort)
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
        except Exception:
            pass

        serp = await _gather_google(page, query)
        docs, sid = [], 1
        for r in serp:
            title, text = await _scrape_page(page, r["url"])
            if text:
                docs.append({"source_id": sid, "url": r["url"], "title": title or r["title"], "text": text})
                sid += 1
            time.sleep(0.7)
        await browser.close()

    save_json(out_path, docs)
    logger.info(f"WROTE → {out_path} ({len(docs)} docs)")
    return out_path

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="data/scraped.json")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.query, args.out, args.headless))
