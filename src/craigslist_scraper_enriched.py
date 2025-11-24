# craigslist_scraper_enriched.py
import argparse, time, re, csv, json, urllib.parse, requests, os
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

PRICE_RE = re.compile(r'\$[\s]*([0-9][0-9,]*)')
SQFT_RE  = re.compile(r'(\d{3,5})\s*(?:ft2|ft²|sqft|square\s*feet)', re.I)
BEDS_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:br|bd|bed)', re.I)
BATH_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:ba|bath)', re.I)
FURNISH_KW = re.compile(r'\b(furnished|fully[-\s]?furnished|unfurnished)\b', re.I)
PETS_KW    = re.compile(r'\b(pets?\s*ok|pet[-\s]?friendly|cats?\s*ok|dogs?\s*ok|no\s*pets)\b', re.I)
UTIL_KW    = re.compile(r'\b(utilities?\s*included|hydro\s*included|heat\s*included|internet\s*included|all[-\s]?inclusive)\b', re.I)
PARK_KW    = re.compile(r'\b(parking|parking\s*included|street\s*parking|underground\s*parking|no\s*parking)\b', re.I)

COLS = ["title","price","beds","baths","sqft","furnished","pets_allowed",
        "utilities_included","parking_available","city","lat","lon",
        "post_date","url","full_text"]

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def http_get(url, tries=3, backoff=1.2, timeout=25):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.ok and r.text:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(backoff * (i + 1))
    return None

def parse_jsonld_search(html):
    """Robust JSON-LD parse with fallback."""
    soup = BeautifulSoup(html, "html.parser")
    # 1) preferred by id
    candidates = []
    node = soup.find("script", id="ld_searchpage_results")
    if node and node.string:
        candidates.append(node.string)
    # 2) fallback: any ld+json ItemList/SearchResultsPage
    if not candidates:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            if not s.string: continue
            try:
                data = json.loads(s.string)
            except Exception:
                continue
            if isinstance(data, dict) and ("itemListElement" in data):
                candidates.append(s.string); break
    if not candidates:
        return []
    try:
        data = json.loads(candidates[0])
    except Exception:
        return []
    out = []
    for elem in data.get("itemListElement", []):
        it = elem.get("item", {})
        out.append({
            "title": it.get("name"),
            "lat": it.get("latitude"),
            "lon": it.get("longitude"),
            "beds": it.get("numberOfBedrooms"),
            "baths": it.get("numberOfBathroomsTotal"),
            "city": (it.get("address") or {}).get("addressLocality"),
        })
    return out

def parse_price_from_title(title):
    if not title: return None
    m = PRICE_RE.search(title)
    return int(m.group(1).replace(",", "")) if m else None

def scrape_post(url):
    html = http_get(url)
    if not html: return {}
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(x.get_text(" ", strip=True) for x in soup.select("section, p, li, span, h1, h2"))

    furnished = None
    if FURNISH_KW.search(text):
        furnished = bool(re.search(r'\b(unfurnished)\b', text, re.I) is None)

    pets_allowed = None
    if PETS_KW.search(text):
        pets_allowed = not bool(re.search(r'\b(no\s*pets)\b', text, re.I))

    utilities_included = bool(UTIL_KW.search(text))
    parking_available  = bool(PARK_KW.search(text) and re.search(r'\bno\s*parking\b', text, re.I) is None)

    sqft = None
    m = SQFT_RE.search(text)
    if m:
        try: sqft = int(m.group(1))
        except: pass

    beds2 = None
    m = BEDS_RE.search(text)
    if m:
        try: beds2 = float(m.group(1))
        except: pass

    baths2 = None
    m = BATH_RE.search(text)
    if m:
        try: baths2 = float(m.group(1))
        except: pass

    t = soup.select_one('time[datetime]')
    date_iso = t['datetime'] if t and t.has_attr('datetime') else None

    return {
        "full_text": text[:5000],
        "furnished": furnished,
        "pets_allowed": pets_allowed,
        "utilities_included": utilities_included,
        "parking_available": parking_available,
        "sqft_text": sqft,
        "beds_text": beds2,
        "baths_text": baths2,
        "post_date": date_iso
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_url", default="https://vancouver.craigslist.org/search/apa")
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--outfile", default="data/rentals_raw.csv")
    ap.add_argument("--no-follow", action="store_true", help="Do NOT open each listing page (faster test)")
    ap.add_argument("--min_price", type=int, default=600)
    ap.add_argument("--max_price", type=int, default=3000)
    ap.add_argument("--max_days", type=int, default=30, help="Only keep posts within last N days (follow mode)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    new_file = not os.path.exists(args.outfile)

    # load seen URLs to avoid duplicates across runs
    seen = set()
    if not new_file:
        try:
            with open(args.outfile, newline="", encoding="utf-8") as f0:
                for r in csv.DictReader(f0):
                    if r.get("url"):
                        seen.add(r["url"])
        except FileNotFoundError:
            pass

    f = open(args.outfile, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=COLS)
    if new_file:
        writer.writeheader(); f.flush(); log(f"CSV created: {args.outfile}")

    # build page URL with price filter
    u = urllib.parse.urlparse(args.start_url)
    base = urllib.parse.urlunparse(u._replace(fragment=""))
    base = re.sub(r"([?&])s=\d+", r"\1", base)
    qsep = "&" if "?" in base else "?"
    base = f"{base}{qsep}min_price={args.min_price}&max_price={args.max_price}"
    page_url = base + ("&s={o}" if "?" in base else "?s={o}")

    total = 0
    for p, o in enumerate(range(0, args.pages * 120, 120), start=1):
        t0 = time.time()
        url = page_url.format(o=o)
        log(f"PAGE {p}/{args.pages} → {url}")
        html = http_get(url, backoff=args.delay)
        if not html:
            log("  [WARN] failed to load page; skipping")
            continue

        if p == 1:
            with open("debug_first_page.html", "w", encoding="utf-8") as dbg:
                dbg.write(html)

        spine = parse_jsonld_search(html)
        soup = BeautifulSoup(html, "html.parser")
        anchors = [a.get("href") for a in soup.select('a[href*="/apa/"]')]
        log(f"  found {len(spine)} jsonld items; {len(anchors)} anchors")

        page_rows = 0
        for i, s in enumerate(spine):
            href = anchors[i] if i < len(anchors) else None
            abs_url = urllib.parse.urljoin(base, href) if href else None

            rec = {
                "title": s.get("title"),
                "price": parse_price_from_title(s.get("title")),
                "beds": s.get("beds"),
                "baths": s.get("baths"),
                "lat": s.get("lat"),
                "lon": s.get("lon"),
                "city": s.get("city"),
                "url": abs_url
            }

            # price pre-filter
            if rec["price"] is not None and not (args.min_price <= rec["price"] <= args.max_price):
                continue

            # dedupe
            if rec.get("url") and rec["url"] in seen:
                continue

            if abs_url and not args.no_follow:
                extra = scrape_post(abs_url)
                rec.update(extra)
                if rec["beds"] is None and extra.get("beds_text") is not None:
                    rec["beds"] = extra["beds_text"]
                if rec["baths"] is None and extra.get("baths_text") is not None:
                    rec["baths"] = extra["baths_text"]
                if extra.get("sqft_text") is not None:
                    rec["sqft"] = extra["sqft_text"]

                # recent N days filter (follow mode only)
                if args.max_days and rec.get("post_date"):
                    try:
                        dt = datetime.fromisoformat(rec["post_date"].replace("Z", "+00:00"))
                        if datetime.now(timezone.utc) - dt > timedelta(days=args.max_days):
                            continue
                    except Exception:
                        pass

            out = {c: rec.get(c) for c in COLS}
            writer.writerow(out)
            seen.add(rec.get("url"))
            page_rows += 1
            total += 1

            # only sleep per-item when following detail pages
            if not args.no_follow:
                time.sleep(args.delay)

            if total % 25 == 0:
                f.flush()
                log(f"  progress: rows={total} (this page: {page_rows})")

        f.flush()
        # page-level small delay
        time.sleep(args.delay)
        log(f"PAGE DONE {p}/{args.pages} | rows this page={page_rows} | total={total} | time={time.time()-t0:.1f}s")

    f.close()
    log(f"[DONE] wrote {total} rows → {args.outfile}")

if __name__ == "__main__":
    main()
