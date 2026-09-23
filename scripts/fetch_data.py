import hashlib
import json
import re
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

SKIP_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".jpg", ".jpeg", ".png", ".gif", ".mp4")
TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}


def normalize_url(url):
    """Make equivalent URLs identical so the visited set works."""
    url, _ = urldefrag(url)  # drop #fragment
    p = urlparse(url)
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING_PARAMS))
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def url_key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def load_robots(client, base, user_agent):
    """Return a RobotFileParser for the host. Missing robots.txt means allow all."""
    rp = RobotFileParser()
    try:
        r = client.get(f"{base}/robots.txt")
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except httpx.HTTPError:
        rp.parse([])
    return rp


def extract_main(soup):
    """Return the main content node. Falls back to the whole body."""
    return soup.find("main") or soup.find(id="container") 


def extract_text(main):
    """v0 text extraction: strip boilerplate tags, keep line breaks so headings/lists/tables stay readable."""
    for tag in main.find_all(["script", "style", "nav", "aside", "form", "noscript"]):
        tag.decompose()
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "dt", "dd"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append("\n" + "#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        elif el.name == "tr":
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in el.find_all(["th", "td"])]
            lines.append(" | ".join(cells))
        else:
            lines.append(text)
    # li/tr contents can be duplicated by nested p tags; drop consecutive duplicates
    out = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return "\n".join(out).strip()


def extract_date_modified(soup):
    ## Something wrong with date modification
    """Best effort. Selectors are a guess based on how canada.ca pages usually look. Verify on a real page."""
    node = soup.find("gcds-date-modified", attrs={"property": "dateModified"})
    return node.get_text(strip=True) if node else None


def crawl(
    seeds,
    allowed_netloc,
    allowed_path_prefixes,
    max_depth=2,
    max_pages=150,
    delay=2.0,
    out_dir="../data/raw",
    user_agent="canada-guidance-rag research (contact: anonymouspelumi.com)",
    scheme="https",
    verbose=True,
):
    """
    Breadth-first crawl from the seeds.
    Returns (manifest, skipped): one dict per fetched page and one dict per link rejected by a filter.
    """
    run_dir = Path(out_dir) / date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = [normalize_url(s) for s in seeds]
    queue = deque((s, 0, None, s) for s in seeds)  # (url, depth, parent_url, seed_url)
    queued = set(seeds)  # everything ever put in the queue = the visited set
    manifest, skipped = [], []

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True, timeout=30) as client:
        robots = load_robots(client, f"{scheme}://{allowed_netloc}", user_agent)

        while queue and len(manifest) < max_pages:
            url, depth, parent, seed = queue.popleft()

            if not robots.can_fetch(user_agent, url):
                skipped.append({"url": url, "parent": parent, "reason": "blocked_by_robots"})
                continue

            record = {"url": url, "depth": depth, "parent": parent, "seed": seed,
                      "fetched_at": datetime.now(timezone.utc).isoformat()}
            try:
                r = client.get(url)
            except httpx.HTTPError as e:
                record.update(status=None, error=str(e))
                manifest.append(record)
                time.sleep(delay)
                continue

            record["status"] = r.status_code
            record["final_url"] = normalize_url(str(r.url))
            ctype = r.headers.get("content-type", "")

            if r.status_code != 200 or "html" not in ctype:
                record["error"] = f"not_html_or_not_ok ({ctype})"
                manifest.append(record)
                time.sleep(delay)
                continue

            key = url_key(url)
            final_dir = run_dir / f"{key}"
            final_dir.mkdir(parents=True,exist_ok=True)
            
            content_hash = hashlib.sha256(r.content).hexdigest()[:16]
            (final_dir / f"{key}.html").write_bytes(r.content)

            soup = BeautifulSoup(r.content, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else None
            date_modified = extract_date_modified(soup)
            main = extract_main(soup)

            # collect links from main content only, so global nav/footer links don't pull the crawl off-topic
            links = [a["href"] for a in main.find_all("a", href=True)]
            text = extract_text(main)
            (final_dir / f"{key}.txt").write_text(text, encoding="utf-8")

            record.update(key=key, content_hash=content_hash, title=title,
                          date_modified=date_modified, n_links=len(links), n_chars=len(text))
            manifest.append(record)
            if verbose:
                print(f"[d{depth}] {len(manifest):>3} {url}")

            if depth < max_depth:
                for href in links:
                    if href.startswith(("mailto:", "tel:", "javascript:")):
                        continue
                    nxt = normalize_url(urljoin(url, href))
                    p = urlparse(nxt)
                    if nxt in queued:
                        continue
                    if p.netloc != allowed_netloc:
                        reason = "off_host"
                    elif p.path.lower().endswith(SKIP_EXTENSIONS):
                        reason = "non_html_extension"
                    elif not any(p.path.startswith(pref) for pref in allowed_path_prefixes):
                        reason = "off_path"
                    else:
                        queued.add(nxt)
                        queue.append((nxt, depth + 1, url, seed))
                        continue
                    skipped.append({"url": nxt, "parent": url, "reason": reason})

            time.sleep(delay)

    if queue and verbose:
        print(f"Stopped at max_pages={max_pages} with {len(queue)} URLs still queued.")

    # write logs next to the snapshots
    with open(run_dir / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "skipped.jsonl", "w") as f:
        for row in skipped:
            f.write(json.dumps(row) + "\n")
    return manifest, skipped



## Set config
SEED_URLS = [
    "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit.html"
]
ALLOWED_NETLOC = "www.canada.ca"
ALLOWED_PATH_PREFIXES = [
   "/en/immigration-refugees-citizenship/services/study-canada"
]
MAX_DEPTH = 2
MAX_PAGES = 150
DELAY_SECONDS = 5.0
USER_AGENT = "canada-guidance-rag research (contact: anonymouspelumi@gmail.com)"

assert not any(s.startswith("TODO") for s in SEED_URLS), "Fill in SEED_URLS"
assert ALLOWED_PATH_PREFIXES, "Fill in ALLOWED_PATH_PREFIXES"

## RUN
manifest, skipped = crawl(
    seeds=SEED_URLS,
    allowed_netloc=ALLOWED_NETLOC,
    allowed_path_prefixes=ALLOWED_PATH_PREFIXES,
    max_depth=MAX_DEPTH,
    max_pages=MAX_PAGES,
    delay=DELAY_SECONDS,
    user_agent=USER_AGENT,
)