import json
from pathlib import Path
# import httpx
# from bs4 import BeautifulSoup
# from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse

# url = "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit.html"
# response = httpx.get(url)

# print(response.status_code)
# print(response.headers.get("content-type", ""))
# print(response.content)

# soup = BeautifulSoup(response.content, "html.parser")
# title = soup.title.get_text(strip=True) if soup.title else None

# print(soup)
# data_modified = soup.find("gcds-date-modified")
# print(data_modified.get_text(strip=True))

# main = soup.find("main") or soup.find(id="container") 
# print(main)

# def normalize_url(url):
#     """Make equivalent URLs identical so the visited set works."""
#     url, _ = urldefrag(url)  # drop #fragment
#     p = urlparse(url)
#     query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING_PARAMS))
#     path = p.path.rstrip("/") or "/"
#     return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))

# links = [a["href"] for a in main.find_all("a", href=True)]
# for link in links:
#     print(link)
# for link in links:
#     nxt = normalize_url(urljoin("https://www.canada.ca/",link))
#     print(nxt)

# link_1 = nxt
# path = urlparse(link_1).path
# print(path.startswith("/en"))



# from pathlib import Path
# from datetime import date, datetime, timezone
# import hashlib

# out_dir = "../data/raw"
# run_dir = Path(out_dir) / date.today().isoformat()

# def url_key(url):
#     return hashlib.sha256(url.encode()).hexdigest()[:16]
# key = url_key(link_1)
# fll_dir = run_dir / f"{key}"
# print(fll_dir)
# fll_dir.mkdir(parents=True,exist_ok=True)
# new_dir = fll_dir / "name"
# (fll_dir / f"{key}.html").write_bytes(response.content)
# new_dir.mkdir(parents=True,exist_ok=True)

items = [json.loads(i) for i in Path("../results/20260930-130650.jsonl").read_text(encoding="utf-8").splitlines()]
items

with open("../data/eval_set/eval_set_singlefact.jsonl", "r", encoding="utf-8") as f:
    gold_standard = json.loads(f.readlines()[0])

print(gold_standard)
print(gold_standard.keys())
# dict_keys(['id', 'question', 'answer', 'category', 'answerable', 'valid_as_of', 'valid_until', 'stale_risk', 'source', 'verified_by', 'gold_passages'])

golds = gold_standard.get("gold_passages")
print(golds)

print(gold_standard.keys())

## Play with rigthpage and right passage
# dict_keys(['id', 'category', 'question', 'answer', 'chunks', 'url_retrieved', 'quote_retrieved', 'url_rank', 'quote_ranks', 'verdict', 'reason', 'grounded', 'unsupported_claims', 'failure_type'])
# print(list(items[0].keys()))
print(items[0]["question"])
print(items[0]["category"])

chunks = items[0]["chunks"]
print(len(chunks))

ok_urls = set()
for g in golds:
    ok_urls.add(g.get("url"))

print(ok_urls)

url_rank = next((c["rank"] for c in chunks if c["url"].strip() in ok_urls), None)

url_rank = next((c["rank"] for c in chunks if c["url"].strip() in ok_urls), None)
print(url_rank)