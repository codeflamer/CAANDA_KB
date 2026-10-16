"""Re-verify eval-set gold quotes against a new crawl snapshot. Report only: never edits the eval sets.

    .venv/Scripts/python scripts/check_gold.py --new data/raw/<date>
    .venv/Scripts/python scripts/check_gold.py --new data/raw/<date> --eval-file data/eval_set/eval_set_v3.jsonl

content_hash is a hash of the raw HTML, so it also changes on markup-only edits: every quote is re-matched
against the new page text ("..." fragments, word tokens: same matching as retrieval_eval.py).

Per gold passage:  unchanged   same content_hash as the gold
                   changed-ok  page changed, quote still found (new content_hash suggested)
                   BROKEN      quote no longer on the page
                   MISSING     page not in the new snapshot (not crawled, or fetch failed)
                   BAD-GOLD    quote not even in the old snapshot (an eval-set bug, not drift)
Per item (stale flags): temporal items and single-fact items with stale_risk high whose page changed, and items
whose valid_until has passed. Exit code 1 if anything is BROKEN, MISSING or BAD-GOLD.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import retrieval_eval as RE  # noqa: E402

DEFAULT_EVAL_FILES = [RE.EVAL_FILE, ROOT / "data/eval_set/eval_set_v3.jsonl"]
FAILING = ("BROKEN", "MISSING", "BAD-GOLD")


class Snapshot:
    """Page lookup. Several pages share a URL path and differ only in the query string (DLI details-flight.html?q=..),
    and gold URLs are stored without it, so: exact URL first, then the path only when it names one page."""

    def __init__(self, run_dir):
        self.pages = RE.load_pages(Path(run_dir))
        self.by_hash = {p["content_hash"]: p for p in self.pages if p.get("content_hash")}
        self.exact, by_path = {}, {}
        for p in self.pages:
            for u in {p["url"], p.get("final_url")} - {None}:
                self.exact[u.strip()] = p
                by_path.setdefault(RE.norm_url(u), {})[p["key"]] = p
        self.by_path = {u: next(iter(ps.values())) for u, ps in by_path.items() if len(ps) == 1}

    def by_url(self, url):
        return self.exact.get(url.strip()) or self.by_path.get(RE.norm_url(url))


def quote_found(quote, page):
    text = RE.norm_text(page["text"])
    return all(f in text for f in RE.quote_fragments(quote))


def load_items(paths):
    items = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                items.setdefault(item["id"], {**item, "_file": Path(path).name})  # v3 repeats the eval_set items
    return list(items.values())


def check_item(item, old, new, today):
    passages, flags = [], []
    page_changed = False
    for j, g in enumerate(item.get("gold_passages") or [], 1):
        o = old.by_hash.get(g.get("content_hash")) or old.by_url(g["url"])  # the exact page the gold was written on
        n = new.by_url(o["url"]) if o else new.by_url(g["url"])
        entry = {"passage": j, "url": o["url"] if o else g["url"], "old_hash": g.get("content_hash")}
        if o is None or not quote_found(g["quote"], o):
            entry["status"] = "BAD-GOLD"
        elif n is None:
            entry["status"] = "MISSING"
        elif n.get("content_hash") == g.get("content_hash"):
            entry["status"] = "unchanged"
        else:
            entry["status"] = "changed-ok" if quote_found(g["quote"], n) else "BROKEN"
            entry["new_hash"] = n.get("content_hash")
        page_changed |= entry["status"] != "unchanged"
        passages.append(entry)
    if page_changed and item.get("category") == "temporal":
        flags.append("temporal item on a changed page: re-check the rule and its dates")
    if page_changed and item.get("stale_risk") == "high":
        flags.append("stale_risk high and the page changed: re-check the answer")
    if item.get("valid_until") and str(item["valid_until"]) < today:
        flags.append(f"valid_until {item['valid_until']} has passed")
    return {"id": item["id"], "category": item.get("category"), "file": item["_file"], "passages": passages, "flags": flags}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="new crawl snapshot dir (with manifest.jsonl)")
    ap.add_argument("--old", default=str(RE.RUN_DIR), help="snapshot the eval set was written on (default: %(default)s)")
    ap.add_argument("--eval-file", nargs="+", default=[str(p) for p in DEFAULT_EVAL_FILES])
    ap.add_argument("--out", help="report path (default: data/eval_set/gold_check_<new dir name>.jsonl)")
    args = ap.parse_args()

    old, new = Snapshot(args.old), Snapshot(args.new)
    today = date.today().isoformat()
    report = [check_item(i, old, new, today) for i in load_items(args.eval_file)]

    out = Path(args.out) if args.out else ROOT / f"data/eval_set/gold_check_{Path(args.new).name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in report:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    statuses = [p["status"] for r in report for p in r["passages"]]
    counts = {s: statuses.count(s) for s in ("unchanged", "changed-ok", *FAILING)}
    print(f"GOLD CHECK {args.old} -> {args.new}: {len(report)} items, {len(statuses)} gold passages")
    RE.print_table([list(counts), list(counts.values())])
    problems = [[r["id"], p["passage"], p["status"], p["url"].rsplit("/", 1)[-1], p.get("new_hash", "")]
                for r in report for p in r["passages"] if p["status"] != "unchanged"]
    if problems:
        print("\nCHANGED PASSAGES")
        RE.print_table([["item", "passage", "status", "page", "new content_hash"]] + problems)
    flagged = [[r["id"], r["category"], "; ".join(r["flags"])] for r in report if r["flags"]]
    if flagged:
        print("\nSTALE FLAGS")
        RE.print_table([["item", "category", "flags"]] + flagged, wrap={2: 70})
    print(f"\nreport written to {out}")
    sys.exit(1 if any(counts[s] for s in FAILING) else 0)


if __name__ == "__main__":
    main()
