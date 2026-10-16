"""Build eval set v3: the 30 human-verified eval_set items (split=dev) + 5 new items per category (split=test),
taken from the LLM drafts, auto-checked, then approved by a human. Never edits the draft files or eval_set.jsonl.

    .venv/Scripts/python scripts/build_eval_set.py candidates
        auto-checks every draft, picks the first 5 passing new drafts per category (skipping --reject ids),
        writes data/eval_set/candidates_v3.jsonl + review_v3.md for the human review
    .venv/Scripts/python scripts/build_eval_set.py freeze --approve ID [ID ...]
        writes data/eval_set/eval_set_v3.jsonl; refuses unless every category has 10 items
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import retrieval_eval as RE  # noqa: E402

EVAL_DIR = ROOT / "data/eval_set"
DRAFTS = {  # category -> draft file
    "single-fact": EVAL_DIR / "eval_set_singlefact.jsonl",
    "multi-hop": EVAL_DIR / "draft_multihop_v0.jsonl",
    "conditional": EVAL_DIR / "draft_conditional_v0.jsonl",
    "temporal": EVAL_DIR / "draft_temporal_v0.jsonl",
    "ambiguous": EVAL_DIR / "draft_ambiguous_v0.jsonl",
    "unanswerable": EVAL_DIR / "draft_unanswerable_v0.jsonl",
}
CANDIDATES = EVAL_DIR / "candidates_v3.jsonl"
REVIEW = EVAL_DIR / "review_v3.md"
OUT = EVAL_DIR / "eval_set_v3.jsonl"
NEW_PER_CATEGORY = 5
CONFIG = ROOT / "configs/ircc_full_bge-base_c180_dedup_rerank_msmarco_k100_split_rr_filter_expand_exact.yaml"


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def check(d, category, indexed_urls, pages, existing_quotes):
    """(reasons, flags) for one draft. Reasons fail the draft; flags are shown to the reviewer."""
    reasons, flags = [], []
    if d.get("category") != category:
        reasons.append(f"category {d.get('category')!r}, expected {category!r}")
    if not (d.get("answer") or d.get("gold_answer")):
        reasons.append("missing answer / gold_answer")
    if category == "unanswerable":
        if d.get("answerable") is not False:
            reasons.append("unanswerable item must have answerable: false")
        if d.get("gold_passages"):
            reasons.append("unanswerable item must have no gold passages")
        return reasons, flags
    if d.get("answerable") is not True:
        reasons.append("answerable item must have answerable: true")
    # quote in page snapshot, URL indexed, schema: same checks as the multi-hop draft validation
    entry = RE.validate_drafts([json.dumps(d)], [], indexed_urls, pages)[0]
    reasons += entry["reasons"]
    if category == "multi-hop":
        flags += entry["flags"]
    known_hashes = {p.get("content_hash") for p in pages}
    for j, g in enumerate(d.get("gold_passages") or [], 1):
        if g.get("content_hash") not in known_hashes:
            flags.append(f"passage {j}: content_hash {g.get('content_hash')} not in the current snapshot (stale?)")
        if RE.norm_text(g.get("quote", "")) in existing_quotes:
            reasons.append(f"passage {j}: same gold quote as an existing eval_set item")
    return reasons, flags


def cmd_candidates(args):
    eval_items = read_jsonl(RE.EVAL_FILE)
    eval_ids = {i["id"] for i in eval_items}
    existing_quotes = {RE.norm_text(g["quote"]) for i in eval_items for g in i.get("gold_passages") or []}
    name, cfg = RE.load_config(CONFIG)
    _, _, indexed_urls, _, _ = RE.open_eval_collection(name, cfg, CONFIG)
    pages = RE.load_pages(RE.RUN_DIR)
    reject = set(args.reject or [])

    picked, failed = [], []
    for category, path in DRAFTS.items():
        n = 0
        for d in read_jsonl(path):
            if d["id"] in eval_ids or n >= NEW_PER_CATEGORY:
                continue
            if d["id"] in reject:
                failed.append((d["id"], "rejected by reviewer"))
                continue
            reasons, flags = check(d, category, indexed_urls, pages, existing_quotes)
            if reasons:
                failed.append((d["id"], "; ".join(reasons)))
                continue
            picked.append({"item": d, "source_file": path.name, "flags": flags})
            n += 1
        if n < NEW_PER_CATEGORY:
            print(f"WARNING: only {n} passing new drafts for {category}")

    with open(CANDIDATES, "w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    write_review(picked, failed)
    print(f"{len(picked)} candidates -> {CANDIDATES.relative_to(ROOT)}, review sheet -> {REVIEW.relative_to(ROOT)}")
    print(dict(Counter(c["item"]["category"] for c in picked)))
    for item_id, why in failed:
        print(f"  skipped {item_id}: {why}")


def write_review(picked, failed):
    out = ["# Eval set v3: review of new items", "",
           "Approve or reject each item. Approved ids are passed to `build_eval_set.py freeze --approve ...`;",
           "rejected ids to `candidates --reject ...` (the next passing draft takes their place).", ""]
    category = None
    for c in picked:
        d = c["item"]
        if d["category"] != category:
            category = d["category"]
            out += [f"## {category}", ""]
        out += [f"### {d['id']}  ({c['source_file']})", "", f"**Q:** {d['question']}", "",
                f"**Answer:** {d.get('gold_answer') or d.get('answer')}", ""]
        if d.get("required_facts"):
            out += ["**Required facts:**", *(f"- {x}" for x in d["required_facts"]), ""]
        for j, g in enumerate(d.get("gold_passages") or [], 1):
            out += [f"**Gold {j}:** {g['url']}", "", f"> {g['quote']}", ""]
        if d.get("notes"):
            out += [f"*Notes:* {d['notes']}", ""]
        if c["flags"]:
            out += [f"⚠ {f}" for f in c["flags"]] + [""]
        out += ["- [ ] approve   - [ ] reject", ""]
    if failed:
        out += ["## Drafts skipped by the auto-check", "", *(f"- {i}: {why}" for i, why in failed), ""]
    REVIEW.write_text("\n".join(out), encoding="utf-8")


def cmd_freeze(args):
    approve = set(args.approve)
    candidates = read_jsonl(CANDIDATES)
    unknown = approve - {c["item"]["id"] for c in candidates}
    if unknown:
        sys.exit(f"not in {CANDIDATES.name}: {sorted(unknown)}")
    rows = [{**i, "split": "dev"} for i in read_jsonl(RE.EVAL_FILE)]
    for c in candidates:
        if c["item"]["id"] in approve:
            rows.append({**c["item"], "verified_by": "human", "split": "test", "source_file": c["source_file"]})
    counts = Counter(r["category"] for r in rows)
    short = {k: v for k, v in counts.items() if v < 10}
    if short or len(counts) < len(DRAFTS):
        sys.exit(f"every category needs 10 items, got {dict(counts)}; reject + re-run candidates first")
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} items -> {OUT.relative_to(ROOT)}  {dict(counts)}  "
          f"splits {dict(Counter(r['split'] for r in rows))}, scored {sum(bool(r.get('gold_passages')) for r in rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidates")
    p.add_argument("--reject", nargs="*", help="draft ids the reviewer rejected")
    p.set_defaults(fn=cmd_candidates)
    p = sub.add_parser("freeze")
    p.add_argument("--approve", nargs="+", required=True)
    p.set_defaults(fn=cmd_freeze)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
