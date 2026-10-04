"""Retrieval-only eval: no LLM calls. Optional local cross-encoder reranking (`reranker:` in the config).

Usage (from repo root):
    .venv/Scripts/python retrieval_eval.py index   --config configs/ircc_full_minilm_c180.yaml [--rebuild]
    .venv/Scripts/python retrieval_eval.py eval    --config configs/ircc_full_minilm_c180.yaml --change "what I changed"
    .venv/Scripts/python retrieval_eval.py compare [<run_id_1> <run_id_2>]   (default: previous vs latest run)
    .venv/Scripts/python retrieval_eval.py table   [run_id ...]
"""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
import textwrap
import traceback
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "data/raw/2026-09-20"
EVAL_FILE = ROOT / "data/eval_set/eval_set.jsonl"
CHROMA_PATH = ROOT / "indexes/retrieval_eval"
EMBED_CACHE = ROOT / "cache/embeddings.sqlite"
RESULTS_DIR = ROOT / "retrieval_results"
RUNS_FILE = ROOT / "retrieval_runs.jsonl"

# Reuse the chunking from the main RAG code so the chunks are identical.
sys.path.insert(0, str(ROOT / "scripts"))
from add_to_vdb import chunk_page  # noqa: E402

DEFAULTS = {"chunking_method": "section_words", "add_heading_path": True, "top_k_retrieve": 50, "reranker": None,
            "dedupe_chunks": False, "query_prefix": ""}
# Settings baked into a collection at index time; eval refuses a config that disagrees with the collection.
INDEX_SETTINGS = ("embedding_model", "chunking_method", "chunk_size", "chunk_overlap", "add_heading_path", "dedupe_chunks")


# ---------- config ----------
def load_config(path):
    path = Path(path)
    cfg = {**DEFAULTS, **yaml.safe_load(path.read_text(encoding="utf-8"))}
    for key in ("embedding_model", "chunk_size", "chunk_overlap"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required setting {key!r}")
    # Collection is named after the config file, unless the config reuses another one
    # (e.g. a rerank-only config, which doesn't change the index). Kept out of cfg so it isn't in config_diff.
    return cfg.pop("collection", None) or path.stem, cfg


# ---------- embeddings (with on-disk cache) ----------
def make_embedder(model_name):
    """Returns embed(list[str]) -> list of vectors. Same model for docs and queries, like add_to_vdb."""
    if model_name == "chroma-default-all-MiniLM-L6-v2":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        ef = DefaultEmbeddingFunction()
        return lambda texts: [np.asarray(v, dtype=np.float32) for v in ef(texts)]
    from sentence_transformers import SentenceTransformer  # only needed for other models
    m = SentenceTransformer(model_name)
    return lambda texts: list(m.encode(texts, batch_size=32, show_progress_bar=False).astype(np.float32))


def embed_cached(texts, model_name, embed, batch_size=64):
    """Embed texts, reusing vectors cached by (model, sha256(text)). Returns (vectors, n_newly_embedded)."""
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(EMBED_CACHE)
    db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, text_hash TEXT, vec BLOB, PRIMARY KEY (model, text_hash))")
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    cached = {}
    for i in range(0, len(hashes), 500):
        part = hashes[i : i + 500]
        q = f"SELECT text_hash, vec FROM emb WHERE model = ? AND text_hash IN ({','.join('?' * len(part))})"
        cached.update({h: np.frombuffer(v, dtype=np.float32) for h, v in db.execute(q, [model_name, *part])})
    todo = list({h: t for h, t in zip(hashes, texts) if h not in cached}.items())
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        vecs = embed([t for _, t in batch])
        rows = [(model_name, h, np.asarray(v, dtype=np.float32).tobytes()) for (h, _), v in zip(batch, vecs)]
        db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows)
        db.commit()
        cached.update({h: np.asarray(v, dtype=np.float32) for (h, _), v in zip(batch, vecs)})
    db.close()
    return [cached[h] for h in hashes], len(todo)


# ---------- index ----------
def load_pages(run_dir):
    pages = []
    for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != 200 or not row.get("key"):
            continue
        text = (run_dir / row["key"] / f"{row['key']}.txt").read_text(encoding="utf-8")
        if text.strip():
            pages.append({**row, "text": text})
    return pages


def make_chunks(pages, cfg):
    if cfg["chunking_method"] != "section_words":
        raise ValueError(f"unknown chunking_method {cfg['chunking_method']!r} (supported: section_words)")
    chunks = []
    for page in pages:
        for c in chunk_page(page, max_words=cfg["chunk_size"], overlap_words=cfg["chunk_overlap"]):
            text = c["text"]
            if not cfg["add_heading_path"]:
                # chunk_page writes "title | heading\n<body>"; keep only the body
                text = text.split("\n", 1)[1] if "\n" in text else text
            meta = {**c["meta"], "content_hash": page.get("content_hash") or "", "chunk_id": c["id"],
                    "heading_path": c["meta"].get("heading", "")}
            chunks.append({"id": c["id"], "text": text, "meta": meta})
    return dedupe(chunks) if cfg["dedupe_chunks"] else chunks


def dedupe(chunks):
    """Keep the first chunk of each group with the same words (norm_text), in manifest order.
    The kept chunk records how many copies there were and the other pages' URLs (JSON string:
    Chroma metadata values must be scalars). DLI detail pages share whole PGWP sections word for word."""
    groups = {}
    for c in chunks:
        groups.setdefault(norm_text(c["text"]), []).append(c)
    kept = []
    for copies in groups.values():
        first = copies[0]
        other_urls = sorted({c["meta"]["url"] for c in copies[1:]} - {first["meta"]["url"]})
        first["meta"].update(n_copies=len(copies), also_urls=json.dumps(other_urls))
        kept.append(first)
    return kept


def cmd_index(args):
    name, cfg = load_config(args.config)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if name in [c.name for c in client.list_collections()]:
        if not args.rebuild:
            sys.exit(f"Collection {name!r} already exists. Pass --rebuild to overwrite it.")
        client.delete_collection(name)

    pages = load_pages(RUN_DIR)
    chunks = make_chunks(pages, cfg)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    if cfg["dedupe_chunks"]:
        print(f"  dedupe: {sum(c['meta']['n_copies'] - 1 for c in chunks)} duplicate chunks dropped")
    vecs, n_new = embed_cached([c["text"] for c in chunks], cfg["embedding_model"], make_embedder(cfg["embedding_model"]))
    print(f"embedded {n_new} new chunks, {len(chunks) - n_new} from cache")

    col = client.create_collection(
        name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
        metadata={"embed_model": cfg["embedding_model"], "config": json.dumps(cfg, sort_keys=True)},
    )
    for i in range(0, len(chunks), 256):
        b = slice(i, i + 256)
        col.add(ids=[c["id"] for c in chunks[b]], documents=[c["text"] for c in chunks[b]],
                metadatas=[c["meta"] for c in chunks[b]], embeddings=[v.tolist() for v in vecs[b]])
    print(f"collection {name!r}: {col.count()} chunks from {len({c['meta']['url'] for c in chunks})} pages")


# ---------- normalization ----------
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"'})


def norm_text(s):
    # Compare word tokens only: the crawler rewrites list markup ("criteria :\n- Associate of Arts"),
    # so punctuation, bullets and whitespace differ from the quote as written in the eval set.
    # Padding with spaces makes `in` match on whole words.
    s = unicodedata.normalize("NFKC", s or "").translate(QUOTE_MAP).lower()
    return " " + " ".join(re.findall(r"\w+", s)) + " "


def quote_fragments(quote):
    """A gold quote may skip text with "..." (see readme_eval.md). Each piece is matched on its own."""
    return [f for f in (norm_text(p) for p in re.split(r"\.\.\.|…", quote)) if f.strip()]


def first_rank(fragments, texts):
    """Rank (1-based) of the first text containing the fragment, worst over all fragments; None if any is missing."""
    ranks = [next((i for i, t in enumerate(texts, 1) if f in t), None) for f in fragments]
    return None if None in ranks else max(ranks)


def norm_url(u):
    return (u or "").strip().split("?", 1)[0]


# ---------- coverage ----------
def load_eval_items():
    items = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [i for i in items if i.get("gold_passages")]  # unanswerable items have no gold: skip


def coverage_check(items, indexed_urls, n_pages):
    missing = []
    for item in items:
        for g in item["gold_passages"]:
            if norm_url(g["url"]) not in indexed_urls:
                missing.append((item["id"], norm_url(g["url"])))
    n_gold = sum(len(i["gold_passages"]) for i in items)
    # DLI detail pages differ only by ?q=..., so dropping query strings merges them
    print(f"\nCOVERAGE: {n_pages} pages indexed ({len(indexed_urls)} distinct URLs without query strings); "
          f"{len(missing)}/{n_gold} gold passages point to pages NOT indexed")
    for item_id, url in missing:
        print(f"  {item_id}: {url}")
    return sorted({u for _, u in missing})


# ---------- eval ----------
def retrieve(col, embed, question, k):
    """Top-k chunks in the order Chroma returns them."""
    r = col.query(query_embeddings=[embed([question])[0].tolist()], n_results=k,
                  include=["documents", "metadatas", "distances"])
    return [{"rank": i, "url": norm_url(m.get("url")), "chunk_id": m.get("chunk_id", id_),
             "score": round(1 - d, 4), "text": doc}
            for i, (id_, doc, m, d) in enumerate(zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]), 1)]


def make_reranker(model_name):
    from sentence_transformers import CrossEncoder  # only needed when a reranker is configured
    return CrossEncoder(model_name)


def rerank(reranker, question, hits):
    """Reorder hits by cross-encoder score. Keeps the original position as retrieval_rank."""
    scores = reranker.predict([(question, h["text"]) for h in hits], show_progress_bar=False)
    order = sorted(zip(hits, scores), key=lambda x: -float(x[1]))
    return [{**h, "rank": i, "retrieval_rank": h["rank"], "rerank_score": round(float(s), 4)}
            for i, (h, s) in enumerate(order, 1)]


def score(item, hits, indexed_urls, all_chunk_texts):
    """Rank of each gold quote in `hits`, the question rank (worst quote), and the miss group."""
    hit_texts = [norm_text(h["text"]) for h in hits]  # hits are in rank order, so position == rank
    quote_ranks = [first_rank(quote_fragments(g["quote"]), hit_texts) for g in item["gold_passages"]]
    rank = None if None in quote_ranks else max(quote_ranks)

    miss_group = None
    if rank is None or rank > 5:
        if any(norm_url(g["url"]) not in indexed_urls for g in item["gold_passages"]):
            miss_group = "0"
        elif any(first_rank(quote_fragments(g["quote"]), all_chunk_texts) is None for g in item["gold_passages"]):
            miss_group = "A"
        elif rank is None or rank > 50:  # ranks 51+ happen when top_k_retrieve > 50
            miss_group = "C"
        else:
            miss_group = "B"
    return {"quote_ranks": quote_ranks, "rank": rank, "miss_group": miss_group}


def metrics(rows):
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    if not n:
        return {"hit5": None, "hit50": None, "mrr": None}
    return {
        "hit5": round(sum(r["rank"] is not None and r["rank"] <= 5 for r in ok) / n, 4),
        "hit50": round(sum(r["rank"] is not None and r["rank"] <= 50 for r in ok) / n, 4),
        "mrr": round(sum(1 / r["rank"] for r in ok if r["rank"]) / n, 4),
    }


def read_runs():
    if not RUNS_FILE.exists():
        return []
    return [json.loads(l) for l in RUNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def config_diff(old, new):
    if old is None:
        return {}
    old = {**DEFAULTS, **old}  # runs recorded before a setting existed used its default
    return {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}


def cmd_eval(args):
    name, cfg = load_config(args.config)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(name, embedding_function=None)
    except Exception:
        sys.exit(f"Collection {name!r} not found. Run: python retrieval_eval.py index --config {args.config}")
    built = {**DEFAULTS, **json.loads((col.metadata or {}).get("config", "{}"))}
    mismatch = {k: [built.get(k), cfg[k]] for k in INDEX_SETTINGS if built.get(k) != cfg[k]}
    if mismatch:
        sys.exit(f"Collection {name!r} was built with different settings {{setting: [collection, config]}}: {mismatch}")

    everything = col.get(include=["documents", "metadatas"])
    # a deduplicated chunk also stands for the pages listed in also_urls
    page_urls = {u for m in everything["metadatas"] for u in [m.get("url"), *json.loads(m.get("also_urls") or "[]")]}
    indexed_urls = {norm_url(u) for u in page_urls}
    all_chunk_texts = [norm_text(d) for d in everything["documents"]]

    items = load_eval_items()
    missing_pages = coverage_check(items, indexed_urls, len(page_urls))

    embed = make_embedder(cfg["embedding_model"])
    reranker = make_reranker(cfg["reranker"]) if cfg["reranker"] else None
    k = cfg["top_k_retrieve"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            row = {"run_id": run_id, "item_id": item["id"], "category": item.get("category"),
                   "question": item["question"]}
            try:
                hits = retrieve(col, embed, cfg["query_prefix"] + item["question"], k)  # prefix: bge-style query instruction
                if reranker:
                    hits = rerank(reranker, item["question"], hits)
                row.update(score(item, hits, indexed_urls, all_chunk_texts))
                keys = ("rank", "retrieval_rank", "url", "chunk_id", "score", "rerank_score")
                row["top5"] = [{**{key: h[key] for key in keys if key in h}, "text": h["text"][:300]} for h in hits[:5]]
            except Exception as e:
                row.update(quote_ranks=None, rank=None, miss_group=None,
                           error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
                print(f"  ERROR {item['id']}: {row['error']}")
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    prev = read_runs()
    run = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "collection": name,
        "config": cfg,
        "what_changed": args.change,
        "config_diff": config_diff(prev[-1]["config"] if prev else None, cfg),
        "num_questions": len(rows),
        "num_errors": sum(bool(r.get("error")) for r in rows),
        "missing_pages": missing_pages, 
        **metrics(rows),
        **{f"miss_{g}": sum(r["miss_group"] == g for r in rows) for g in ("0", "A", "B", "C")},
        "per_category": {c: metrics(rs) for c, rs in sorted(by_cat.items())},
        "results_file": str(out_path.relative_to(ROOT)),
    }
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"\nRUN {run_id}  collection={name}  questions={run['num_questions']}  errors={run['num_errors']}")
    if missing_pages:
        print(f"  WARNING: {len(missing_pages)} gold page(s) not indexed -> those questions are miss group 0")
    print(f"  Hit@5={run['hit5']}  Hit@50={run['hit50']}  MRR={run['mrr']}  "
          f"misses 0/A/B/C = {run['miss_0']}/{run['miss_A']}/{run['miss_B']}/{run['miss_C']}")
    print_table([["category", "n", "Hit@5", "Hit@50", "MRR"]] +
                [[c, len(by_cat[c]), m["hit5"], m["hit50"], m["mrr"]] for c, m in run["per_category"].items()])
    print("\nALL RUNS")
    print_runs_table(read_runs())
    if len(prev) >= 1:  # what changed vs the previous run
        cmd_compare(argparse.Namespace(run_1=prev[-1]["run_id"], run_2=run_id))


# ---------- tables ----------
def print_table(table, wrap=None):
    """Box-drawn table; first row is the (centered) header, a separator between every row.
    wrap={column_index: width} wraps long text in that column onto extra lines."""
    wrap = wrap or {}
    table = [[textwrap.wrap("" if c is None else str(c), wrap[j]) or [""] if j in wrap
              else ["" if c is None else str(c)] for j, c in enumerate(row)] for row in table]
    widths = [max(len(line) for cell in col for line in cell) for col in zip(*table)]
    rule = lambda left, mid, right: left + mid.join("─" * (w + 2) for w in widths) + right

    def print_row(cells, align):
        for k in range(max(len(c) for c in cells)):
            print("│" + "│".join(f" {align(c[k] if k < len(c) else '', w)} " for c, w in zip(cells, widths)) + "│")

    print(rule("┌", "┬", "┐"))
    print_row(table[0], str.center)
    for cells in table[1:]:
        print(rule("├", "┼", "┤"))
        print_row(cells, str.ljust)
    print(rule("└", "┴", "┘"))


def print_runs_table(runs):
    best, best_idx = -1, None
    for i, r in enumerate(runs):
        if r["hit5"] is not None and r["hit5"] > best:
            best, best_idx = r["hit5"], i
    table = [["run", "what changed", "Hit@5", "Hit@50", "MRR", "0/A/B/C"]]
    for i, r in enumerate(runs):
        mark = ("!" if r["missing_pages"] else "") + (" *" if i == best_idx else "")
        table.append([r["run_id"] + mark, r["what_changed"] or "",
                      r["hit5"], r["hit50"], r["mrr"], f"{r['miss_0']}/{r['miss_A']}/{r['miss_B']}/{r['miss_C']}"])
    print_table(table, wrap={1: 30})
    print("* = best Hit@5 so far   ! = run had gold pages missing from the index")


def cmd_table(args):
    """Per-question ranks side by side for the given runs (default: all runs)."""
    runs = read_runs()
    if args.run_ids:
        by_id = {r["run_id"]: r for r in runs}
        runs = [by_id[i] for i in args.run_ids]
    results = [{x["item_id"]: x for x in map(json.loads, (ROOT / r["results_file"]).read_text(encoding="utf-8").splitlines())}
               for r in runs]

    def cell(x):
        if x is None:
            return "n/a"
        if x["rank"] is not None and x["rank"] <= 5:
            return f"{x['rank']} ✓"
        return f"{'-' if x['rank'] is None else x['rank']} ({x['miss_group']})"

    print_runs_table(runs)
    print()
    item_ids = list(dict.fromkeys(i for res in results for i in res))
    table = [["item", "category", *[r["run_id"] for r in runs]]]
    for iid in item_ids:
        category = next(res[iid]["category"] for res in results if iid in res)
        table.append([iid, category, *[cell(res.get(iid)) for res in results]])
    for label, key in (("Hit@5", "hit5"), ("Hit@50", "hit50"), ("MRR", "mrr")):
        table.append([label, "", *[r[key] for r in runs]])
    print_table(table)
    print("✓ = hit (rank <= 5)   - = not in top k   (0/A/B/C) = miss group   rank = worst of the question's quote ranks")


# ---------- compare ----------
def cmd_compare(args):
    all_runs = read_runs()
    runs = {r["run_id"]: r for r in all_runs}
    if not (args.run_1 and args.run_2):  # default: previous run vs latest run
        if len(all_runs) < 2:
            sys.exit("Need at least two runs to compare.")
        args.run_1, args.run_2 = all_runs[-2]["run_id"], all_runs[-1]["run_id"]

    def load(run_id):
        path = ROOT / runs[run_id]["results_file"] if run_id in runs else RESULTS_DIR / f"{run_id}.jsonl"
        return {r["item_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}

    print(f"\nCOMPARE {args.run_1} -> {args.run_2}")
    if args.run_1 in runs and args.run_2 in runs:
        r1, r2 = runs[args.run_1], runs[args.run_2]
        groups = lambda r: f"{r['miss_0']}/{r['miss_A']}/{r['miss_B']}/{r['miss_C']}"
        table = [["metric", args.run_1, args.run_2, "change"]]
        for label, key in (("Hit@5", "hit5"), ("Hit@50", "hit50"), ("MRR", "mrr")):
            delta = "" if r1[key] is None or r2[key] is None else f"{r2[key] - r1[key]:+.4f}"
            table.append([label, r1[key], r2[key], delta])
        table.append(["0/A/B/C", groups(r1), groups(r2), ""])
        table.append(["what changed", "", r2["what_changed"] or "", ""])
        print_table(table)

    a, b = load(args.run_1), load(args.run_2)
    key = lambda r: float("inf") if r["rank"] is None else r["rank"]
    better, worse = [], []
    for item_id in sorted(set(a) & set(b)):
        ra, rb = a[item_id], b[item_id]
        show = lambda rank: "-" if rank is None else rank
        line = [item_id, ra["category"], show(ra["rank"]), show(rb["rank"]), ra["miss_group"] or "hit", rb["miss_group"] or "hit"]
        if key(rb) < key(ra):
            better.append(line)
        elif key(rb) > key(ra):
            worse.append(line)
    header = ["item_id", "category", f"rank {args.run_1}", f"rank {args.run_2}", "group 1", "group 2"]
    for title, lines in (("IMPROVED", better), ("WORSE", worse)):
        print(f"\n{title} ({len(lines)})")
        print_table([header] + lines) if lines else print("  none")
    print(f"\nunchanged: {len(set(a) & set(b)) - len(better) - len(worse)}")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")  # Windows console
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--config", required=True)
    p.add_argument("--rebuild", action="store_true", help="overwrite the collection if it exists")
    p = sub.add_parser("eval")
    p.add_argument("--config", required=True)
    p.add_argument("--change", required=True, help="what you changed since the last run")
    p = sub.add_parser("compare")
    p.add_argument("run_1", nargs="?", help="default: previous run")
    p.add_argument("run_2", nargs="?", help="default: latest run")
    p = sub.add_parser("table", help="per-question ranks side by side")
    p.add_argument("run_ids", nargs="*", help="runs to show (default: all)")
    args = ap.parse_args()
    {"index": cmd_index, "eval": cmd_eval, "compare": cmd_compare, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
