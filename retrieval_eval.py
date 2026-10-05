"""Retrieval-only eval: no LLM calls. Optional local cross-encoder reranking (`reranker:` in the config).

Usage (from repo root):
    .venv/Scripts/python retrieval_eval.py index   --config configs/ircc_full_minilm_c180.yaml [--rebuild]
    .venv/Scripts/python retrieval_eval.py eval    --config configs/ircc_full_minilm_c180.yaml --change "what I changed"
    .venv/Scripts/python retrieval_eval.py compare [<run_id_1> <run_id_2>]   (default: previous vs latest run)
    .venv/Scripts/python retrieval_eval.py table   [run_id ...]
    .venv/Scripts/python retrieval_eval.py multihop --config configs/<name>.yaml --change "what I changed" [--validate-only]
    .venv/Scripts/python retrieval_eval.py multihop-compare <run_id_1> <run_id_2>
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
from add_to_vdb import chunk_page, split_sections  # noqa: E402

DEFAULTS = {"chunking_method": "section_words", "add_heading_path": True, "top_k_retrieve": 50, "reranker": None,
            "dedupe_chunks": False, "query_prefix": "",
            "split_queries": False, "fusion": "rrf", "subquery_filter": False,
            "expand_section": False, "expand_max_words": 450}
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
def load_eval_items(path=EVAL_FILE):
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [i for i in items if i.get("gold_passages")]  # unanswerable items have no gold: skip


def eval_set_fingerprint(items):
    """Short hash of what is scored: ids, questions and gold passages. Adding a question or editing a quote
    changes it, so runs on different eval set versions can be told apart."""
    scored = sorted(({"id": i["id"], "question": i["question"], "gold_passages": i["gold_passages"]} for i in items),
                    key=lambda x: x["id"])
    return hashlib.sha256(json.dumps(scored, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def eval_set_labels(runs):
    """{eval_set_hash: 'v1', 'v2', ...} in order of first appearance."""
    labels = {}
    for r in runs:
        h = r.get("eval_set_hash")
        if h and h not in labels:
            labels[h] = f"v{len(labels) + 1}"
    return labels


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


# Split points: sentence ends, a spaced dash, or a comma that starts a new question clause
# (", and am I ...", ", does that mean ...", ", can I ..."). Rule-based, no LLM, and blind to the category.
SPLIT_RE = re.compile(
    r"(?<=[?.!])\s+|\s+[—–-]\s+"
    r"|,\s+(?:and\s+|but\s+|so\s+)?(?=(?:can|could|does|do|did|is|am|are|will|would|should|what|how)\b)",
    re.IGNORECASE)


def split_question(question, min_words=4):
    """Sub-queries of a compound question; [] if it doesn't split into 2+ parts of min_words+ words."""
    parts = [p.strip(" ,;?") for p in SPLIT_RE.split(question)]
    parts = [p for p in parts if len(p.split()) >= min_words]
    return parts if len(parts) >= 2 else []


STOPWORDS = set("""a an the i i'm me my you your it its that this these those is am are was were be been do does did
can could will would should may might have has had to of for in on at by with from about as if or and but so not no
just still even also anything something really actually then than there here what how when where which who why""".split())


def is_vague(part, min_content_words=3):
    """A piece with too few content words to retrieve on its own, e.g. 'does that change anything for me'."""
    return sum(w not in STOPWORDS for w in re.findall(r"[a-z']+", part.lower())) < min_content_words


def make_queries(question, cfg):
    """Queries to run for one question; more than one means the question was split.
    Without subquery_filter: full question + every piece. With it: only the non-vague pieces
    (the full question would count the first piece twice); back to the full question if < 2 pieces survive."""
    pieces = split_question(question) if cfg["split_queries"] else []
    if not pieces:
        return [question]
    if not cfg["subquery_filter"]:
        return [question, *pieces]
    pieces = [p for p in pieces if not is_vague(p)]
    return pieces if len(pieces) >= 2 else [question]


def round_robin_fuse(ranked_lists, k):
    """Every list's #1, then every list's #2, ... (skipping chunks already taken). Top k, re-ranked 1..k."""
    seen, order = set(), []
    for i in range(max(len(hits) for hits in ranked_lists)):
        for hits in ranked_lists:
            if i < len(hits) and hits[i]["chunk_id"] not in seen:
                seen.add(hits[i]["chunk_id"])
                order.append(hits[i])
    drop = ("retrieval_rank", "rerank_score")  # refer to one input list only, misleading after fusion
    return [{**{key: v for key, v in h.items() if key not in drop}, "rank": i} for i, h in enumerate(order[:k], 1)]


def rrf_fuse(ranked_lists, k, c=60):
    """Reciprocal rank fusion: chunk score = sum of 1/(c + rank) over the lists it appears in. Top k, re-ranked 1..k."""
    fused, best = defaultdict(float), {}
    for hits in ranked_lists:
        for h in hits:
            fused[h["chunk_id"]] += 1 / (c + h["rank"])
            best.setdefault(h["chunk_id"], h)
    order = sorted(fused, key=lambda cid: -fused[cid])[:k]
    drop = ("retrieval_rank", "rerank_score")  # refer to one input list only, misleading after fusion
    return [{**{key: v for key, v in best[cid].items() if key not in drop}, "rank": i, "rrf_score": round(fused[cid], 5)}
            for i, cid in enumerate(order, 1)]


# ---------- section expansion ("small-to-big") ----------
def build_section_index(pages, cfg):
    """chunk_id -> the section it came from. Replays chunk_page's windowing (no dedup) so the section and the
    chunk's word offset are exact; dedup keeps original chunk_ids, so this also works for dedup collections."""
    size, step = cfg["chunk_size"], cfg["chunk_size"] - cfg["chunk_overlap"]
    idx = {}
    for page in pages:
        ids, n = [], 0
        for s, (heading, body) in enumerate(split_sections(page["text"])):
            words = body.split()
            for start in range(0, max(len(words), 1), step):
                if not words[start : start + size]:
                    continue
                chunk_id = f"{page['key']}_{page.get('content_hash', 'x')}_{n}"
                idx[chunk_id] = {"section": f"{page['url']}#{s}", "words": words, "start": start,
                                 "head": f"{page.get('title') or ''} | {heading}"}
                ids.append(chunk_id)
                n += 1
                if start + size >= len(words):
                    break
        expected = [c["id"] for c in chunk_page(page, max_words=size, overlap_words=cfg["chunk_overlap"])]
        assert ids == expected, f"section index out of sync with chunk_page for {page['url']}"
    return idx


def expand_hit(hit, idx, cfg):
    """Replace a chunk by its whole section, or by an expand_max_words window centred on the chunk if the
    section is longer. Chunks not in the index (shouldn't happen) are returned unchanged."""
    e = idx.get(hit["chunk_id"])
    if e is None:
        return {**hit, "unit": hit["chunk_id"]}
    words, cap = e["words"], cfg["expand_max_words"]
    if len(words) <= cap:
        lo, hi = 0, len(words)
    else:
        end = min(e["start"] + cfg["chunk_size"], len(words))
        lo = max(0, e["start"] - (cap - (end - e["start"])) // 2)
        hi = min(len(words), lo + cap)
        lo = max(0, hi - cap)
    body = " ".join(words[lo:hi])
    text = f"{e['head']}\n{body}".strip() if cfg["add_heading_path"] else body
    return {**hit, "text": text, "unit": f"{e['section']}[{lo}:{hi}]", "expanded_words": hi - lo}


def expand_and_collapse(hits, idx, cfg):
    """Expand every hit, keep the first (best-ranked) hit per unit, re-rank 1..n."""
    seen, out = set(), []
    for h in (expand_hit(h, idx, cfg) for h in hits):
        if h["unit"] not in seen:
            seen.add(h["unit"])
            out.append({**h, "chunk_rank": h["rank"], "rank": len(out) + 1})
    return out


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


def ctx_words5(rows):
    """Mean words in the top-5 units: Hit@5 with bigger units (section expansion) costs more context."""
    vals = [r["ctx_words5"] for r in rows if not r.get("error") and "ctx_words5" in r]
    return round(sum(vals) / len(vals)) if vals and len(vals) == len([r for r in rows if not r.get("error")]) else None


def read_runs():
    if not RUNS_FILE.exists():
        return []
    return [json.loads(l) for l in RUNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def config_diff(old, new):
    if old is None:
        return {}
    old = {**DEFAULTS, **old}  # runs recorded before a setting existed used its default
    return {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}


def open_eval_collection(name, cfg, config_path):
    """Open the collection, check it was built with the config's index settings, and collect what scoring needs.
    Returns (col, page_urls, indexed_urls, all_chunk_texts, section_idx). Shared by eval and multihop."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(name, embedding_function=None)
    except Exception:
        sys.exit(f"Collection {name!r} not found. Run: python retrieval_eval.py index --config {config_path}")
    built = {**DEFAULTS, **json.loads((col.metadata or {}).get("config", "{}"))}
    mismatch = {k: [built.get(k), cfg[k]] for k in INDEX_SETTINGS if built.get(k) != cfg[k]}
    if mismatch:
        sys.exit(f"Collection {name!r} was built with different settings {{setting: [collection, config]}}: {mismatch}")

    everything = col.get(include=["documents", "metadatas"])
    # a deduplicated chunk also stands for the pages listed in also_urls
    page_urls = {u for m in everything["metadatas"] for u in [m.get("url"), *json.loads(m.get("also_urls") or "[]")]}
    indexed_urls = {norm_url(u) for u in page_urls}
    section_idx = build_section_index(load_pages(RUN_DIR), cfg) if cfg["expand_section"] else None
    if section_idx:
        # group A = "quote not in any retrievable unit": with expansion the units are expanded sections
        all_chunk_texts = [norm_text(expand_hit({"chunk_id": m["chunk_id"], "text": d}, section_idx, cfg)["text"])
                           for d, m in zip(everything["documents"], everything["metadatas"])]
    else:
        all_chunk_texts = [norm_text(d) for d in everything["documents"]]
    return col, page_urls, indexed_urls, all_chunk_texts, section_idx


def retrieve_for_question(question, cfg, col, embed, reranker, section_idx):
    """The full retrieval pipeline for one question: split → retrieve → rerank → fuse → expand.
    Returns (ranked hits, queries run). Shared by eval and multihop."""
    k = cfg["top_k_retrieve"]
    # One query unless split_queries splits the question; then each query's list is fused.
    queries = make_queries(question, cfg)
    ranked_lists = []
    for q in queries:
        hits = retrieve(col, embed, cfg["query_prefix"] + q, k)  # prefix: bge-style query instruction
        if reranker:
            hits = rerank(reranker, q, hits)
        ranked_lists.append(hits)
    if len(queries) == 1:
        hits = ranked_lists[0]
    else:
        fuse = {"rrf": rrf_fuse, "round_robin": round_robin_fuse}[cfg["fusion"]]
        hits = fuse(ranked_lists, k)
    if section_idx:
        hits = expand_and_collapse(hits, section_idx, cfg)
    return hits, queries


def cmd_eval(args):
    name, cfg = load_config(args.config)
    col, page_urls, indexed_urls, all_chunk_texts, section_idx = open_eval_collection(name, cfg, args.config)

    eval_file = Path(args.eval_file).resolve()
    items = load_eval_items(eval_file)
    missing_pages = coverage_check(items, indexed_urls, len(page_urls))

    embed = make_embedder(cfg["embedding_model"])
    reranker = make_reranker(cfg["reranker"]) if cfg["reranker"] else None
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            row = {"run_id": run_id, "item_id": item["id"], "category": item.get("category"),
                   "question": item["question"]}
            try:
                hits, queries = retrieve_for_question(item["question"], cfg, col, embed, reranker, section_idx)
                if len(queries) > 1:
                    row["sub_queries"] = queries
                row.update(score(item, hits, indexed_urls, all_chunk_texts))
                row["ctx_words5"] = sum(len(h["text"].split()) for h in hits[:5])  # text the LLM would get in 5 slots
                keys = ("rank", "chunk_rank", "retrieval_rank", "url", "chunk_id", "unit", "expanded_words",
                        "score", "rerank_score", "rrf_score")
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
        "eval_file": str(eval_file.relative_to(ROOT)) if eval_file.is_relative_to(ROOT) else str(eval_file),
        "eval_set_hash": eval_set_fingerprint(items),
        "eval_set_n": len(items),
        "num_questions": len(rows),
        "num_errors": sum(bool(r.get("error")) for r in rows),
        "missing_pages": missing_pages, 
        **metrics(rows),
        **{f"miss_{g}": sum(r["miss_group"] == g for r in rows) for g in ("0", "A", "B", "C")},
        "per_category": {c: metrics(rs) for c, rs in sorted(by_cat.items())},
        "ctx_words5": ctx_words5(rows),
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
    labels = eval_set_labels(read_runs())  # labels from the full history, so they don't shift when filtering
    label = lambda r: labels.get(r.get("eval_set_hash"), "v?")
    best = {}  # best Hit@5 per eval set version: scores on different question sets aren't comparable
    for i, r in enumerate(runs):
        v = label(r)
        if r["hit5"] is not None and (v not in best or r["hit5"] > runs[best[v]]["hit5"]):
            best[v] = i
    table = [["run", "what changed", "Hit@5", "Hit@50", "MRR", "0/A/B/C"]]
    for i, r in enumerate(runs):
        if i and label(r) != label(runs[i - 1]):
            table.append([f"── eval set {label(r)} ──", f"{r.get('eval_set_n', r['num_questions'])} questions", "", "", "", ""])
        mark = ("!" if r["missing_pages"] else "") + (" *" if best.get(label(r)) == i else "")
        table.append([f"{r['run_id']} [{label(r)}]{mark}", r["what_changed"] or "",
                      r["hit5"], r["hit50"], r["mrr"], f"{r['miss_0']}/{r['miss_A']}/{r['miss_B']}/{r['miss_C']}"])
    print_table(table, wrap={1: 30})
    print("[vN] = eval set version   * = best Hit@5 within that version   ! = run had gold pages missing from the index")


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

    a, b = load(args.run_1), load(args.run_2)
    shared = sorted(set(a) & set(b))
    labels = eval_set_labels(all_runs)
    v1, v2 = (labels.get(runs.get(r, {}).get("eval_set_hash"), "v?") for r in (args.run_1, args.run_2))
    print(f"\nCOMPARE {args.run_1} [{v1}] -> {args.run_2} [{v2}]")
    if v1 != v2 or set(a) != set(b):
        print(f"  Different eval set versions: metrics below are recomputed on the {len(shared)} shared questions "
              f"({len(set(a) - set(b))} only in run 1, {len(set(b) - set(a))} only in run 2).")
    # Metrics from the per-question rows of the shared questions, so both sides are on the same questions.
    m1, m2 = metrics([a[i] for i in shared]), metrics([b[i] for i in shared])
    groups = lambda res: "/".join(str(sum(res[i]["miss_group"] == g for i in shared)) for g in ("0", "A", "B", "C"))
    table = [["metric", args.run_1, args.run_2, "change"]]
    for label, key in (("Hit@5", "hit5"), ("Hit@50", "hit50"), ("MRR", "mrr")):
        delta = "" if m1[key] is None or m2[key] is None else f"{m2[key] - m1[key]:+.4f}"
        table.append([label, m1[key], m2[key], delta])
    table.append(["0/A/B/C", groups(a), groups(b), ""])
    c1, c2 = ctx_words5([a[i] for i in shared]), ctx_words5([b[i] for i in shared])
    table.append(["ctx words (top 5)", c1 or "n/a", c2 or "n/a", f"{c2 - c1:+d}" if c1 and c2 else ""])
    if args.run_2 in runs:
        table.append(["what changed", "", runs[args.run_2]["what_changed"] or "", ""])
    print_table(table)

    key = lambda r: float("inf") if r["rank"] is None else r["rank"]
    better, worse = [], []
    for item_id in shared:
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
    print(f"\nunchanged: {len(shared) - len(better) - len(worse)}")


# ---------- multi-hop eval ----------
# Separate from the general eval: own items (eval_set multi-hop + validated drafts), passage-level metrics and
# own output folder. Reuses the general pipeline (open_eval_collection, retrieve_for_question) and matching.
MULTIHOP_DIR = ROOT / "multihop_eval"
MULTIHOP_RUNS = MULTIHOP_DIR / "runs.jsonl"
MULTIHOP_RESULTS = MULTIHOP_DIR / "results"
DRAFT_MULTIHOP_FILE = ROOT / "data/eval_set/draft_multihop_v0.jsonl"


def validate_drafts(draft_lines, eval_items, indexed_urls, pages):
    """One report dict per draft line: status pass / fail / duplicate, reasons, flags. Never edits the drafts."""
    by_hash = {p["content_hash"]: p["text"] for p in pages if p.get("content_hash")}
    by_url = {p["url"].strip(): p["text"] for p in pages}
    eval_by_id = {i["id"]: i for i in eval_items}
    report = []
    for n, line in enumerate(draft_lines, 1):
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            report.append({"id": f"<line {n}>", "status": "fail", "reasons": [f"invalid JSON: {e}"], "flags": [], "n_passages": 0})
            continue
        golds = d.get("gold_passages")
        entry = {"id": d.get("id"), "status": "pass", "reasons": [], "flags": [],
                 "n_passages": len(golds) if isinstance(golds, list) else 0}
        if d.get("id") in eval_by_id:
            entry["status"] = "duplicate"
            entry["reasons"].append("same id as an eval_set item; the eval_set version is used")
            if d.get("gold_passages") != eval_by_id[d["id"]]["gold_passages"]:
                entry["reasons"].append("gold passages differ from the eval_set version")
            report.append(entry)
            continue
        if not isinstance(d.get("id"), str) or not d["id"].strip():
            entry["reasons"].append("missing id")
        if not isinstance(d.get("question"), str) or not d["question"].strip():
            entry["reasons"].append("missing question")
        if not isinstance(golds, list) or not golds:
            entry["reasons"].append("missing gold_passages")
            golds = []
        for j, g in enumerate(golds, 1):
            if not isinstance(g, dict) or not (g.get("url") or "").strip() or not (g.get("quote") or "").strip():
                entry["reasons"].append(f"passage {j}: missing url or quote")
                continue
            text = by_hash.get(g.get("content_hash")) or by_url.get(g["url"].strip())
            if text is None:
                entry["reasons"].append(f"passage {j}: no page snapshot for content_hash/url {g['url']}")
            else:
                page = norm_text(text)
                missing = [f.strip() for f in quote_fragments(g["quote"]) if f not in page]
                if missing:
                    entry["reasons"].append(f"passage {j}: quote not in page snapshot (first missing piece: '{missing[0][:80]}')")
            if norm_url(g["url"]) not in indexed_urls:
                entry["reasons"].append(f"passage {j}: URL not in the collection: {g['url']}")
        if len(golds) < 2:
            entry["flags"].append(f"only {len(golds)} gold passage(s)")
        entry["status"] = "fail" if entry["reasons"] else "pass"
        report.append(entry)
    return report


def passage_miss_group(gold, rank, indexed_urls, all_chunk_texts):
    """Miss group of one gold passage, same rules as score(): None if in the top 5, else 0 / A / B / C."""
    if rank is not None and rank <= 5:
        return None
    if norm_url(gold["url"]) not in indexed_urls:
        return "0"
    if first_rank(quote_fragments(gold["quote"]), all_chunk_texts) is None:
        return "A"
    if rank is None or rank > 50:
        return "C"
    return "B"


def multihop_metrics(rows):
    ok = [r for r in rows if not r.get("error")]
    ranks = [rk for r in ok for rk in r["passage_ranks"]]
    groups = [g for r in ok for g in r["miss_groups"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "num_items": len(ok),
        "joint_hit5": mean([r["joint_hit@5"] for r in ok]),
        "joint_hit10": mean([r["joint_hit@10"] for r in ok]),
        "passage_recall5": mean([r["passage_recall@5"] for r in ok]),
        "passage_mrr": mean([1 / rk if rk else 0 for rk in ranks]),
        **{f"miss_{g}": groups.count(g) for g in ("0", "A", "B", "C")},
    }


def read_multihop_runs():
    if not MULTIHOP_RUNS.exists():
        return []
    return [json.loads(l) for l in MULTIHOP_RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def print_multihop_runs_table(runs):
    labels = eval_set_labels(runs)
    best = max(range(len(runs)), key=lambda i: (runs[i]["joint_hit5"] or -1, -i), default=None)
    table = [["run_id", "what_changed", "config_diff", "joint@5", "joint@10", "recall@5", "0/A/B/C"]]
    for i, r in enumerate(runs):
        diff = ", ".join(f"{k}: {o}->{n}" for k, (o, n) in r["config_diff"].items()) or "-"
        table.append([f"{r['run_id']} [{labels.get(r.get('eval_set_hash'), 'v?')}]" + (" *" if i == best else ""),
                      r["what_changed"] or "", diff, r["joint_hit5"], r["joint_hit10"], r["passage_recall5"],
                      f"{r['miss_0']}/{r['miss_A']}/{r['miss_B']}/{r['miss_C']}"])
    print_table(table, wrap={1: 30, 2: 30})
    print("* = best joint@5   [vN] = multi-hop item set version   0/A/B/C = missing passages (not in top 5) by group")


def cmd_multihop(args):
    name, cfg = load_config(args.config)
    col, page_urls, indexed_urls, all_chunk_texts, section_idx = open_eval_collection(name, cfg, args.config)

    eval_items = [i for i in load_eval_items(EVAL_FILE) if i.get("category") == "multi-hop"]
    draft_lines = [l for l in DRAFT_MULTIHOP_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    report = validate_drafts(draft_lines, eval_items, indexed_urls, load_pages(RUN_DIR))
    MULTIHOP_DIR.mkdir(exist_ok=True)
    with open(MULTIHOP_DIR / "draft_validation.jsonl", "w", encoding="utf-8") as f:
        for entry in report:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts = {s: sum(e["status"] == s for e in report) for s in ("pass", "fail", "duplicate")}
    print(f"\nDRAFT VALIDATION ({DRAFT_MULTIHOP_FILE.name}, {len(report)} drafts): "
          f"{counts['pass']} pass, {counts['fail']} fail, {counts['duplicate']} duplicate of eval_set")
    problems = [[e["id"], e["status"], "; ".join(e["reasons"])] for e in report if e["status"] != "pass"]
    if problems:
        print_table([["draft", "status", "reasons"]] + problems, wrap={2: 90})
    flagged = [(i["id"], f"only {len(i['gold_passages'])} gold passage(s)") for i in eval_items if len(i["gold_passages"]) < 2]
    flagged += [(e["id"], "; ".join(e["flags"])) for e in report if e["flags"] and e["status"] == "pass"]
    for item_id, flag in flagged:
        print(f"  FLAG {item_id}: {flag}")
    print("  report written to", (MULTIHOP_DIR / "draft_validation.jsonl").relative_to(ROOT))
    if args.validate_only:
        return

    passed = {e["id"] for e in report if e["status"] == "pass"}
    drafts = [d for d in map(json.loads, draft_lines) if d.get("id") in passed]
    items = [(i, "eval_set") for i in eval_items] + [(d, "draft_v0") for d in drafts]

    embed = make_embedder(cfg["embedding_model"])
    reranker = make_reranker(cfg["reranker"]) if cfg["reranker"] else None
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    MULTIHOP_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = MULTIHOP_RESULTS / f"{run_id}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for item, source in items:
            row = {"run_id": run_id, "item_id": item["id"], "source": source, "question": item["question"]}
            try:
                hits, queries = retrieve_for_question(item["question"], cfg, col, embed, reranker, section_idx)
                if len(queries) > 1:
                    row["sub_queries"] = queries
                hit_texts = [norm_text(h["text"]) for h in hits]  # hits are in rank order, so position == rank
                ranks = [first_rank(quote_fragments(g["quote"]), hit_texts) for g in item["gold_passages"]]
                row.update({
                    "passage_ranks": ranks,
                    "joint_hit@5": all(rk is not None and rk <= 5 for rk in ranks),
                    "joint_hit@10": all(rk is not None and rk <= 10 for rk in ranks),
                    "passage_recall@5": round(sum(rk is not None and rk <= 5 for rk in ranks) / len(ranks), 4),
                    "miss_groups": [passage_miss_group(g, rk, indexed_urls, all_chunk_texts)
                                    for g, rk in zip(item["gold_passages"], ranks)],
                    "top10": [{**{key: h[key] for key in ("rank", "url", "chunk_id", "score") if key in h},
                               "text": h["text"][:300]} for h in hits[:10]],
                })
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
                print(f"  ERROR {item['id']}: {row['error']}")
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    prev = read_multihop_runs()
    run = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "collection": name,
        "config": cfg,
        "what_changed": args.change,
        "config_diff": config_diff(prev[-1]["config"] if prev else None, cfg),
        "eval_set_hash": eval_set_fingerprint([i for i, _ in items]),  # version of the multi-hop item set
        "num_items": {s: len(rs) for s, rs in by_source.items()},
        "num_drafts_rejected": counts["fail"],
        "num_errors": sum(bool(r.get("error")) for r in rows),
        **multihop_metrics(rows),
        "by_source": {s: multihop_metrics(rs) for s, rs in sorted(by_source.items())},
        "results_file": str(out_path.relative_to(ROOT)),
    }
    with open(MULTIHOP_RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"\nMULTI-HOP RUN {run_id}  collection={name}  items={run['num_items']}  errors={run['num_errors']}")
    table = [["source", "items", "joint@5", "joint@10", "recall@5", "passage MRR", "missing 0/A/B/C"]]
    for label, m in [*run["by_source"].items(), ("ALL", multihop_metrics(rows))]:
        table.append([label, m["num_items"], m["joint_hit5"], m["joint_hit10"], m["passage_recall5"], m["passage_mrr"],
                      f"{m['miss_0']}/{m['miss_A']}/{m['miss_B']}/{m['miss_C']}"])
    print_table(table)
    print("\nALL MULTI-HOP RUNS")
    print_multihop_runs_table(read_multihop_runs())


def cmd_multihop_compare(args):
    runs = {r["run_id"]: r for r in read_multihop_runs()}
    for rid in (args.run_1, args.run_2):
        if rid not in runs:
            sys.exit(f"Unknown multi-hop run {rid!r}")
    load = lambda rid: {r["item_id"]: r for r in map(json.loads, (ROOT / runs[rid]["results_file"]).read_text(encoding="utf-8").splitlines())}
    a, b = load(args.run_1), load(args.run_2)
    shared = sorted(set(a) & set(b))
    print(f"\nMULTI-HOP COMPARE {args.run_1} -> {args.run_2}")
    if set(a) != set(b):
        print(f"  Different item sets: metrics below are on the {len(shared)} shared items "
              f"({len(set(a) - set(b))} only in run 1, {len(set(b) - set(a))} only in run 2).")
    m1, m2 = multihop_metrics([a[i] for i in shared]), multihop_metrics([b[i] for i in shared])
    table = [["metric", args.run_1, args.run_2, "change"]]
    for label, key in (("joint@5", "joint_hit5"), ("joint@10", "joint_hit10"), ("recall@5", "passage_recall5"),
                       ("passage MRR", "passage_mrr")):
        delta = "" if m1[key] is None or m2[key] is None else f"{m2[key] - m1[key]:+.4f}"
        table.append([label, m1[key], m2[key], delta])
    groups = lambda m: f"{m['miss_0']}/{m['miss_A']}/{m['miss_B']}/{m['miss_C']}"
    table.append(["missing 0/A/B/C", groups(m1), groups(m2), ""])
    table.append(["what changed", "", runs[args.run_2]["what_changed"] or "", ""])
    print_table(table)

    # an item improves if its worst passage rank improves (None = not retrieved = worst)
    worst = lambda r: float("inf") if r.get("error") or None in r["passage_ranks"] else max(r["passage_ranks"])
    show = lambda r: "error" if r.get("error") else "[" + ", ".join("-" if x is None else str(x) for x in r["passage_ranks"]) + "]"
    hit = lambda r: "joint@5" if r.get("joint_hit@5") else ("joint@10" if r.get("joint_hit@10") else "miss")
    better, worse = [], []
    for i in shared:
        line = [i, a[i]["source"], show(a[i]), show(b[i]), hit(a[i]), hit(b[i])]
        if worst(b[i]) < worst(a[i]):
            better.append(line)
        elif worst(b[i]) > worst(a[i]):
            worse.append(line)
    header = ["item", "source", f"passage ranks {args.run_1}", f"passage ranks {args.run_2}", "before", "after"]
    for title, lines in (("IMPROVED", better), ("WORSE", worse)):
        print(f"\n{title} ({len(lines)})")
        print_table([header] + lines) if lines else print("  none")
    print(f"\nunchanged: {len(shared) - len(better) - len(worse)}")


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
    p.add_argument("--eval-file", default=str(EVAL_FILE), help="eval set to score (default: %(default)s)")
    p = sub.add_parser("compare")
    p.add_argument("run_1", nargs="?", help="default: previous run")
    p.add_argument("run_2", nargs="?", help="default: latest run")
    p = sub.add_parser("table", help="per-question ranks side by side")
    p.add_argument("run_ids", nargs="*", help="runs to show (default: all)")
    p = sub.add_parser("multihop", help="multi-hop eval (eval_set multi-hop items + validated drafts)")
    p.add_argument("--config", required=True)
    p.add_argument("--change", required=True, help="what you changed since the last multi-hop run")
    p.add_argument("--validate-only", action="store_true", help="only validate the draft items and write the report")
    p = sub.add_parser("multihop-compare", help="items that improved / got worse between two multi-hop runs")
    p.add_argument("run_1")
    p.add_argument("run_2")
    args = ap.parse_args()
    {"index": cmd_index, "eval": cmd_eval, "compare": cmd_compare, "table": cmd_table,
     "multihop": cmd_multihop, "multihop-compare": cmd_multihop_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
