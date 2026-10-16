"""Retrieval-only eval: no LLM calls. Optional local cross-encoder reranking (`reranker:` in the config).

Usage (from repo root):
    .venv/Scripts/python retrieval_eval.py index   --config configs/ircc_full_minilm_c180.yaml [--rebuild]
    .venv/Scripts/python retrieval_eval.py eval    --config configs/ircc_full_minilm_c180.yaml --change "what I changed"
    .venv/Scripts/python retrieval_eval.py compare [<run_id_1> <run_id_2>]   (default: previous vs latest run)
    .venv/Scripts/python retrieval_eval.py table   [run_id ...]
    .venv/Scripts/python retrieval_eval.py multihop --config configs/<name>.yaml --change "what I changed" [--validate-only]
    .venv/Scripts/python retrieval_eval.py multihop-compare <run_id_1> <run_id_2>
    .venv/Scripts/python retrieval_eval.py multihop-rescore [run_id ...]   (add new metrics to past multi-hop runs)
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
from collections import Counter, defaultdict
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
LLM_CACHE = ROOT / "cache/llm_cache.jsonl"
RESULTS_DIR = ROOT / "retrieval_results"
RUNS_FILE = ROOT / "retrieval_runs.jsonl"

# Reuse the chunking from the main RAG code so the chunks are identical.
sys.path.insert(0, str(ROOT / "scripts"))
from add_to_vdb import chunk_page, split_sections  # noqa: E402

DEFAULTS = {"chunking_method": "section_words", "add_heading_path": True, "top_k_retrieve": 50, "reranker": None,
            "dedupe_chunks": False, "query_prefix": "",
            "split_queries": False, "fusion": "rrf", "subquery_filter": False,
            "expand_section": False, "expand_max_words": 450,
            "search": "hnsw",
            "split_method": "rules", "chain_hops": 1, "chain_context_k": 3, "llm_model": "gpt-5-mini",
            "decompose_prompt": "v1",
            "rt_rag": False, "rt_trees": 5, "rt_max_leaves": 4, "rt_answer_k": 3, "rt_rewrites": 3,
            "rt_answer_samples": 1,
            "route": "none", "chain_categories": ["multi-hop"],
            "bm25_k": 0}
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
class ExactSearch:
    """Exact cosine search over a collection's stored embeddings (`search: exact`). Chroma's HNSW search
    returns only ~94-97 of the true top 100, and a different set per process. Same query()/result shape as
    Chroma's, so retrieve() works with either. Ties are broken by chunk id, so the order is always the same."""

    def __init__(self, ids, documents, metadatas, embeddings):
        order = np.argsort(np.asarray(ids))  # rows in chunk-id order -> stable sort below breaks ties by id
        self.ids = [ids[i] for i in order]
        self.documents = [documents[i] for i in order]
        self.metadatas = [metadatas[i] for i in order]
        m = np.asarray(embeddings, dtype=np.float32)[order]
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def query(self, query_embeddings, n_results, include=None):
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        sims = self.matrix @ (q / np.linalg.norm(q))
        top = np.argsort(-sims, kind="stable")[:n_results]
        return {"ids": [[self.ids[i] for i in top]], "documents": [[self.documents[i] for i in top]],
                "metadatas": [[self.metadatas[i] for i in top]], "distances": [[float(1 - sims[i]) for i in top]]}


def retrieve(col, embed, question, k):
    """Top-k chunks in the order the search returns them (Chroma HNSW, or ExactSearch)."""
    r = col.query(query_embeddings=[embed([question])[0].tolist()], n_results=k,
                  include=["documents", "metadatas", "distances"])
    return [{"rank": i, "url": norm_url(m.get("url")), "chunk_id": m.get("chunk_id", id_),
             "score": round(1 - d, 4), "text": doc}
            for i, (id_, doc, m, d) in enumerate(zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]), 1)]


# ---------- BM25 (bm25_k > 0): sparse candidates added to the dense pool before the rerank ----------
STOPWORDS = frozenset("a an and are as at be but by can could do does for from had has have how i if in into is it "
                      "its me my no not of on or our so than that the their them then there these they this to was "
                      "we were what when where which who why will with would you your".split())


def bm25_tokens(text):
    return [t for t in norm_text(text).split() if t not in STOPWORDS]


class BM25:
    """Okapi BM25 (k1 1.5, b 0.75, non-negative idf) over the same chunk texts as the dense index, heading path
    included. Built from an ExactSearch, so chunks are in chunk-id order and ties break by chunk id."""

    def __init__(self, ids, documents, metadatas, k1=1.5, b=0.75):
        self.ids, self.documents, self.metadatas, self.k1 = ids, documents, metadatas, k1
        lengths = np.zeros(len(documents), dtype=np.float32)
        postings = defaultdict(lambda: ([], []))  # term -> (chunk indexes, term frequencies)
        for i, doc in enumerate(documents):
            toks = bm25_tokens(doc)
            lengths[i] = len(toks)
            for t, tf in Counter(toks).items():
                postings[t][0].append(i)
                postings[t][1].append(tf)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())  # per-chunk length normalisation
        n = len(documents)
        self.postings = {t: (np.asarray(d), np.asarray(f, dtype=np.float32), np.log((n - len(d) + 0.5) / (len(d) + 0.5) + 1))
                         for t, (d, f) in postings.items()}

    def search(self, query, k):
        scores = np.zeros(len(self.documents), dtype=np.float32)
        for t in set(bm25_tokens(query)):
            if t in self.postings:
                docs, tf, idf = self.postings[t]
                scores[docs] += idf * tf * (self.k1 + 1) / (tf + self.norm[docs])
        top = [i for i in np.argsort(-scores, kind="stable")[:k] if scores[i] > 0]
        return [{"rank": r, "url": norm_url(self.metadatas[i].get("url")), "chunk_id": self.metadatas[i].get("chunk_id", self.ids[i]),
                 "bm25_score": round(float(scores[i]), 4), "bm25_rank": r, "text": self.documents[i]}
                for r, i in enumerate(top, 1)]


def candidates(col, embed, q, cfg):
    """Candidate pool for one query: the dense top top_k_retrieve; with bm25_k > 0 also the BM25 top bm25_k chunks
    the dense list missed, appended in BM25 order (the reranker sets the final order). Dense hits BM25 also
    found carry their bm25_rank."""
    hits = retrieve(col, embed, cfg["query_prefix"] + q, cfg["top_k_retrieve"])  # prefix: bge-style query instruction
    if not cfg["bm25_k"]:
        return hits
    sparse = col.bm25.search(q, cfg["bm25_k"])  # raw query: the prefix is for the embedding model only
    bm25_rank = {h["chunk_id"]: h["rank"] for h in sparse}
    pool = [{**h, **({"bm25_rank": bm25_rank[h["chunk_id"]]} if h["chunk_id"] in bm25_rank else {})} for h in hits]
    dense_ids = {h["chunk_id"] for h in hits}
    pool += [h for h in sparse if h["chunk_id"] not in dense_ids]
    return [{**h, "rank": i} for i, h in enumerate(pool, 1)]


def make_reranker(model_name):
    from sentence_transformers import CrossEncoder  # only needed when a reranker is configured
    return CrossEncoder(model_name)


def rerank(reranker, question, hits):
    """Reorder hits by cross-encoder score. Keeps the original position as retrieval_rank."""
    scores = reranker.predict([(question, h["text"]) for h in hits], show_progress_bar=False)
    order = sorted(zip(hits, scores), key=lambda x: -float(x[1]))
    return [{**h, "rank": i, "retrieval_rank": h["rank"], "rerank_score": round(float(s), 4)}
            for i, (h, s) in enumerate(order, 1)]


# ---------- LLM query generation (split_method: llm, chain_hops: 2), cached ----------
# Every response is cached in LLM_CACHE keyed by model + prompt name + version + input, so re-runs make no calls
# and give the same queries. Bump a prompt's version when its text changes.
LLM_STATS = {"calls": 0, "cache_hits": 0}


class LLMError(RuntimeError):
    """An LLM call failed; aborts the run (not counted as a per-item error)."""
_llm = {"client": None, "cache": None}

DECOMPOSE_V1 = ("decompose", "v1", """You turn a user's question about Canadian immigration (IRCC) into search
queries for a retrieval system over IRCC web pages. If the question asks about several distinct things, write one
query per thing (at most 4); otherwise write one query. Every query must be self-contained: repeat the context it
needs from the rest of the question (who the person is, program type and length, permit status, school type, dates),
and replace pronouns and vague references ("that", "both of us", "the new place") with what they refer to. Use the
official terms where you know them (e.g. "post-graduation work permit (PGWP)", "provincial attestation letter (PAL)",
"designated learning institution (DLI)"). Do not answer the question.
Return JSON: {"queries": ["...", "..."]}""")

# v2: v1 produced 50-80-word research briefs ("Include IRCC guidance on ...") that diluted the embedding and confused
# the ms-marco reranker, and sometimes merged two topics into one query. v2 asks for short search-engine queries.
DECOMPOSE_V2 = ("decompose", "v2", """You turn a user's question about Canadian immigration (IRCC) into short search
queries for a search engine over IRCC web pages.
Rules:
- Write one query per distinct thing the user asks (at most 4). Never merge two asks into one query.
- Each query is at most 20 words, phrased like a search-engine query or a short question.
- Make each query understandable on its own: replace pronouns and vague references ("that", "both of us", "the new
  place") with what they mean, and keep only the details from the question that change the answer (e.g. "5-month
  program", "no study permit", "PhD", "public university", "permit expired").
- Use official terms where you know them (post-graduation work permit, provincial attestation letter, designated
  learning institution), but do not add topics the user did not ask about.
- No lists of things to "include", no search operators (site:, quotes), no answers.
Return JSON: {"queries": ["...", "..."]}""")
DECOMPOSE_PROMPTS = {"v1": DECOMPOSE_V1, "v2": DECOMPOSE_V2}

CHAIN_PROMPT = ("chain", "v1", """You help a retrieval system answer a user's question about Canadian immigration
(IRCC). You get the question and the passages found so far. Decide what information the question needs that these
passages do NOT already contain, and write 0 to 2 self-contained search queries for that missing information (carry
the needed context from the question; use official IRCC terms where you know them). If the passages already cover
everything, return no queries. Do not answer the question.
Return JSON: {"missing": "<one sentence>", "queries": ["..."]}""")


def llm_json(cfg, prompt, user, sample=0):
    """JSON response for (system prompt, user text), from LLM_CACHE if present, else one API call (then cached).
    sample > 0 gives an independent cached draw for the same input (RT-RAG's multiple candidate trees/answers)."""
    name, version, system = prompt
    parts = [cfg["llm_model"], name, version, user] + ([sample] if sample else [])  # sample 0 keeps the old keys
    key = hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()
    if _llm["cache"] is None:
        _llm["cache"] = {}
        if LLM_CACHE.exists():
            for line in LLM_CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    _llm["cache"][e["key"]] = e["response"]
    if key in _llm["cache"]:
        LLM_STATS["cache_hits"] += 1
        return _llm["cache"][key]
    if _llm["client"] is None:
        from openai import OpenAI  # only needed when an LLM setting is used
        _llm["client"] = OpenAI(max_retries=3)  # OPENAI_API_KEY from env (.env loaded by add_to_vdb)
    try:
        r = _llm["client"].chat.completions.create(
            model=cfg["llm_model"], response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        response = json.loads(r.choices[0].message.content)
    except Exception as e:  # quota, auth, network, bad JSON: stop the run rather than score 22 errors
        raise LLMError(f"{type(e).__name__}: {e}") from e
    LLM_STATS["calls"] += 1
    _llm["cache"][key] = response
    LLM_CACHE.parent.mkdir(exist_ok=True)
    with open(LLM_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "model": cfg["llm_model"], "prompt": f"{name}/{version}", "user": user,
                            "response": response}, ensure_ascii=False) + "\n")
    return response


def clean_queries(response, limit):
    qs = response.get("queries") if isinstance(response, dict) else None
    return [q.strip() for q in (qs or []) if isinstance(q, str) and q.strip()][:limit]


def llm_decompose(question, cfg):
    """Self-contained sub-queries for the question (1–4); the question itself if the LLM returns none."""
    prompt = DECOMPOSE_PROMPTS[cfg["decompose_prompt"]]
    return clean_queries(llm_json(cfg, prompt, question), 4) or [question]


ROUTE_PROMPT = ("route", "v1", """Classify a user's question about Canadian immigration (IRCC) by what answering it
requires. Categories:
- "single-fact": one fact from one passage.
- "multi-hop": the answer needs facts from two or more different topics/pages (e.g. the user asks two linked things,
  or their situation touches two separate rules); no single passage answers it.
- "conditional": the answer depends on a condition (if X then Y, otherwise Z).
- "temporal": about a rule that changed or may have changed over time ("is that still true", "since", "new rule").
- "ambiguous": the question leaves out a detail that changes the answer (several possible interpretations).
- "unanswerable": IRCC web pages would not contain the answer (personal case status, locations, opinions, other
  countries' rules).
Return JSON: {"category": "<one of the above>"}""")
ROUTE_CATEGORIES = ("single-fact", "multi-hop", "conditional", "temporal", "ambiguous", "unanswerable")


def llm_route(question, cfg):
    """Predicted category of the question (from the question text only), or 'unknown'."""
    c = llm_json(cfg, ROUTE_PROMPT, question).get("category")
    return c if c in ROUTE_CATEGORIES else "unknown"


def route_summary(rows, gold_category=None):
    """Router accuracy vs the true category (row["category"], or gold_category for multi-hop items) and
    predicted-category counts per true category. None if the run did not route."""
    routed = [r for r in rows if r.get("route_category")]
    if not routed:
        return None
    gold = lambda r: gold_category or r.get("category")
    confusion = defaultdict(lambda: defaultdict(int))
    for r in routed:
        confusion[gold(r)][r["route_category"]] += 1
    return {"route_accuracy": round(sum(r["route_category"] == gold(r) for r in routed) / len(routed), 4),
            "routed_to_chain": sum(r.get("route_pipeline") == "chain" for r in routed),
            "route_confusion": {g: dict(p) for g, p in sorted(confusion.items())}}


def print_route_summary(rs):
    if not rs:
        return
    print(f"ROUTER  accuracy={rs['route_accuracy']}  sent to chain={rs['routed_to_chain']}")
    print_table([["true category", "predicted (count)"]] +
                [[g, ", ".join(f"{p}: {n}" for p, n in sorted(ps.items()))] for g, ps in rs["route_confusion"].items()])


def llm_follow_up(question, hits, cfg):
    """(missing, 0–2 follow-up queries) for what the top chain_context_k hop-1 passages don't cover."""
    passages = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits[:cfg["chain_context_k"]], 1))
    response = llm_json(cfg, CHAIN_PROMPT, f"QUESTION:\n{question}\n\nPASSAGES FOUND SO FAR:\n{passages}")
    return response.get("missing", ""), clean_queries(response, 2)


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
    split_method llm: the LLM's self-contained sub-queries (subquery_filter not applied).
    Without subquery_filter: full question + every piece. With it: only the non-vague pieces
    (the full question would count the first piece twice); back to the full question if < 2 pieces survive."""
    if cfg["split_method"] == "llm":
        return llm_decompose(question, cfg)
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


def miss_counts(rows):
    return {f"miss_{g}": sum(r["miss_group"] == g for r in rows) for g in ("0", "A", "B", "C")}


# ---------- dev / test split ----------
# Eval sets with a `split` field (eval_set_v3): compare experiments on dev; test is held out. Test rows are
# scored and recorded like any other, but hidden from eval / compare / table output unless --show-test.
def visible(rows, show_test):
    return rows if show_test else [r for r in rows if r.get("split") != "test"]


def split_metrics(rows):
    """{split: metrics + n + miss counts}, or {} if the eval set has no split field."""
    by_split = defaultdict(list)
    for r in rows:
        if r.get("split"):
            by_split[r["split"]].append(r)
    return {s: {"n": len(rs), **metrics(rs), **miss_counts(rs)} for s, rs in sorted(by_split.items())}


def run_view(run, show_test):
    """The run's headline numbers as the user may see them: dev-only when the run has a hidden test split."""
    ps = run.get("per_split") or {}
    if show_test or "test" not in ps or "dev" not in ps:
        return run, False
    return ps["dev"], True


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
    Returns (col, page_urls, indexed_urls, all_chunk_texts, section_idx). Shared by eval and multihop.
    With `search: exact`, col is an ExactSearch over the collection's stored embeddings."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        col = client.get_collection(name, embedding_function=None)
    except Exception:
        sys.exit(f"Collection {name!r} not found. Run: python retrieval_eval.py index --config {config_path}")
    built = {**DEFAULTS, **json.loads((col.metadata or {}).get("config", "{}"))}
    mismatch = {k: [built.get(k), cfg[k]] for k in INDEX_SETTINGS if built.get(k) != cfg[k]}
    if mismatch:
        sys.exit(f"Collection {name!r} was built with different settings {{setting: [collection, config]}}: {mismatch}")

    exact = cfg["search"] == "exact"
    if cfg["search"] not in ("hnsw", "exact"):
        sys.exit(f"search must be 'hnsw' or 'exact', got {cfg['search']!r}")
    if cfg["bm25_k"] and not exact:
        sys.exit("bm25_k needs search: exact (BM25 is built from the exact-search chunk list)")
    everything = col.get(include=["documents", "metadatas", *(["embeddings"] if exact else [])])
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
    if exact:
        col = ExactSearch(everything["ids"], everything["documents"], everything["metadatas"], everything["embeddings"])
        if cfg["bm25_k"]:
            col.bm25 = BM25(col.ids, col.documents, col.metadatas)
    return col, page_urls, indexed_urls, all_chunk_texts, section_idx


def chain_hop(question, hits, cfg, col, embed, reranker, section_idx):
    """Hop 2: follow-up queries from the top hop-1 passages (as the LLM would see them, i.e. expanded if
    expansion is on); each is retrieved + reranked, then round-robin fused after the hop-1 list, so hop 1's #1 stays
    #1. Returns (hits, missing, follow-up queries); no follow-up queries → hop-1 list unchanged."""
    k = cfg["top_k_retrieve"]
    context = expand_and_collapse(hits[:cfg["chain_context_k"] * 3], section_idx, cfg) if section_idx else hits
    missing, follow_ups = llm_follow_up(question, context, cfg)
    lists = [hits]
    for q in follow_ups:
        h2 = candidates(col, embed, q, cfg)
        lists.append(rerank(reranker, q, h2) if reranker else h2)
    return (round_robin_fuse(lists, k) if follow_ups else hits), missing, follow_ups


# ---------- RT-RAG (Reasoning in Trees, arXiv 2601.11255), adapted for retrieval-only eval ----------
# 1. Entity analysis + reasoning tree (core query, known / unknown entities; parallel / sequential / direct
#    decomposition, depth <= 2). rt_trees candidate trees are sampled; consensus = the most frequent
#    (depth, node count) shape, first tree of that shape wins.
# 2. Bottom-up (post-order) traversal. Leaf: retrieve + rerank, LLM answers from the top rt_answer_k passages or says
#    None → query rewritten (up to rt_rewrites times). Sequential children get earlier siblings' answers substituted
#    for "[answer N]". A parent whose child stays unanswered becomes a leaf (adaptive leaf: retrieved directly);
#    otherwise the LLM combines the child answers (no retrieval). rt_answer_samples answers per node, majority kept
#    (rejection sampling; 1 = off).
# 3. Ranked list for scoring = round-robin fusion of every retrieval list, in traversal order (leaves first).
RT_TREE_PROMPT = ("rt_tree", "v1", """You plan how to research a user's question about Canadian immigration (IRCC)
with a search engine over IRCC web pages. First analyse the question:
- core_query: what the user ultimately wants to know (one sentence).
- known_entities: facts the user states about their situation (program, length, permit status, school, dates, who).
- unknown_entities: facts that must be looked up before the core query can be answered.
Then build a reasoning tree. Each node has a "question" and optional "children"; a node with children also has a
"pattern":
- "parallel": the children are independent sub-questions whose answers together answer the node.
- "sequential": a later child needs an earlier child's answer; refer to it as "[answer 1]", "[answer 2]" (1-based
  position among its siblings).
A node without children is a leaf and must be answerable by a single search. Use "direct" (no children) for the root
if the question is already single-hop. Max depth 2 below the root, at most 4 leaves in total.
Every node question must be self-contained: include the known entities that matter for it (e.g. "5-month program",
"no study permit", "PhD at a public university"), no pronouns like "that" or "both of us". Keep questions short
(at most 25 words). Do not answer anything.
Return JSON: {"core_query": "...", "known_entities": ["..."], "unknown_entities": ["..."],
"tree": {"question": "...", "pattern": "parallel|sequential|direct", "children": [{"question": "...", ...}]}}""")

RT_ANSWER_PROMPT = ("rt_answer", "v1", """Answer the question using ONLY the passages from IRCC web pages below.
Answer in at most 2 sentences. If the passages do not contain the information needed, answer exactly "None".
Return JSON: {"answer": "..."}""")

RT_REWRITE_PROMPT = ("rt_rewrite", "v1", """A search over IRCC (Canadian immigration) web pages did not find the answer
to the search query below. Rewrite the query so it keeps the same meaning but uses different wording: synonyms, the
official IRCC term, or a more general phrasing of the same need. At most 20 words, no search operators.
Return JSON: {"query": "..."}""")

RT_COMBINE_PROMPT = ("rt_combine", "v1", """Answer the question using ONLY the answers to its sub-questions below.
Answer in at most 3 sentences. If they are not enough, answer exactly "None".
Return JSON: {"answer": "..."}""")


def rt_shape(node, depth=0):
    """(depth, node count) of a tree."""
    kids = node.get("children") or []
    if not kids:
        return depth, 1
    shapes = [rt_shape(c, depth + 1) for c in kids]
    return max(d for d, _ in shapes), 1 + sum(n for _, n in shapes)


def rt_clean_tree(node, cfg, depth=0):
    """Valid node dict or None; cuts below depth 2 and drops empty questions."""
    if not isinstance(node, dict) or not isinstance(node.get("question"), str) or not node["question"].strip():
        return None
    kids = [c for c in (rt_clean_tree(c, cfg, depth + 1) for c in (node.get("children") or [])[:4]) if c] if depth < 2 else []
    pattern = node.get("pattern") if node.get("pattern") in ("parallel", "sequential") and kids else "direct"
    return {"question": node["question"].strip(), "pattern": pattern, "children": kids}


def rt_count_leaves(node):
    return 1 if not node["children"] else sum(rt_count_leaves(c) for c in node["children"])


def rt_build_tree(question, cfg):
    """Sample rt_trees trees, keep the first one with the most frequent shape. Falls back to a single-node tree."""
    candidates = []
    for i in range(cfg["rt_trees"]):
        r = llm_json(cfg, RT_TREE_PROMPT, question, sample=i + 1)
        tree = rt_clean_tree(r.get("tree") if isinstance(r, dict) else None, cfg)
        if tree and rt_count_leaves(tree) <= cfg["rt_max_leaves"]:
            candidates.append((r, tree))
    if not candidates:
        return {"question": question, "pattern": "direct", "children": []}, {}, []
    shapes = [rt_shape(t) for _, t in candidates]
    best = max(shapes, key=lambda s: (shapes.count(s), -shapes.index(s)))
    r, tree = candidates[shapes.index(best)]
    analysis = {k: r.get(k) for k in ("core_query", "known_entities", "unknown_entities")}
    return tree, analysis, [list(s) for s in shapes]


def rt_answer(cfg, prompt, user):
    """Majority answer over rt_answer_samples draws (rejection sampling); None if the majority is 'None'."""
    answers = []
    for i in range(cfg["rt_answer_samples"]):
        a = llm_json(cfg, prompt, user, sample=i)  # sample 0 = the plain cached call
        a = (a.get("answer") if isinstance(a, dict) else None) or "None"
        answers.append(a.strip())
    norm = [a.lower().strip(" .") for a in answers]
    top = max(norm, key=lambda a: (norm.count(a), -norm.index(a)))
    return None if top == "none" else answers[norm.index(top)]


def rt_rag_retrieve(question, cfg, col, embed, reranker, section_idx):
    """RT-RAG traversal for one question. Returns (fused hits, queries searched, trace dict)."""
    k = cfg["top_k_retrieve"]
    lists, queries, log = [], [], []
    tree, analysis, shapes = rt_build_tree(question, cfg)

    def search(q):
        hits = retrieve(col, embed, cfg["query_prefix"] + q, k)
        hits = rerank(reranker, q, hits) if reranker else hits
        lists.append(hits)
        queries.append(q)
        return hits

    def answer_leaf(q):
        for attempt in range(cfg["rt_rewrites"] + 1):
            hits = search(q)
            context = expand_and_collapse(hits[:cfg["rt_answer_k"] * 3], section_idx, cfg) if section_idx else hits
            passages = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(context[:cfg["rt_answer_k"]], 1))
            ans = rt_answer(cfg, RT_ANSWER_PROMPT, f"QUESTION:\n{q}\n\nPASSAGES:\n{passages}")
            log.append({"query": q, "attempt": attempt, "answer": ans})
            if ans is not None or attempt == cfg["rt_rewrites"]:
                return ans
            q = (llm_json(cfg, RT_REWRITE_PROMPT, f"{q}\n(rewrite #{attempt + 1})").get("query") or q).strip()

    def solve(node, sibling_answers):
        q = node["question"]
        for n, a in enumerate(sibling_answers, 1):  # sequential: substitute earlier siblings' answers
            q = q.replace(f"[answer {n}]", a or "unknown")
        if not node["children"]:
            return answer_leaf(q)
        child_answers = []
        for child in node["children"]:
            child_answers.append(solve(child, child_answers if node["pattern"] == "sequential" else []))
        if any(a is None for a in child_answers):  # adaptive leaf: the parent is searched directly
            log.append({"query": q, "note": "child unanswered -> parent searched as a leaf"})
            return answer_leaf(q)
        subs = "\n".join(f"- {c['question']}: {a}" for c, a in zip(node["children"], child_answers))
        ans = rt_answer(cfg, RT_COMBINE_PROMPT, f"QUESTION:\n{q}\n\nSUB-ANSWERS:\n{subs}")
        log.append({"query": q, "combined_from_children": True, "answer": ans})
        return ans

    root_answer = solve(tree, [])
    hits = round_robin_fuse(lists, k) if len(lists) > 1 else lists[0]
    trace = {"rt_analysis": analysis, "rt_tree": tree, "rt_tree_shapes": shapes, "rt_steps": log,
             "rt_root_answer": root_answer}
    return hits, queries, trace


def retrieve_for_question(question, cfg, col, embed, reranker, section_idx, trace=None):
    """The full retrieval pipeline for one question: split → retrieve → rerank → fuse → [chain hop 2] → expand.
    Returns (ranked hits, queries run). Shared by eval and multihop. If a trace dict is given, the chain hop
    records its missing-info note and hop-2 queries in it."""
    if cfg["rt_rag"]:  # RT-RAG replaces split → retrieve → fuse with a reasoning-tree traversal
        hits, queries, rt_trace = rt_rag_retrieve(question, cfg, col, embed, reranker, section_idx)
        if trace is not None:
            trace.update(rt_trace)
        return (expand_and_collapse(hits, section_idx, cfg) if section_idx else hits), queries
    k = cfg["top_k_retrieve"]
    # One query unless split_queries splits the question; then each query's list is fused.
    queries = make_queries(question, cfg)
    ranked_lists = []
    for q in queries:
        hits = candidates(col, embed, q, cfg)
        if reranker:
            hits = rerank(reranker, q, hits)
        ranked_lists.append(hits)
    if len(queries) == 1:
        hits = ranked_lists[0]
    else:
        fuse = {"rrf": rrf_fuse, "round_robin": round_robin_fuse}[cfg["fusion"]]
        hits = fuse(ranked_lists, k)
    use_chain = cfg["chain_hops"] >= 2
    if cfg["route"] == "llm":  # router: only predicted categories in chain_categories get the chain hop
        category = llm_route(question, cfg)
        use_chain = use_chain and category in cfg["chain_categories"]
        if trace is not None:
            trace.update(route_category=category, route_pipeline="chain" if use_chain else "base")
    if use_chain:  # chain retrieval: LLM reads hop-1 passages, queries what is still missing
        hits, missing, follow_ups = chain_hop(question, hits, cfg, col, embed, reranker, section_idx)
        if trace is not None:
            trace.update(chain_missing=missing, hop2_queries=follow_ups)
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
            if item.get("split"):
                row["split"] = item["split"]
            try:
                trace = {}  # filled only by optional features (chain / RT-RAG / routing); empty otherwise
                hits, queries = retrieve_for_question(item["question"], cfg, col, embed, reranker, section_idx, trace)
                if len(queries) > 1:
                    row["sub_queries"] = queries
                row.update(trace)
                row.update(score(item, hits, indexed_urls, all_chunk_texts))
                row["ctx_words5"] = sum(len(h["text"].split()) for h in hits[:5])  # text the LLM would get in 5 slots
                keys = ("rank", "chunk_rank", "retrieval_rank", "url", "chunk_id", "unit", "expanded_words",
                        "score", "rerank_score", "rrf_score", "bm25_rank")
                row["top5"] = [{**{key: h[key] for key in keys if key in h}, "text": h["text"][:300]} for h in hits[:5]]
            except LLMError as e:
                f.close()
                out_path.unlink()  # no partial results file, no run record; cached LLM responses are kept
                sys.exit(f"\nLLM call failed on {item['id']}, run aborted (nothing recorded): {e}")
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
        **miss_counts(rows),
        "per_category": {c: metrics(rs) for c, rs in sorted(by_cat.items())},
        **({"per_split": split_metrics(rows)} if split_metrics(rows) else {}),
        "ctx_words5": ctx_words5(rows),
        "results_file": str(out_path.relative_to(ROOT)),
    }
    if LLM_STATS["calls"] or LLM_STATS["cache_hits"]:  # only runs that use LLM features get these fields
        run.update(llm_calls=LLM_STATS["calls"], llm_cache_hits=LLM_STATS["cache_hits"])
    if route_summary(rows):
        run.update(route_summary(rows))
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"\nRUN {run_id}  collection={name}  questions={run['num_questions']}  errors={run['num_errors']}")
    if missing_pages:
        print(f"  WARNING: {len(missing_pages)} gold page(s) not indexed -> those questions are miss group 0")
    shown = visible(rows, args.show_test)
    if len(shown) < len(rows):
        print(f"  DEV SPLIT ONLY: {len(rows) - len(shown)} test questions scored and recorded but hidden "
              f"(--show-test, or `compare --show-test`, to see them)")
    m, g = metrics(shown), miss_counts(shown)
    print(f"  Hit@5={m['hit5']}  Hit@50={m['hit50']}  MRR={m['mrr']}  "
          f"misses 0/A/B/C = {g['miss_0']}/{g['miss_A']}/{g['miss_B']}/{g['miss_C']}")
    shown_cat = defaultdict(list)
    for r in shown:
        shown_cat[r["category"]].append(r)
    print_table([["category", "n", "Hit@5", "Hit@50", "MRR"]] +
                [[c, len(rs), *(metrics(rs)[k] for k in ("hit5", "hit50", "mrr"))] for c, rs in sorted(shown_cat.items())])
    if args.show_test and run.get("per_split"):
        print_table([["split", "n", "Hit@5", "Hit@50", "MRR"]] +
                    [[s, v["n"], v["hit5"], v["hit50"], v["mrr"]] for s, v in run["per_split"].items()])
    print_route_summary(route_summary(rows))
    print("\nALL RUNS")
    print_runs_table(read_runs(), args.show_test)
    if len(prev) >= 1:  # what changed vs the previous run
        cmd_compare(argparse.Namespace(run_1=prev[-1]["run_id"], run_2=run_id, show_test=args.show_test))


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


def print_runs_table(runs, show_test=False):
    labels = eval_set_labels(read_runs())  # labels from the full history, so they don't shift when filtering
    label = lambda r: labels.get(r.get("eval_set_hash"), "v?")
    views = [run_view(r, show_test) for r in runs]
    best = {}  # best Hit@5 per eval set version: scores on different question sets aren't comparable
    for i, (m, _) in enumerate(views):
        v = label(runs[i])
        if m["hit5"] is not None and (v not in best or m["hit5"] > views[best[v]][0]["hit5"]):
            best[v] = i
    table = [["run", "what changed", "Hit@5", "Hit@50", "MRR", "0/A/B/C"]]
    for i, r in enumerate(runs):
        m, dev_only = views[i]
        if i and label(r) != label(runs[i - 1]):
            n = f"{m['n']} dev of " if dev_only else ""
            table.append([f"── eval set {label(r)} ──", f"{n}{r.get('eval_set_n', r['num_questions'])} questions", "", "", "", ""])
        mark = ("!" if r["missing_pages"] else "") + (" dev" if dev_only else "") + (" *" if best.get(label(r)) == i else "")
        table.append([f"{r['run_id']} [{label(r)}]{mark}", r["what_changed"] or "",
                      m["hit5"], m["hit50"], m["mrr"], f"{m['miss_0']}/{m['miss_A']}/{m['miss_B']}/{m['miss_C']}"])
    print_table(table, wrap={1: 30})
    print("[vN] = eval set version   * = best Hit@5 within that version   ! = run had gold pages missing from the index"
          + ("   dev = dev split only, test held out (--show-test)" if any(d for _, d in views) else ""))


def cmd_table(args):
    """Per-question ranks side by side for the given runs (default: all runs)."""
    runs = read_runs()
    if args.run_ids:
        by_id = {r["run_id"]: r for r in runs}
        runs = [by_id[i] for i in args.run_ids]
    results = [{x["item_id"]: x for x in visible(list(map(json.loads, (ROOT / r["results_file"]).read_text(encoding="utf-8").splitlines())), args.show_test)}
               for r in runs]

    def cell(x):
        if x is None:
            return "n/a"
        if x["rank"] is not None and x["rank"] <= 5:
            return f"{x['rank']} ✓"
        return f"{'-' if x['rank'] is None else x['rank']} ({x['miss_group']})"

    print_runs_table(runs, args.show_test)
    print()
    item_ids = list(dict.fromkeys(i for res in results for i in res))
    table = [["item", "category", *[r["run_id"] for r in runs]]]
    for iid in item_ids:
        category = next(res[iid]["category"] for res in results if iid in res)
        table.append([iid, category, *[cell(res.get(iid)) for res in results]])
    for label, key in (("Hit@5", "hit5"), ("Hit@50", "hit50"), ("MRR", "mrr")):
        table.append([label, "", *[run_view(r, args.show_test)[0][key] for r in runs]])
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
    show_test = getattr(args, "show_test", False)
    hidden = {i for res in (a, b) for i, r in res.items() if r.get("split") == "test"} if not show_test else set()
    a, b = ({i: r for i, r in res.items() if i not in hidden} for res in (a, b))
    shared = sorted(set(a) & set(b))
    labels = eval_set_labels(all_runs)
    v1, v2 = (labels.get(runs.get(r, {}).get("eval_set_hash"), "v?") for r in (args.run_1, args.run_2))
    print(f"\nCOMPARE {args.run_1} [{v1}] -> {args.run_2} [{v2}]")
    if hidden:
        print(f"  DEV SPLIT ONLY: {len(hidden)} test questions hidden (--show-test to include them).")
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
    """Item metrics from passage ranks (None = not retrieved). hit@k: any gold passage in the top k (like the
    general eval's Hit@k); joint@k: all of them (headline: joint@5); mrr_first = 1/best rank (like the general
    MRR); mrr_joint = 1/worst rank (depth needed to have all the evidence); passage_mrr = mean 1/rank per passage."""
    ok = [r for r in rows if not r.get("error")]
    item_ranks = [[rk or float("inf") for rk in r["passage_ranks"]] for r in ok]
    ranks = [rk for rs in item_ranks for rk in rs]
    groups = [g for r in ok for g in r["miss_groups"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "num_items": len(ok),
        "joint_hit5": mean([max(rs) <= 5 for rs in item_ranks]),
        "joint_hit10": mean([max(rs) <= 10 for rs in item_ranks]),
        "joint_hit50": mean([max(rs) <= 50 for rs in item_ranks]),
        "hit5": mean([min(rs) <= 5 for rs in item_ranks]),
        "hit10": mean([min(rs) <= 10 for rs in item_ranks]),
        "hit50": mean([min(rs) <= 50 for rs in item_ranks]),
        "passage_recall5": mean([sum(rk <= 5 for rk in rs) / len(rs) for rs in item_ranks]),
        "passage_recall10": mean([sum(rk <= 10 for rk in rs) / len(rs) for rs in item_ranks]),
        "mrr_first": mean([1 / min(rs) for rs in item_ranks]),
        "mrr_joint": mean([1 / max(rs) for rs in item_ranks]),
        "passage_mrr": mean([1 / rk for rk in ranks]),
        **{f"miss_{g}": groups.count(g) for g in ("0", "A", "B", "C")},
    }


MULTIHOP_TABLE_COLS = (("joint@5", "joint_hit5"), ("joint@10", "joint_hit10"), ("joint@50", "joint_hit50"),
                       ("Hit@5", "hit5"), ("Hit@10", "hit10"), ("Hit@50", "hit50"), ("recall@5", "passage_recall5"),
                       ("recall@10", "passage_recall10"), ("MRR first", "mrr_first"), ("MRR joint", "mrr_joint"),
                       ("passage MRR", "passage_mrr"))


def read_multihop_runs():
    if not MULTIHOP_RUNS.exists():
        return []
    return [json.loads(l) for l in MULTIHOP_RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def print_multihop_runs_table(runs):
    labels = eval_set_labels(runs)
    best = max(range(len(runs)), key=lambda i: (runs[i]["joint_hit5"] or -1, runs[i].get("mrr_joint") or -1, -i),
               default=None)
    table = [["run_id", "what_changed", "config_diff", *(c for c, _ in MULTIHOP_TABLE_COLS), "0/A/B/C"]]
    for i, r in enumerate(runs):
        diff = ", ".join(f"{k}: {o}->{n}" for k, (o, n) in r["config_diff"].items()) or "-"
        table.append([f"{r['run_id']} [{labels.get(r.get('eval_set_hash'), 'v?')}]" + (" *" if i == best else ""),
                      r["what_changed"] or "", diff, *(r.get(key, "-") for _, key in MULTIHOP_TABLE_COLS),
                      f"{r['miss_0']}/{r['miss_A']}/{r['miss_B']}/{r['miss_C']}"])
    print_table(table, wrap={1: 30, 2: 30})
    print("* = best joint@5 (tie-break MRR joint)   [vN] = multi-hop item set version   "
          "0/A/B/C = missing passages (not in top 5) by group   - = not computed (run multihop-rescore)")


def cmd_multihop(args):
    name, cfg = load_config(args.config)
    col, page_urls, indexed_urls, all_chunk_texts, section_idx = open_eval_collection(name, cfg, args.config)

    # Item set = draft_multihop_v0 only; the eval_set multi-hop items are no longer mixed in (no duplicates).
    eval_items = []
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
    items = [(d, "draft_v0") for d in drafts]

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
                trace = {}
                hits, queries = retrieve_for_question(item["question"], cfg, col, embed, reranker, section_idx, trace)
                if queries != [item["question"]]:  # split, or rewritten by the LLM
                    row["sub_queries"] = queries
                row.update(trace)
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
            except LLMError as e:
                f.close()
                out_path.unlink()  # no partial results file, no run record; cached LLM responses are kept
                sys.exit(f"\nLLM call failed on {item['id']}, run aborted (nothing recorded): {e}")
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
        "llm_calls": LLM_STATS["calls"], "llm_cache_hits": LLM_STATS["cache_hits"],
        **multihop_metrics(rows),
        **(route_summary(rows, "multi-hop") or {}),
        "by_source": {s: multihop_metrics(rs) for s, rs in sorted(by_source.items())},
        "results_file": str(out_path.relative_to(ROOT)),
    }
    with open(MULTIHOP_RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")

    print(f"\nMULTI-HOP RUN {run_id}  collection={name}  items={run['num_items']}  errors={run['num_errors']}  "
          f"LLM calls={run['llm_calls']} cache hits={run['llm_cache_hits']}")
    table = [["source", "items", *(c for c, _ in MULTIHOP_TABLE_COLS), "missing 0/A/B/C"]]
    for label, m in [*run["by_source"].items(), ("ALL", multihop_metrics(rows))]:
        table.append([label, m["num_items"], *(m[key] for _, key in MULTIHOP_TABLE_COLS),
                      f"{m['miss_0']}/{m['miss_A']}/{m['miss_B']}/{m['miss_C']}"])
    print_table(table)
    print_route_summary(route_summary(rows, "multi-hop"))
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
    for label, key in MULTIHOP_TABLE_COLS:
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


def cmd_multihop_rescore(args):
    """Add metrics that a past run lacks, recomputed from its saved passage ranks (no retrieval). Existing
    values in runs.jsonl are never overwritten."""
    runs = read_multihop_runs()
    for run in runs:
        if args.run_ids and run["run_id"] not in args.run_ids:
            continue
        rows = [json.loads(l) for l in (ROOT / run["results_file"]).read_text(encoding="utf-8").splitlines() if l.strip()]
        added = [k for k in multihop_metrics(rows) if k not in run and k != "num_items"]  # num_items is a dict here
        for k in added:
            run[k] = multihop_metrics(rows)[k]
        by_source = defaultdict(list)
        for r in rows:
            by_source[r["source"]].append(r)
        for s, rs in by_source.items():
            for k, v in multihop_metrics(rs).items():
                run["by_source"].setdefault(s, {}).setdefault(k, v)
        print(f"  {run['run_id']}: added {', '.join(added) or 'nothing'}")
    with open(MULTIHOP_RUNS, "w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
    print("\nALL MULTI-HOP RUNS")
    print_multihop_runs_table(runs)


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
    show_test_help = "also show the held-out test split (eval sets with a `split` field); default: dev only"
    p.add_argument("--show-test", action="store_true", help=show_test_help)
    p = sub.add_parser("compare")
    p.add_argument("run_1", nargs="?", help="default: previous run")
    p.add_argument("run_2", nargs="?", help="default: latest run")
    p.add_argument("--show-test", action="store_true", help=show_test_help)
    p = sub.add_parser("table", help="per-question ranks side by side")
    p.add_argument("run_ids", nargs="*", help="runs to show (default: all)")
    p.add_argument("--show-test", action="store_true", help=show_test_help)
    p = sub.add_parser("multihop", help="multi-hop eval (eval_set multi-hop items + validated drafts)")
    p.add_argument("--config", required=True)
    p.add_argument("--change", required=True, help="what you changed since the last multi-hop run")
    p.add_argument("--validate-only", action="store_true", help="only validate the draft items and write the report")
    p = sub.add_parser("multihop-compare", help="items that improved / got worse between two multi-hop runs")
    p.add_argument("run_1")
    p.add_argument("run_2")
    p = sub.add_parser("multihop-rescore", help="add new metrics to past multi-hop runs from their saved ranks")
    p.add_argument("run_ids", nargs="*", help="runs to rescore (default: all)")
    args = ap.parse_args()
    {"index": cmd_index, "eval": cmd_eval, "compare": cmd_compare, "table": cmd_table,
     "multihop": cmd_multihop, "multihop-compare": cmd_multihop_compare,
     "multihop-rescore": cmd_multihop_rescore}[args.cmd](args)


if __name__ == "__main__":
    main()
