"""Prototype eval harness: RAG answer -> retrieval check -> correctness judge -> grounding judge.

Usage (from repo root):
    .venv/Scripts/python scripts/run_eval.py --category single-fact
"""
import argparse
import json
import os
import re
import subprocess
import sys
import traceback
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data/eval_set/eval_set.jsonl"
MANIFEST = ROOT / "data/raw/2026-09-20/manifest.jsonl"
RESULTS_DIR = ROOT / "results"
RUNS_FILE = ROOT / "runs.jsonl"

# Mirrors the hardcoded values inside add_to_vdb.call_rag -- keep in sync by hand.
RAG_SETTINGS = {
    "entrypoint": "scripts/add_to_vdb.py:call_rag",
    "model": "gpt-5-mini",
    "top_k": 5,
    "chunk_size_words": 180,
    "chunk_overlap_words": 30,
    "embed_model": "chroma-default-all-MiniLM-L6-v2",
    "collection": "guidance_v0",
}

load_dotenv(ROOT / ".env")
# call_rag uses paths relative to scripts/ (indexes/chroma, ../data/raw/...)
os.chdir(ROOT / "scripts")
sys.path.insert(0, str(ROOT / "scripts"))
from add_to_vdb import call_rag  # noqa: E402


# ---------- retrieval check ----------
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def norm(s):
    # Compare word tokens only: the crawler rewrites list markup ("criteria : - Associate of Arts"),
    # so punctuation/bullets/whitespace differ from the quote as it appears on the page.
    s = unicodedata.normalize("NFKC", s or "").translate(QUOTE_MAP).lower()
    return " " + " ".join(re.findall(r"\w+", s)) + " "


def check_retrieval(item, chunks, manifest):
    golds = item.get("gold_passages") or []
    if not golds:
        return {"url_retrieved": None, "quote_retrieved": None, "url_rank": None, "quote_ranks": []}
    ok_urls = set()
    for g in golds:
        ok_urls.add(g["url"])
        row = manifest.get(g.get("content_hash"))
        if row:
            ok_urls.update(u for u in (row.get("url"), row.get("final_url")) if u)
    url_rank = next((c["rank"] for c in chunks if c["url"] in ok_urls), None)
    quote_ranks = []
    for g in golds:
        q = norm(g["quote"])
        quote_ranks.append(next((c["rank"] for c in chunks if q in norm(c["text"])), None))
    return {
        "url_retrieved": url_rank is not None,
        "quote_retrieved": all(r is not None for r in quote_ranks),
        "url_rank": url_rank,
        "quote_ranks": quote_ranks,
    }


# ---------- LLM judges ----------
CATEGORY_RULES = {
    "conditional": "The answer must state the condition(s) under which it holds; omitting the condition is at best partial.",
    "unanswerable": "The sources don't answer this. A correct answer declines or redirects to where to find it; any made-up specifics make it incorrect.",
    "ambiguous": "The answer must recognise the ambiguity or cover the relevant cases; picking one case silently is at best partial.",
    "temporal": "The answer must reflect the current rule, not an outdated one.",
    "multi-hop": "Every required fact must be present; missing one is partial.",
}

CORRECTNESS_SYSTEM = """You grade answers from an assistant about Canadian IRCC study-permit guidance.
Compare the MODEL ANSWER to the GOLD ANSWER and GOLD QUOTE. Judge substance, not wording or length.
Extra correct detail is fine. Contradicting the gold, or missing its key point, is not.
verdict: "correct" (key point right, nothing contradicting), "partial" (right direction but missing or
hedging a key element), "incorrect" (wrong, contradicting, or fails to answer).
Return JSON only: {"verdict": "correct"|"partial"|"incorrect", "reason": "<one or two sentences>"}"""

GROUNDING_SYSTEM = """You check whether an answer is supported by the retrieved SOURCES below, and nothing else.
Use no outside knowledge. A claim is unsupported if no source states or directly implies it.
Ignore citation markers like [1], and ignore the boilerplate disclaimer ("Not official or legal advice...").
Statements that the sources don't contain the answer count as supported.
Return JSON only: {"grounded": true|false, "unsupported_claims": ["<claim>", ...]}"""


def ask_json(client, model, system, user):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


def judge_correctness(client, model, item, answer):
    gold = item.get("gold_answer") or item.get("answer") or ""
    quotes = "\n".join(f"- {g['quote']}" for g in item.get("gold_passages") or []) or "(none)"
    rule = CATEGORY_RULES.get(item.get("category"), "")
    user = (f"CATEGORY: {item.get('category')}\n" + (f"CATEGORY RULE: {rule}\n" if rule else "")
            + f"\nQUESTION: {item['question']}\n\nGOLD ANSWER: {gold}\n\nGOLD QUOTE(S):\n{quotes}"
            + f"\n\nMODEL ANSWER:\n{answer}")
    out = ask_json(client, model, CORRECTNESS_SYSTEM, user)
    if out.get("verdict") not in ("correct", "partial", "incorrect"):
        raise ValueError(f"bad verdict: {out!r}")
    return {"verdict": out["verdict"], "reason": out.get("reason", "")}


def judge_grounding(client, model, answer, chunks):
    sources = "\n\n".join(f"[{c['rank']}] {c['url']}\n{c['text']}" for c in chunks) or "(no sources)"
    out = ask_json(client, model, GROUNDING_SYSTEM, f"SOURCES:\n{sources}\n\nANSWER:\n{answer}")
    if not isinstance(out.get("grounded"), bool):
        raise ValueError(f"bad grounding output: {out!r}")
    return {"grounded": out["grounded"], "unsupported_claims": out.get("unsupported_claims") or []}


def failure_type(row):
    if row.get("error"):
        return "error"
    correct = row["verdict"] == "correct"
    if correct and row["grounded"]:
        return None
    if row["quote_retrieved"] is False:
        return "retrieval"
    if not correct:
        return "reading"
    return "hallucination"


# ---------- run ----------
def eval_one(client, judge_model, item, manifest):
    row = {"id": item["id"], "category": item.get("category"), "question": item["question"]}
    stage = "rag"
    try:
        hits, answer = call_rag(item["question"])
        chunks = [{"rank": i, "url": h.get("url"), "score": round(h.get("score", 0), 4), "text": h.get("text", "")}
                  for i, h in enumerate(hits, 1)]
        row.update(answer=answer, chunks=chunks)
        stage = "retrieval_check"
        row.update(check_retrieval(item, chunks, manifest))
        stage = "correctness_judge"
        row.update(judge_correctness(client, judge_model, item, answer))
        stage = "grounding_judge"
        row.update(judge_grounding(client, judge_model, answer, chunks))
    except Exception as e:
        row["error"] = {"stage": stage, "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    row["failure_type"] = failure_type(row)
    return row


def print_summary(rows):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cols = ["n", "quote_ret", "correct", "partial", "incorrect", "grounded", "errors"]

    def stats(rs):
        ok = [r for r in rs if not r.get("error")]
        applicable = [r for r in ok if r.get("quote_retrieved") is not None]
        return [
            len(rs),
            f"{sum(r['quote_retrieved'] for r in applicable)}/{len(applicable)}" if applicable else "n/a",
            sum(r.get("verdict") == "correct" for r in ok),
            sum(r.get("verdict") == "partial" for r in ok),
            sum(r.get("verdict") == "incorrect" for r in ok),
            f"{sum(r['grounded'] for r in ok)}/{len(ok)}" if ok else "n/a",
            len(rs) - len(ok),
        ]

    table = [[cat, *stats(rs)] for cat, rs in sorted(by_cat.items())] + [["TOTAL", *stats(rows)]]
    header = ["category", *cols]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]
    fmt = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for line in table:
        print(fmt(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default=str(EVAL_FILE))
    ap.add_argument("--category", help="only run this category, e.g. single-fact")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--judge-model", default="gpt-5")
    args = ap.parse_args()

    items = [json.loads(l) for l in Path(args.eval_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.category:
        items = [i for i in items if i.get("category") == args.category]
    items = items[: args.limit] if args.limit else items
    manifest = {}
    for l in MANIFEST.read_text(encoding="utf-8").splitlines():
        if l.strip():
            m = json.loads(l)
            manifest[m.get("content_hash")] = m

    now = datetime.now(timezone.utc).astimezone()
    run_id = now.strftime("%Y%m%d-%H%M%S")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None

    client = OpenAI(max_retries=3)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.jsonl"
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            row = eval_one(client, args.judge_model, item, manifest)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = row["error"]["message"] if row.get("error") else \
                f"{row['verdict']}, quote={row['quote_retrieved']}, grounded={row['grounded']}"
            print(f"[{n}/{len(items)}] {row['id']}: {status} -> failure={row['failure_type']}")

    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_id": run_id, "date": now.isoformat(timespec="seconds"), "git_commit": commit,
            "rag": RAG_SETTINGS, "judge_model": args.judge_model,
            "eval_file": str(Path(args.eval_file).resolve().relative_to(ROOT)),
            "category_filter": args.category, "limit": args.limit, "n": len(rows),
            "results_file": str(out_path.relative_to(ROOT)), 
        }) + "\n")

    print(f"\nrun {run_id} -> {out_path.relative_to(ROOT)}\n")
    print_summary(rows)


if __name__ == "__main__":
    main()
