# Eval set specification

How the evaluation questions in `data/eval_set/` were built, what every field means, and the rules
a gold passage must follow so that `retrieval_eval.py` (retrieval-only) and `scripts/run_eval.py`
(full RAG + judges) can score it. Follow this document when adding or editing items.

## Files

| File | What it is |
|---|---|
| `eval_set.jsonl` | **The eval set used by both harnesses.** 30 items, 5 per category, all human-verified. |
| `draft_<category>_v0.jsonl` | Candidate pools per category (17–25 items each). Mostly `llm_draft` / `unverified`. Items are promoted into `eval_set.jsonl` after review. |
| `eval_set_singlefact.jsonl` | Candidate pool for `single-fact` (older, simpler schema with `answer`, `valid_until`, `stale_risk`). |

All items are written against **one crawl snapshot**: `data/raw/2026-09-20/` (`manifest.jsonl` +
`<key>/<key>.txt`). Quotes must come from that snapshot's `.txt` files, not from the live website.

## Item schema

One JSON object per line, UTF-8.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `sf-NNNN`, `mh-NNN`, `cond-NNN`, `temp-NNN`, `amb-NNN`, `unans-NNN`. Never reused. |
| `question` | str | Written the way a real student would ask: informal, paraphrased, sometimes with a wrong assumption. |
| `category` | str | One of the six categories below. |
| `answerable` | bool | `false` only for `unanswerable`. |
| `gold_answer` (`answer` in single-fact) | str | The correct answer, using only what the corpus says. |
| `required_facts` | list[str] | Atomic facts a correct answer must contain. Empty for `unanswerable`. |
| `gold_passages` | list[obj] | Evidence: `{url, content_hash, quote}`. Empty for `unanswerable`. Rules below. |
| `valid_as_of` | date | Date the item was checked against the corpus (for `temporal`: the rule's effective date). |
| `valid_until` / `stale_risk` | date / low·medium·high | single-fact only: how likely the fact is to change. |
| `source` | str | `synthetic_from_page` (written by a person from a page) or `llm_draft` (drafted by an LLM from a page). |
| `verified_by` | str | `human` once a person has checked question, answer and every quote against the `.txt` file; otherwise `unverified`. Only `human` items go into `eval_set.jsonl`. |
| `vocab_overlap_flag` | str | Which words/phrases the question shares with the gold quote (`"none"` if only generic terms). Questions should paraphrase, so retrieval is not tested by copy-paste. |
| `notes` | str | Category-specific justification (see below). |

## How items are generated

1. **Pick source page(s)** from the crawl manifest (two or more pages for `multi-hop`).
2. **Draft** the question, gold answer, required facts and gold quote(s) from the page text,
   either by hand (`synthetic_from_page`) or with an LLM (`llm_draft`, `verified_by: unverified`).
3. **Verify by a person**: open the page's `.txt` file, confirm each quote is copied from it
   (rules below), confirm the answer follows from the quotes alone, fill `vocab_overlap_flag` and
   `notes`, set `verified_by: human`.
4. **Promote** into `eval_set.jsonl`, keeping the category balance (currently 5 each).
5. **Check matching** with the snippet at the end of this file; every quote must be found in its page.

## Categories

| Category | What it tests | Required in `notes` |
|---|---|---|
| `single-fact` | One fact from one passage. | (optional) |
| `multi-hop` | Answer needs facts from **two different pages**; neither page alone answers it. ≥2 gold passages. | Why each page alone is insufficient. |
| `conditional` | Answer depends on a condition stated in the source. | What happens when the condition holds and when it fails. |
| `temporal` | A rule that changed on a date; the answer must give the current rule. | Effective date, before-state, after-state. |
| `ambiguous` | The question leaves out a detail that changes the answer. ≥1 gold passage per interpretation. | The missing detail and each interpretation's answer. |
| `unanswerable` | The corpus does not contain the answer; a correct answer declines/redirects. No gold passages. | Search terms tried against the `.txt` files and the near-miss pages found. |

## Gold passage rules (read this before writing a quote)

Retrieval is scored by checking whether a retrieved **chunk contains the quote**. A quote that is not
copied faithfully can never be found, and the question is counted as a miss (group A) even when
retrieval returned the right passage.

### 1. Copy, don't paraphrase
Each quote is copied from the page's `.txt` file. Do not reword, reorder, summarise or fix typos.

### 2. Formatting differences are fine
Before comparing, both the quote and the chunk are reduced to **lowercase word tokens**
(`norm_text()` in `retrieval_eval.py`). These differences are therefore ignored:

- punctuation and spacing (`criteria :` vs `criteria:`), commas, parentheses, `|` table separators
- list bullets and line breaks the crawler adds (`criteria :\n- Associate of Arts` vs `criteria: Associate of Arts`)
- curly vs straight quotes and apostrophes, upper/lower case

So you may join list items or table rows into one line, but the **words** must be identical and in the same order.

### 3. Use `...` whenever you skip text (most important rule)
**`...` is not decoration: the matcher splits the quote on it.** Each piece between `...` is
searched for separately, so `...` is how you tell the code "the text in between is left out".

- Insert ` ... ` at **every** place where you skip anything: a sentence, a list item, a heading, a whole section.
- Each piece must be copied verbatim and be **contiguous** on the page.
- The quote's rank is the **worst** rank of its pieces, and the quote counts as found only if
  **every** piece is found in the retrieved chunks. Pieces may sit in different chunks.
- Make each piece distinctive (roughly 6+ words). A piece like `Associate of Arts` alone can match unrelated chunks.
- `…` (single Unicode ellipsis character) works the same as `...`.

Joining non-adjacent sentences **without** `...` silently breaks matching. Real example (amb-004):

```text
Page:   ...These amounts are updated each year.
        These amounts do not include tuition and transportation costs.      <- skipped
        All provinces and territories except Quebec ...

Wrong:  "... These amounts are updated each year. All provinces and territories except Quebec ..."
Right:  "... These amounts are updated each year. ... All provinces and territories except Quebec ..."
```

And a correct one (mh-003), where a whole section sits between the two pieces:

```text
"You can get faster processing if you are enrolling in a doctoral degree program, apply for a study permit
 from outside Canada, and submit your application online. ... Your application will not be processed faster
 if you are enrolling in a master’s degree program."
```

### 4. One gold passage per piece of evidence
If the answer needs evidence from two pages, write two `gold_passages` entries. Every gold passage
must be retrieved for the question to count as a hit (question rank = worst quote rank).

### 5. `url` and `content_hash`
- `url` must be the exact `url` from `manifest.jsonl`, **including the query string**
  (DLI detail pages differ only by `?q=...`). Copy it from the manifest, not the browser.
- `content_hash` must match the manifest row for that page. It identifies the exact page version
  and is authoritative if the URL is ambiguous. (Known issue: `sf-0004` has the URL without its `?q=` part;
  its `content_hash` points to the right page.)
- After a re-crawl, re-check every quote whose page's `content_hash` changed.

## After changing the eval set: re-run the reference configs

Every `retrieval_eval.py eval` run records a fingerprint of the questions it scored (`eval_set_hash`:
ids, questions and gold passages). Adding, removing or editing an item, including fixing one quote,
creates a new version, shown as `[v1]`, `[v2]`, ... next to each run in the runs table. Hit@5 from
different versions is computed on different questions and is **not** comparable:

- the runs table marks where the version changes and stars the best Hit@5 *within* each version;
- `compare` on two runs from different versions recomputes the metrics on their shared questions only and says so.

So after changing `eval_set.jsonl`, first re-run the reference configs on the new version (no LLM, embeddings
are cached, about a minute each), for example the MiniLM dedup baseline and the current best config, with
`--change "re-run <config> on eval set vN"`. New experiments then have a same-version row to compare against.
To try a candidate eval set without touching the real file: `eval --config ... --eval-file <path>`.

## Check your quotes

Run from the repo root after adding or editing items; it prints any quote that is not found in its page:

```bash
.venv/Scripts/python - <<'EOF'
import retrieval_eval as r
pages = {p["content_hash"]: r.norm_text(p["text"]) for p in r.load_pages(r.RUN_DIR)}
for it in r.load_eval_items():
    for g in it["gold_passages"]:
        page = pages.get(g["content_hash"])
        if page is None or any(f not in page for f in r.quote_fragments(g["quote"])):
            print(it["id"], "NOT FOUND:", g["quote"][:80])
EOF
```

A quote found in its page but never inside a single chunk (miss group A in `retrieval_eval.py`) is a
real chunking problem, not an eval set problem: leave the quote as it is.
