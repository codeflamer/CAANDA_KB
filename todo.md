# CanadaGuidance RAG: progress & todo

## Goals
- [ ] Build a credible eval harness for the IRCC RAG (retrieval-only + end-to-end), then deploy it on AWS
  - [x] Retrieval-only eval mode (free, targets the biggest problem: retrieval) — `retrieval_eval.py`
  - [ ] Deploy the RAG assistant with an automated eval harness on AWS (Bedrock, SageMaker Pipelines, Terraform)

## Next: make the harness credible
### Eval set
- [ ] Grow eval_set beyond 25 scored questions (5 per category is too few: one question = 0.2 of a category score)
- [ ] Human-verify every item, including the 17 LLM-drafted multi-hop items; record `verified_by`
- [ ] Held-out test split: tune on a dev split only, report final numbers on a test split never used for tuning
      (current scores are optimistic — the pipeline was tuned on the same 25 + 22 questions)
- [ ] Re-verify gold quotes automatically when a page's content_hash changes after a re-crawl; flag stale temporal items

### Judge validation
- [ ] Hand-label a sample of answers (correct / partial / incorrect, grounded) and measure agreement with the gpt-5 judge
- [ ] Judge consistency: judge the same answers several times; report how often the verdict flips
- [ ] Second judge from another model family; investigate disagreements
- [ ] Separate answer noise from judge noise (old pipeline moved 11–12 → 14 correct between two runs)

### Reproducibility
- [ ] Pin model snapshots (answer, judge, router/chain LLM) instead of aliases; record them in every run
- [ ] Cache answer + judge calls in run_eval.py like retrieval_eval.py does, so re-runs are free and identical
- [ ] Every run records git commit, config, eval_set hash and crawl snapshot (retrieval_eval lacks the commit,
      run_eval lacks the eval_set hash)
- [ ] One run-record format and one results layout for retrieval-only and end-to-end runs
- [ ] Document the methodology (metrics, judge prompts, scoring rules) in readme_eval.md

## Retrieval issues investigated
- [x] Broken by chunking: many gold quotes not found in the retrieved chunk (cut mid-passage / quote matching) —
      fixed quote matching ("..." fragments) and section expansion
- [x] Ranked too low — solved with dedup + better embedding model (bge-base) + reranker + expanded context
- [ ] Not found at all — still open: hybrid search (BM25 + dense), heading context, other embedding models
- [x] Multi-hop: one passage found, the other missing or ranked too low; want both ranked high —
      dedicated multi-hop eval (22 questions) + chain retrieval

## Done: retrieval-only eval (eval_set, 25 scored questions; Hit@5 = all gold passages in top 5)
| Step | Hit@5 | MRR | Run |
|---|---|---|---|
| Baseline: MiniLM, 180-word chunks, top 5 | 0.24 | 0.21 | 20261004-144051 |
| Fix quote matching (word tokens + "..." fragments) | 0.32 | 0.28 | 20261004-153258 |
| + cross-encoder rerank (ms-marco-MiniLM), pool 50 | 0.52 | 0.36 | 20261004-162443 |
| + dedup repeated chunks | 0.56 | 0.38 | 20261004-171121 |
| + rerank pool 100 | 0.60 | 0.39 | 20261004-171828 |
| + bge-base-en-v1.5 embeddings | 0.60 | 0.40 | 20261004-172841 |
| + split compound questions, round-robin fusion | 0.64 | 0.41 | 20261004-200956 |
| + drop vague sub-queries | 0.68 | 0.39 | 20261004-201547 |
| + section expansion (small-to-big, 450 words) | 0.72 | 0.50 | 20261005-024922 |
| + exact cosine search (removes run-to-run noise) | 0.72 | 0.50 | 20261010-192123 |
| + LLM router: chain retrieval for multi-hop only (FINAL) | 0.72 | 0.504 | 20261011-172721 |

Rejected: bge-reranker-base (no better than ms-marco); chain retrieval on every question (Hit@5 0.68, hurts
one-passage questions).

## Done: multi-hop eval (22 questions, `retrieval_eval.py multihop`; joint@5 = both passages in top 5)
- [x] Extended metrics (joint@5/10/50, Hit@k, recall, MRR first/joint); item set = draft_multihop_v0 only; mh-003 quote fixed
- Baseline 0.32 (20261011-124651) → **chain retrieval 0.59** (20261011-141955, 14 better / 3 worse)
- Rejected: LLM split v1 0.14 (long queries), v2 0.36 (drops the user's scenario sentence), LLM split + chain 0.36,
  RT-RAG / Reasoning in Trees 0.41 (same scenario loss, 239 LLM calls)
- Router + chain on the 22: 0.36 — router labels only 10/22 as multi-hop

## Done: end-to-end (30 eval_set questions, gpt-5 judge, `scripts/run_eval.py`)
- [x] Pipeline ported into `scripts/add_to_vdb.py` (`call_rag(..., pipeline="v2")`, now the default; `v0` = old)
- [x] `run_eval.py --pipeline`, fixed quote check, run ids can't collide
- Old pipeline 20261011-183408: 14 correct / 12 partial / 4 incorrect, quote 8/25
- New pipeline 20261011-183415: **20 / 9 / 1**, quote 18/25, retrieval failures 14 → 8
- Reports: retrieval conclusion https://claude.ai/artifact/1vAxUgbnTzJBGfaTj4Z1YW ,
  end-to-end https://claude.ai/artifact/QyQ2AP3vhGGjARapvJqi9W

## Next: retrieval
- [ ] Router v2: yes/no "does this need facts from 2+ topics?" instead of 6 categories (v1 calls 10/22 multi-hop "conditional")
- [ ] Hybrid BM25 + dense for passages never retrieved (mh-003 passage 1; BM25 found it at rank 31 vs dense 238)
- [ ] IRCC acronym glossary (PAL/TAL, PGWP, DLI, IME) for query expansion
- [ ] Ambiguous questions (Hit@5 0.6): amb-001 / amb-002 have one interpretation ranked 21–26
- [ ] cond-001 gold passage at rank 11
- [ ] Remaining chain misses on the multi-hop set: mh-003, 005, 011, 013, 016, 022

## Next: answer quality
- [ ] Prompt: don't over-generalise a rule (temp-005 stated a master's/PhD exemption as applying to all public DLIs)
- [ ] Prompt: unanswerable questions should redirect, not add unsupported specifics (unans-003)
- [ ] Crawler: no page has `date_modified`, so "prefer the more recent page" never applies — capture it
- [ ] Reading failures (retrieved but answer wrong): 4 in the v2 run — inspect

## Next: engineering / deploy
- [ ] Decide: remove or keep the rejected code paths in `retrieval_eval.py` (LLM split, RT-RAG; off by default)
- [ ] Measure latency + cost per question of v2 (router + chain = up to 2 extra LLM calls)
- [ ] App entry point (`main.py`) on the v2 pipeline
- [ ] AWS deployment (Bedrock, SageMaker Pipelines, Terraform) with the eval harness in the pipeline
