# Architecture Notes

## What I'm building

A candidate ranking system for the Redrob hackathon.
Input: candidates.jsonl (100K candidates) + job_description.txt
Output: submission.csv with top 100 candidates ranked.

## High level approach

Two stages:

**Stage 1 - Retrieval**
Use embeddings (BGE-small model) + BM25 to find the top ~3000 candidates
out of 100K. Combine both using RRF (reciprocal rank fusion).
This is so I dont miss good candidates who dont use the exact same words
as the JD.

**Stage 2 - Ranking**
Score the shortlist using a rule-based recruiter intelligence scorer.
Not just keyword matching - actually look at:

- career history (what companies, product vs IT services)
- behavioral signals (when was last active, do they respond to recruiters)
- skill fit (weighted by verified assessment scores where available)
- domain specificity (did they actually build search/ranking/recsys systems)
- honeypot detection (internally inconsistent profiles)

Final score = weighted base fit × multiplicative modifiers

## Key decisions

**Why not just use embeddings?**
The JD explicitly says keyword matching is a trap. A marketing manager with
"RAG" and "Pinecone" in their skills list should not rank above someone
who built a recommendation system but didn't use those exact words.
Embeddings help with retrieval. The actual ranking comes from reading
what the person actually did in their career.

**Why no LLM calls at ranking time?**
Competition rules: CPU only, no network, 5 min budget.
All heavy work happens in the precompute step (embeddings, BM25 index,
feature engineering). The ranking step just loads precomputed artifacts
and does fast arithmetic.

**Why multiplicative modifiers for disqualifiers?**
A candidate who hasn't logged in for 6 months shouldn't rank in the top 10
even if their profile looks perfect. But high engagement shouldn't make a
bad-fit candidate look good either. Multiplicative keeps the structure honest.

## Repo structure

```
core/          - shared scoring engine (used by CLI and platform)
precompute/    - offline steps (embeddings, BM25, features)
rank.py        - the graded single command
platform/      - FastAPI + Postgres + Streamlit demo (not in graded path)
evaluation/    - proxy eval against small hand-labeled set
tests/         - pytest suite
```

## Current status

- [x] schema.py - candidate data models
- [x] config.py - settings
- [x] loading.py - JSONL + JSON array loading
- [x] cleaning.py - salary inversion fix only
- [ ] features/ - behavioral, honeypot, skill_fit, domain_specificity, product_vs_services
- [ ] scoring.py
- [ ] embeddings.py + bm25.py + retrieval.py
- [ ] rank.py (the actual graded command)
- [ ] platform/ (demo)

## Things I'm still figuring out

- Exact weight values for the scoring formula. Starting with principled guesses
  from reading the JD carefully, then checking against a small hand-labeled set.
- Whether career_pattern (title chasing detection) is worth the time given
  deadline is June 28. Probably skip if it takes more than half a day.
