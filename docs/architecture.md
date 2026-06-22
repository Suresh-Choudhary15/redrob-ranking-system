# Architecture Notes

## Approach

Two-track: CLI pipeline (graded) + optional platform demo.

## Retrieval

BM25 + dense embeddings (BGE-small), fused via RRF.

## Ranking

Rule-based recruiter-intelligence scorer on top of retrieval.

## Key decisions

- No LLM calls at ranking time (5-min constraint)
- Behavioral signals as multiplier, not additive
- Honeypot gate before final top-100

## TODO

- [ ] loading.py
- [ ] cleaning.py
- [ ] feature modules
