# scripts/extract_fixtures.py — run once, not part of the graded pipeline
import json
import random
from pathlib import Path

SOURCE = Path("docs/candidates.jsonl")          # change to json.load(f) if yours is a JSON array
OUTPUT = Path("tests/fixtures/sample_candidates.json")
POOL_SIZE = 60     # sample a larger pool than you need
KEEP = 12          # then manually trim/curate down to this many
SEED = 42          # fixed seed — reruns produce the same starting pool


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    records = load_jsonl(SOURCE)
    random.seed(SEED)
    pool = random.sample(records, POOL_SIZE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote a {len(pool)}-record pool to {OUTPUT} — now manually trim to {KEEP} for diversity.")


if __name__ == "__main__":
    main()