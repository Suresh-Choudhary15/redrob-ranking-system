"""Session-scoped pytest fixtures for the core test suite.

Loads both JSON fixture files (the real-data slice and the hand-authored
synthetic edge cases) exactly once per test session and exposes them in
two forms: as raw, un-validated Python data, for tests that perform
validation themselves, and as file paths, for tests that exercise
core/loading.py's own file-reading behavior end to end.

Fixtures intentionally return raw dicts, not core.schema.Candidate
instances. Pre-validating here would mean schema- and loading-layer tests
no longer actually exercise validation — they'd just be re-using a result
someone else already validated.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_candidates_path() -> Path:
    """Path to the real-data fixture slice: tests/fixtures/sample_candidates.json."""
    return FIXTURES_DIR / "sample_candidates.json"


@pytest.fixture(scope="session")
def synthetic_edge_cases_path() -> Path:
    """Path to the hand-authored edge cases: tests/fixtures/synthetic_edge_cases.json."""
    return FIXTURES_DIR / "synthetic_edge_cases.json"


@pytest.fixture(scope="session")
def sample_candidates_data(sample_candidates_path: Path) -> list[dict]:
    """Raw, un-validated records from sample_candidates.json, loaded once per session.

    A list of plain dicts mirroring the real dataset's shape, copied
    verbatim from the project's sample data. Every record here is expected
    to be schema-valid; confirming that is test_schema.py's job, not this
    fixture's.
    """
    with sample_candidates_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def synthetic_edge_cases_data(synthetic_edge_cases_path: Path) -> dict[str, dict]:
    """Raw, un-validated edge-case records from synthetic_edge_cases.json, loaded once per session.

    Keyed by scenario name ("salary_inversion", "all_optional_fields_omitted",
    "negative_sentinel_values", "minimal_valid", "missing_required_field")
    rather than positional, so tests reference a specific case unambiguously
    and survive future reordering of the fixture file. Not every value here
    is expected to be schema-valid: "missing_required_field" is deliberately
    invalid and is used only by core/loading.py tests, never asserted as a
    successful core.schema.Candidate validation.
    """
    with synthetic_edge_cases_path.open("r", encoding="utf-8") as f:
        return json.load(f)