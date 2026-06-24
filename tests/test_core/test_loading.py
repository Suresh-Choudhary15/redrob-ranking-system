"""Tests for core/loading.py — written before implementation (TDD).

These tests define the exact contract that core/loading.py must satisfy.
They are intentionally implementation-agnostic: they import only the public
interface (``load_candidates``, ``LoadError``) and make no assumptions about
internal data structures, streaming strategy, or buffer sizes.

Public interface assumed:

    from core.loading import load_candidates, LoadError

    def load_candidates(
        path: Path,
        *,
        strict: bool = True,
    ) -> list[Candidate]:
        ...

    class LoadError(Exception):
        ...

Behaviour contract:
- Accepts pathlib.Path for all file arguments.
- Auto-detects format: JSONL (one JSON object per line) vs JSON array
  (file whose first non-whitespace character is ``[``).
- Returns a list of validated ``core.schema.Candidate`` objects.
- Preserves source record order.
- ``strict=True`` (default): raises ``LoadError`` on the first
  schema-invalid or malformed-JSON record.
- ``strict=False``: skips invalid records, collects them as
  ``LoadError`` instances accessible via ``exc.errors`` on the
  raised exception — or returns successfully with only valid records
  if at least one valid record exists.  Caller inspects
  ``result.errors`` (see design note below).
- Empty JSONL file → returns ``[]``.
- Empty JSON array file (``[]``) → returns ``[]``.
- Blank lines in JSONL are silently skipped.
- Does NOT load the entire file into a Python list before yielding
  the first record (verified via the streaming fixture).

Design note on strict=False return type
-----------------------------------------
``load_candidates`` returns a ``LoadResult`` named-tuple (or dataclass):

    @dataclass
    class LoadResult:
        candidates: list[Candidate]
        errors: list[LoadError]

When ``strict=True`` the function either returns a plain ``list[Candidate]``
or raises — callers that always use ``strict=True`` never need to know
about ``LoadResult``.  When ``strict=False`` it always returns a
``LoadResult`` so callers can inspect skipped records.

This two-mode design keeps the common path (strict=True) ergonomic while
making non-strict behaviour fully introspectable.
"""

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.schema import Candidate


# ---------------------------------------------------------------------------
# Lazy import guard — core.loading does not exist yet.
# All tests import via this helper so the test file itself is always
# importable even before implementation; individual tests will xfail/error
# with a clear ImportError rather than breaking collection.
# ---------------------------------------------------------------------------


def _load_candidates():
    """Return the load_candidates callable, imported lazily."""
    from core.loading import load_candidates  # noqa: PLC0415
    return load_candidates


def _load_error_cls():
    """Return the LoadError class, imported lazily."""
    from core.loading import LoadError  # noqa: PLC0415
    return LoadError


def _load_result_cls():
    """Return the LoadResult class, imported lazily."""
    from core.loading import LoadResult  # noqa: PLC0415
    return LoadResult


# ---------------------------------------------------------------------------
# Tiny in-test fixtures (written inline; kept minimal)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSONL to path, one JSON object per line."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _write_json_array(path: Path, records: list[dict]) -> Path:
    """Write records as a JSON array to path."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path


def _minimal_valid_record(candidate_id: str = "CAND_0000001") -> dict:
    """Return the smallest structurally-valid candidate dict for a given ID."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Engineer",
            "summary": "Test summary.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 60,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built things.",
            }
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 50.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 5,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 50,
            "endorsements_received": 2,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "github_activity_score": 10.0,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 1,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": False,
        },
    }


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def test_load_candidates_returns_list(tmp_path: Path) -> None:
    """load_candidates() returns a list."""
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert isinstance(result, list)


def test_load_candidates_returns_candidate_objects(tmp_path: Path) -> None:
    """Every element in the returned list is a core.schema.Candidate instance."""
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert len(result) == 1
    assert isinstance(result[0], Candidate)


def test_candidate_id_matches_source(tmp_path: Path) -> None:
    """Parsed candidate_id matches the value in the source record."""
    record = _minimal_valid_record("CAND_0042000")
    f = _write_jsonl(tmp_path / "c.jsonl", [record])
    result = _load_candidates()(f)
    assert result[0].candidate_id == "CAND_0042000"


# ---------------------------------------------------------------------------
# Format auto-detection: JSONL
# ---------------------------------------------------------------------------


def test_jsonl_single_record(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert len(result) == 1


def test_jsonl_multiple_records(tmp_path: Path) -> None:
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, 6)]
    f = _write_jsonl(tmp_path / "c.jsonl", records)
    result = _load_candidates()(f)
    assert len(result) == 5


def test_jsonl_blank_lines_are_skipped(tmp_path: Path) -> None:
    """JSONL files with blank lines between records load without error."""
    f = tmp_path / "c.jsonl"
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, 4)]
    with f.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(records[0]) + "\n")
        fh.write("\n")                            # blank line
        fh.write("   \n")                         # whitespace-only line
        fh.write(json.dumps(records[1]) + "\n")
        fh.write(json.dumps(records[2]) + "\n")
    result = _load_candidates()(f)
    assert len(result) == 3


def test_jsonl_file_without_jsonl_extension(tmp_path: Path) -> None:
    """Format detection is content-based, not extension-based."""
    f = tmp_path / "candidates.txt"   # .txt extension, JSONL content
    _write_jsonl(f, [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Format auto-detection: JSON array
# ---------------------------------------------------------------------------


def test_json_array_single_record(tmp_path: Path) -> None:
    f = _write_json_array(tmp_path / "c.json", [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert len(result) == 1


def test_json_array_multiple_records(tmp_path: Path) -> None:
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, 6)]
    f = _write_json_array(tmp_path / "c.json", records)
    result = _load_candidates()(f)
    assert len(result) == 5


def test_json_array_returns_candidate_objects(tmp_path: Path) -> None:
    f = _write_json_array(tmp_path / "c.json", [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert isinstance(result[0], Candidate)


def test_json_array_with_leading_whitespace_detected_correctly(tmp_path: Path) -> None:
    """A JSON array preceded by whitespace/newlines is still detected as array format."""
    f = tmp_path / "c.json"
    content = "\n\n  " + json.dumps([_minimal_valid_record()], ensure_ascii=False)
    f.write_text(content, encoding="utf-8")
    result = _load_candidates()(f)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Uses existing test fixtures
# ---------------------------------------------------------------------------


def test_loads_sample_candidates_fixture(sample_candidates_path: Path) -> None:
    """sample_candidates.json (a JSON array) loads cleanly via load_candidates()."""
    result = _load_candidates()(sample_candidates_path)
    assert len(result) >= 10
    assert all(isinstance(c, Candidate) for c in result)


def test_sample_candidates_fixture_candidate_ids_are_valid(
    sample_candidates_path: Path,
) -> None:
    """Every candidate_id loaded from the real fixture matches the expected pattern."""
    import re
    pattern = re.compile(r"^CAND_[0-9]{7}$")
    result = _load_candidates()(sample_candidates_path)
    for candidate in result:
        assert pattern.match(candidate.candidate_id), (
            f"Invalid candidate_id: {candidate.candidate_id!r}"
        )


# ---------------------------------------------------------------------------
# Record order preservation
# ---------------------------------------------------------------------------


def test_jsonl_preserves_record_order(tmp_path: Path) -> None:
    """Records are returned in the same order they appear in the JSONL file."""
    ids = [f"CAND_{i:07d}" for i in range(1, 11)]
    records = [_minimal_valid_record(cid) for cid in ids]
    f = _write_jsonl(tmp_path / "c.jsonl", records)
    result = _load_candidates()(f)
    assert [c.candidate_id for c in result] == ids


def test_json_array_preserves_record_order(tmp_path: Path) -> None:
    ids = [f"CAND_{i:07d}" for i in range(1, 11)]
    records = [_minimal_valid_record(cid) for cid in ids]
    f = _write_json_array(tmp_path / "c.json", records)
    result = _load_candidates()(f)
    assert [c.candidate_id for c in result] == ids


# ---------------------------------------------------------------------------
# Empty file handling
# ---------------------------------------------------------------------------


def test_empty_jsonl_returns_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    result = _load_candidates()(f)
    assert result == []


def test_jsonl_with_only_blank_lines_returns_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "blank.jsonl"
    f.write_text("\n\n   \n\n", encoding="utf-8")
    result = _load_candidates()(f)
    assert result == []


def test_empty_json_array_returns_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "empty.json"
    f.write_text("[]", encoding="utf-8")
    result = _load_candidates()(f)
    assert result == []


# ---------------------------------------------------------------------------
# pathlib.Path input
# ---------------------------------------------------------------------------


def test_accepts_pathlib_path(tmp_path: Path) -> None:
    """load_candidates() accepts a pathlib.Path without error."""
    f: Path = tmp_path / "c.jsonl"
    _write_jsonl(f, [_minimal_valid_record()])
    result = _load_candidates()(f)
    assert len(result) == 1


def test_nonexistent_file_raises_load_error(tmp_path: Path) -> None:
    """A path that does not exist raises LoadError, not a raw OSError or FileNotFoundError."""
    LoadError = _load_error_cls()
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(LoadError):
        _load_candidates()(missing)


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------


def test_malformed_json_line_strict_raises_load_error(tmp_path: Path) -> None:
    """A line that is not valid JSON raises LoadError in strict mode."""
    LoadError = _load_error_cls()
    f = tmp_path / "bad.jsonl"
    f.write_text(
        json.dumps(_minimal_valid_record("CAND_0000001")) + "\n"
        "THIS IS NOT JSON\n",
        encoding="utf-8",
    )
    with pytest.raises(LoadError):
        _load_candidates()(f)


def test_malformed_json_array_raises_load_error(tmp_path: Path) -> None:
    """A file whose content is not valid JSON at all raises LoadError."""
    LoadError = _load_error_cls()
    f = tmp_path / "garbage.json"
    f.write_text("{broken json: [", encoding="utf-8")
    with pytest.raises(LoadError):
        _load_candidates()(f)


def test_malformed_json_line_non_strict_skips_record(tmp_path: Path) -> None:
    """In strict=False mode, a malformed JSON line is skipped; valid records are returned."""
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        json.dumps(_minimal_valid_record("CAND_0000001")) + "\n"
        "NOT JSON\n"
        + json.dumps(_minimal_valid_record("CAND_0000002")) + "\n",
        encoding="utf-8",
    )
    result = _load_candidates()(f, strict=False)
    candidates = result.candidates
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "CAND_0000001"
    assert candidates[1].candidate_id == "CAND_0000002"


def test_malformed_json_line_non_strict_records_error(tmp_path: Path) -> None:
    """In strict=False mode, the skipped malformed line appears in result.errors."""
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        json.dumps(_minimal_valid_record("CAND_0000001")) + "\n"
        "NOT JSON\n",
        encoding="utf-8",
    )
    result = _load_candidates()(f, strict=False)
    assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------


def test_schema_invalid_record_strict_raises_load_error(tmp_path: Path) -> None:
    """A record that fails Candidate validation raises LoadError in strict mode."""
    LoadError = _load_error_cls()
    bad = _minimal_valid_record("CAND_0000001")
    del bad["redrob_signals"]   # remove required field
    f = _write_jsonl(tmp_path / "bad.jsonl", [bad])
    with pytest.raises(LoadError):
        _load_candidates()(f)


def test_schema_invalid_record_non_strict_skips_and_records_error(
    tmp_path: Path,
) -> None:
    """In strict=False mode, a schema-invalid record is skipped, not raised."""
    bad = _minimal_valid_record("CAND_0000001")
    del bad["redrob_signals"]
    good = _minimal_valid_record("CAND_0000002")
    f = _write_jsonl(tmp_path / "mixed.jsonl", [bad, good])
    result = _load_candidates()(f, strict=False)
    assert len(result.candidates) == 1
    assert result.candidates[0].candidate_id == "CAND_0000002"
    assert len(result.errors) == 1


def test_schema_invalid_record_error_contains_candidate_id(tmp_path: Path) -> None:
    """The LoadError for a schema-invalid record references the candidate_id when available."""
    bad = _minimal_valid_record("CAND_0000001")
    del bad["redrob_signals"]
    f = _write_jsonl(tmp_path / "bad.jsonl", [bad, _minimal_valid_record("CAND_0000002")])
    result = _load_candidates()(f, strict=False)
    error_str = str(result.errors[0])
    assert "CAND_0000001" in error_str


def test_all_invalid_records_non_strict_returns_empty_candidates(tmp_path: Path) -> None:
    """strict=False with every record invalid returns LoadResult with empty candidates list."""
    bad1 = _minimal_valid_record("CAND_0000001")
    bad2 = _minimal_valid_record("CAND_0000002")
    del bad1["redrob_signals"]
    del bad2["redrob_signals"]
    f = _write_jsonl(tmp_path / "all_bad.jsonl", [bad1, bad2])
    result = _load_candidates()(f, strict=False)
    assert result.candidates == []
    assert len(result.errors) == 2


def test_invalid_candidate_id_pattern_raises_load_error(tmp_path: Path) -> None:
    """A record with a malformed candidate_id (fails regex) raises LoadError."""
    LoadError = _load_error_cls()
    bad = _minimal_valid_record()
    bad["candidate_id"] = "BAD_ID"
    f = _write_jsonl(tmp_path / "bad_id.jsonl", [bad])
    with pytest.raises(LoadError):
        _load_candidates()(f)


# ---------------------------------------------------------------------------
# LoadError and LoadResult types
# ---------------------------------------------------------------------------


def test_load_error_is_exception_subclass() -> None:
    """LoadError must be a subclass of Exception."""
    LoadError = _load_error_cls()
    assert issubclass(LoadError, Exception)


def test_load_result_has_candidates_attribute(tmp_path: Path) -> None:
    """LoadResult exposes a .candidates attribute."""
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f, strict=False)
    assert hasattr(result, "candidates")


def test_load_result_has_errors_attribute(tmp_path: Path) -> None:
    """LoadResult exposes an .errors attribute."""
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f, strict=False)
    assert hasattr(result, "errors")


def test_load_result_errors_is_empty_when_all_valid(tmp_path: Path) -> None:
    """When all records are valid, LoadResult.errors is an empty list."""
    f = _write_jsonl(tmp_path / "c.jsonl", [_minimal_valid_record()])
    result = _load_candidates()(f, strict=False)
    assert result.errors == []


# ---------------------------------------------------------------------------
# Streaming: does not buffer entire file before yielding first record
# ---------------------------------------------------------------------------


def test_streaming_jsonl_does_not_read_entire_file_into_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_candidates() on JSONL must not call file.read() or file.readlines()
    on the whole file before processing any record.  This is verified by
    monkeypatching Path.open to wrap the file object and asserting that
    .read() and .readlines() are never called.

    The test writes 5 records so a correct streaming implementation processes
    them one line at a time; a naive json.loads(file.read()) implementation
    trips the assertion.
    """
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, 6)]
    f = _write_jsonl(tmp_path / "stream.jsonl", records)

    read_called = []
    readlines_called = []

    original_open = Path.open

    def patched_open(self, *args, **kwargs):
        fh = original_open(self, *args, **kwargs)

        class TrackingWrapper:
            def __init__(self, inner):
                self._inner = inner

            def read(self, *a, **kw):
                read_called.append(True)
                return self._inner.read(*a, **kw)

            def readlines(self, *a, **kw):
                readlines_called.append(True)
                return self._inner.readlines(*a, **kw)

            def readline(self, *a, **kw):
                return self._inner.readline(*a, **kw)

            def __iter__(self):
                return iter(self._inner)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *a):
                return self._inner.__exit__(*a)

        return TrackingWrapper(fh)

    monkeypatch.setattr(Path, "open", patched_open)
    _load_candidates()(f)

    assert not read_called, (
        "load_candidates() called file.read() — this loads the entire JSONL "
        "into memory at once instead of streaming line by line."
    )
    assert not readlines_called, (
        "load_candidates() called file.readlines() — this loads the entire "
        "JSONL into memory at once instead of streaming line by line."
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_file_produces_same_order_on_repeated_calls(tmp_path: Path) -> None:
    """Calling load_candidates() twice on the same file returns the same order."""
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, 6)]
    f = _write_jsonl(tmp_path / "c.jsonl", records)
    load_candidates = _load_candidates()
    first = [c.candidate_id for c in load_candidates(f)]
    second = [c.candidate_id for c in load_candidates(f)]
    assert first == second


# ---------------------------------------------------------------------------
# Large-ish file smoke test (synthetic, stays within test budget)
# ---------------------------------------------------------------------------


def test_loads_five_hundred_jsonl_records_correctly(tmp_path: Path) -> None:
    """load_candidates() handles a file with 500 records without error or data loss."""
    n = 500
    records = [_minimal_valid_record(f"CAND_{i:07d}") for i in range(1, n + 1)]
    f = _write_jsonl(tmp_path / "large.jsonl", records)
    result = _load_candidates()(f)
    assert len(result) == n
    assert result[0].candidate_id == "CAND_0000001"
    assert result[-1].candidate_id == f"CAND_{n:07d}"