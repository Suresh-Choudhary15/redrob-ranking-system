"""Tests for core/schema.py — Candidate and related Pydantic v2 models.

Positive cases confirm every real-slice and synthetic-valid record parses
successfully and round-trips without loss. Negative cases confirm each
class of malformed input — a bad candidate_id, an out-of-range numeric
field, an invalid enum value, a missing required field — raises a
pydantic.ValidationError, tested one violation at a time so a failure
points at exactly which constraint broke.

Depends on nothing beyond core.schema and the fixtures defined in
tests/conftest.py.
"""

import copy

import pytest
from pydantic import ValidationError

from core.schema import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_data_loss(original: dict, dumped: dict) -> None:
    """Assert every key/value present in `original` survives unchanged in `dumped`.

    Recurses into nested dicts and lists of dicts. Keys present in `dumped`
    but absent from `original` — pydantic-filled defaults for omitted
    optional fields, such as an absent `certifications` list defaulting to
    `[]` — are not data loss and are deliberately ignored here.
    """
    for key, original_value in original.items():
        assert key in dumped, f"key {key!r} missing from round-tripped output"
        dumped_value = dumped[key]
        if isinstance(original_value, dict):
            assert isinstance(dumped_value, dict)
            _assert_no_data_loss(original_value, dumped_value)
        elif isinstance(original_value, list):
            assert isinstance(dumped_value, list)
            assert len(original_value) == len(dumped_value)
            for original_item, dumped_item in zip(original_value, dumped_value):
                if isinstance(original_item, dict):
                    _assert_no_data_loss(original_item, dumped_item)
                else:
                    assert original_item == dumped_item
        else:
            assert original_value == dumped_value, (
                f"value changed for key {key!r}: {original_value!r} -> {dumped_value!r}"
            )


def _mutate(record: dict, field_path: tuple, value: object) -> dict:
    """Deep-copy `record` and set `value` at the given path of dict/list keys."""
    mutated = copy.deepcopy(record)
    target = mutated
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    return mutated


# ---------------------------------------------------------------------------
# Positive validation: real sample records
# ---------------------------------------------------------------------------


def test_sample_candidates_fixture_has_expected_size(sample_candidates_data: list[dict]) -> None:
    """Sanity check on the fixture file itself, independent of schema validation."""
    assert 10 <= len(sample_candidates_data) <= 15


def test_every_sample_candidate_validates(sample_candidates_data: list[dict]) -> None:
    """Every record in the real-data fixture slice parses without error.

    Iterates rather than using a hardcoded parametrize count, so the test
    stays correct if the fixture file's size ever changes; a failure
    reports exactly which candidate_id broke validation.
    """
    for record in sample_candidates_data:
        try:
            Candidate.model_validate(record)
        except ValidationError as exc:
            pytest.fail(f"{record.get('candidate_id', '<unknown>')} failed validation: {exc}")


# ---------------------------------------------------------------------------
# Positive validation: synthetic edge cases that are schema-valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_name",
    [
        "salary_inversion",
        "all_optional_fields_omitted",
        "negative_sentinel_values",
        "minimal_valid",
    ],
)
def test_valid_synthetic_edge_cases_validate(
    synthetic_edge_cases_data: dict[str, dict], case_name: str
) -> None:
    """Every synthetic case except the deliberately-invalid one parses successfully.

    Covers `salary_inversion` (schema.py is deliberately permissive about
    min > max — that's a cleaning concern, not a validation concern) and
    `negative_sentinel_values` (-1 is a documented, valid in-range value
    for github_activity_score and offer_acceptance_rate, not something
    schema.py special-cases or rejects).
    """
    record = synthetic_edge_cases_data[case_name]
    Candidate.model_validate(record)


def test_omitted_optional_fields_fill_documented_defaults(
    synthetic_edge_cases_data: dict[str, dict],
) -> None:
    """Optional fields omitted on input are filled with their documented defaults,
    not silently dropped, corrupted, or left in some other unexpected state.
    """
    record = synthetic_edge_cases_data["all_optional_fields_omitted"]
    candidate = Candidate.model_validate(record)

    assert candidate.certifications == []
    assert candidate.languages == []
    assert candidate.education[0].grade is None
    assert candidate.education[0].tier is None
    assert candidate.skills[0].duration_months is None


# ---------------------------------------------------------------------------
# Round-trip: parse, dump, confirm no data loss
# ---------------------------------------------------------------------------


def test_every_sample_candidate_round_trips_without_loss(
    sample_candidates_data: list[dict],
) -> None:
    """Every field present in each real record survives a parse-then-dump round trip."""
    for record in sample_candidates_data:
        candidate = Candidate.model_validate(record)
        dumped = candidate.model_dump(mode="json")
        try:
            _assert_no_data_loss(record, dumped)
        except AssertionError as exc:
            pytest.fail(f"{record['candidate_id']}: {exc}")


@pytest.mark.parametrize(
    "case_name",
    ["salary_inversion", "negative_sentinel_values", "minimal_valid"],
)
def test_synthetic_edge_case_round_trips_without_loss(
    synthetic_edge_cases_data: dict[str, dict], case_name: str
) -> None:
    """Fields explicitly present in a synthetic record survive a round trip too."""
    record = synthetic_edge_cases_data[case_name]
    candidate = Candidate.model_validate(record)
    dumped = candidate.model_dump(mode="json")
    _assert_no_data_loss(record, dumped)


def test_partially_provided_record_round_trips_provided_fields(
    synthetic_edge_cases_data: dict[str, dict],
) -> None:
    """Even when optional fields are omitted, every field that *was* provided
    must still be preserved exactly through a round trip — distinct from
    the "defaults get filled" check above, which only inspects the parsed
    model, not a re-serialized round trip.
    """
    record = synthetic_edge_cases_data["all_optional_fields_omitted"]
    candidate = Candidate.model_validate(record)
    dumped = candidate.model_dump(mode="json")
    _assert_no_data_loss(record, dumped)


# ---------------------------------------------------------------------------
# Negative: invalid candidate_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_candidate_id",
    [
        "CAND_123",            # too few digits
        "CAND_00000001",       # too many digits
        "CANDIDATE_0000001",   # wrong prefix
        "cand_0000001",        # wrong case
        "CAND-0000001",        # wrong separator
        "0000001",             # missing prefix entirely
        "",                    # empty string
    ],
)
def test_invalid_candidate_id_raises(
    sample_candidates_data: list[dict], bad_candidate_id: str
) -> None:
    """A candidate_id that doesn't match CAND_[0-9]{7} is rejected."""
    record = _mutate(sample_candidates_data[0], ("candidate_id",), bad_candidate_id)

    with pytest.raises(ValidationError) as exc_info:
        Candidate.model_validate(record)

    assert "candidate_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Negative: invalid enum values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_path, bad_value",
    [
        (("profile", "current_company_size"), "tiny"),
        (("skills", 0, "proficiency"), "expert-ish"),
        (("redrob_signals", "preferred_work_mode"), "occasionally_remote"),
        (("education", 0, "tier"), "tier_99"),
        (("languages", 0, "proficiency"), "fluent"),
    ],
)
def test_invalid_enum_value_raises(
    sample_candidates_data: list[dict], field_path: tuple, bad_value: object
) -> None:
    """A value outside an enum's allowed set is rejected, for each enum field."""
    record = _mutate(sample_candidates_data[0], field_path, bad_value)

    with pytest.raises(ValidationError):
        Candidate.model_validate(record)


# ---------------------------------------------------------------------------
# Negative: missing required fields
# ---------------------------------------------------------------------------


def test_missing_redrob_signals_raises(
    synthetic_edge_cases_data: dict[str, dict],
) -> None:
    """The dedicated missing_required_field fixture (no redrob_signals key) is rejected."""
    record = synthetic_edge_cases_data["missing_required_field"]

    with pytest.raises(ValidationError) as exc_info:
        Candidate.model_validate(record)

    assert "redrob_signals" in str(exc_info.value)


@pytest.mark.parametrize(
    "missing_key",
    ["candidate_id", "profile", "career_history", "education", "skills", "redrob_signals"],
)
def test_each_required_top_level_field_is_enforced(
    sample_candidates_data: list[dict], missing_key: str
) -> None:
    """Every top-level required key, removed one at a time, is individually enforced."""
    record = copy.deepcopy(sample_candidates_data[0])
    del record[missing_key]

    with pytest.raises(ValidationError) as exc_info:
        Candidate.model_validate(record)

    assert missing_key in str(exc_info.value)


def test_missing_required_nested_field_raises(sample_candidates_data: list[dict]) -> None:
    """A required field missing from a nested object, not just the top level, is enforced."""
    record = copy.deepcopy(sample_candidates_data[0])
    del record["redrob_signals"]["github_activity_score"]

    with pytest.raises(ValidationError) as exc_info:
        Candidate.model_validate(record)

    assert "github_activity_score" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Negative: out-of-range numeric fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_path, bad_value",
    [
        (("profile", "years_of_experience"), 51),            # max is 50
        (("profile", "years_of_experience"), -1),             # min is 0
        (("redrob_signals", "github_activity_score"), -2),    # min is -1
        (("redrob_signals", "github_activity_score"), 101),   # max is 100
        (("redrob_signals", "offer_acceptance_rate"), -2),    # min is -1
        (("redrob_signals", "notice_period_days"), 181),      # max is 180
    ],
)
def test_out_of_range_numeric_value_raises(
    sample_candidates_data: list[dict], field_path: tuple, bad_value: object
) -> None:
    """A numeric value outside its documented range is rejected."""
    record = _mutate(sample_candidates_data[0], field_path, bad_value)

    with pytest.raises(ValidationError):
        Candidate.model_validate(record)