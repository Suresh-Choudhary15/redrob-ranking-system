"""Tests for core/cleaning.py — written before implementation (TDD).

Defines the exact contract that core/cleaning.py must satisfy.

Public interface assumed:

    from core.cleaning import (
        clean_candidates,
        clean_candidate,
        fix_salary_inversion,
        CleaningIssue,
        CleaningResult,
        CleaningSummary,
    )

Behaviour contract
------------------
- ``fix_salary_inversion(candidate)`` returns ``(Candidate, CleaningIssue | None)``.
  When ``min > max``, values are swapped and a ``CleaningIssue`` is returned.
  When ``min <= max``, the candidate is returned unchanged and ``None`` is returned.

- ``clean_candidate(candidate)`` returns ``(Candidate, list[CleaningIssue])``.
  Applies all individual fix functions; currently only salary inversion.

- ``clean_candidates(candidates)`` returns a ``CleaningResult``.
  Iterates the pool, accumulates issues, never drops a candidate.

Hard guardrails (the non-negotiable part)
------------------------------------------
- ``-1`` sentinels on ``github_activity_score`` and ``offer_acceptance_rate``
  must never be modified — they are missing-data sentinels, not error values,
  and the honeypot detector depends on them reaching Phase 2 unmodified.
- ``career_history`` content (dates, durations, descriptions) must never be
  modified — tenure inconsistency is honeypot-detector evidence.
- ``skills[].proficiency`` and ``skills[].duration_months`` must never be
  modified — expert-proficiency-with-zero-duration is honeypot-detector
  evidence.
- No candidate is ever removed from the pool — input length always equals
  output length.
- In-place mutation of ``Candidate`` objects is forbidden; cleaning must
  return new objects (verified by identity check).
"""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.schema import Candidate


# ---------------------------------------------------------------------------
# Lazy import helpers — core.cleaning does not exist yet.
# ---------------------------------------------------------------------------


def _imports():
    from core.cleaning import (  # noqa: PLC0415
        CleaningIssue,
        CleaningResult,
        CleaningSummary,
        clean_candidate,
        clean_candidates,
        fix_salary_inversion,
    )
    return (
        CleaningIssue,
        CleaningResult,
        CleaningSummary,
        clean_candidate,
        clean_candidates,
        fix_salary_inversion,
    )


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    candidate_id: str = "CAND_0000001",
    salary_min: float = 15.0,
    salary_max: float = 30.0,
    github_activity_score: float = 25.0,
    offer_acceptance_rate: float = 0.5,
    career_duration_months: int = 24,
    skill_proficiency: str = "advanced",
    skill_duration_months: int = 12,
) -> Candidate:
    """Return a minimal but fully valid Candidate with configurable fields."""
    data = {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Candidate",
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
                "duration_months": career_duration_months,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built systems.",
            }
        ],
        "education": [],
        "skills": [
            {
                "name": "Python",
                "proficiency": skill_proficiency,
                "endorsements": 10,
                "duration_months": skill_duration_months,
            }
        ],
        "redrob_signals": {
            "profile_completeness_score": 70.0,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 1,
            "recruiter_response_rate": 0.6,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 5,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {
                "min": salary_min,
                "max": salary_max,
            },
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "github_activity_score": github_activity_score,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": offer_acceptance_rate,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": False,
        },
    }
    return Candidate.model_validate(data)


def _load_sample_candidates(sample_candidates_path: Path) -> list[Candidate]:
    """Load and validate the real-data fixture slice."""
    with sample_candidates_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Candidate.model_validate(r) for r in raw]


# ---------------------------------------------------------------------------
# CleaningIssue type
# ---------------------------------------------------------------------------


def test_cleaning_issue_is_instantiable() -> None:
    CleaningIssue, *_ = _imports()
    issue = CleaningIssue(
        candidate_id="CAND_0000001",
        field="expected_salary_range_inr_lpa",
        original_value={"min": 35.0, "max": 18.0},
        corrected_value={"min": 18.0, "max": 35.0},
        description="Salary min > max; swapped.",
    )
    assert issue.candidate_id == "CAND_0000001"
    assert issue.field == "expected_salary_range_inr_lpa"


def test_cleaning_issue_has_required_fields() -> None:
    CleaningIssue, *_ = _imports()
    issue = CleaningIssue(
        candidate_id="CAND_0000001",
        field="expected_salary_range_inr_lpa",
        original_value={"min": 35.0, "max": 18.0},
        corrected_value={"min": 18.0, "max": 35.0},
        description="test",
    )
    assert hasattr(issue, "candidate_id")
    assert hasattr(issue, "field")
    assert hasattr(issue, "original_value")
    assert hasattr(issue, "corrected_value")
    assert hasattr(issue, "description")


# ---------------------------------------------------------------------------
# CleaningSummary type
# ---------------------------------------------------------------------------


def test_cleaning_summary_is_instantiable() -> None:
    _, _, CleaningSummary, *_ = _imports()
    summary = CleaningSummary(
        total_candidates=10,
        candidates_modified=2,
        issues=[],
    )
    assert summary.total_candidates == 10
    assert summary.candidates_modified == 2
    assert summary.issues == []


def test_cleaning_summary_has_required_fields() -> None:
    _, _, CleaningSummary, *_ = _imports()
    summary = CleaningSummary(total_candidates=5, candidates_modified=0, issues=[])
    assert hasattr(summary, "total_candidates")
    assert hasattr(summary, "candidates_modified")
    assert hasattr(summary, "issues")


# ---------------------------------------------------------------------------
# CleaningResult type
# ---------------------------------------------------------------------------


def test_cleaning_result_is_instantiable() -> None:
    _, CleaningResult, CleaningSummary, *_ = _imports()
    result = CleaningResult(
        candidates=[],
        summary=CleaningSummary(total_candidates=0, candidates_modified=0, issues=[]),
    )
    assert result.candidates == []


def test_cleaning_result_has_required_fields() -> None:
    _, CleaningResult, CleaningSummary, *_ = _imports()
    result = CleaningResult(
        candidates=[],
        summary=CleaningSummary(total_candidates=0, candidates_modified=0, issues=[]),
    )
    assert hasattr(result, "candidates")
    assert hasattr(result, "summary")


# ---------------------------------------------------------------------------
# fix_salary_inversion — no inversion cases
# ---------------------------------------------------------------------------


def test_no_inversion_returns_none_issue() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=15.0, salary_max=30.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue is None


def test_no_inversion_returns_same_values() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=15.0, salary_max=30.0)
    corrected, _ = fix_salary_inversion(candidate)
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.min == 15.0
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.max == 30.0


def test_equal_min_max_returns_none_issue() -> None:
    """min == max is valid — a single-point salary expectation — not an inversion."""
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=20.0, salary_max=20.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue is None


def test_both_zero_returns_none_issue() -> None:
    """min == max == 0 is valid (minimal_valid fixture pattern), not an inversion."""
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=0.0, salary_max=0.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue is None


# ---------------------------------------------------------------------------
# fix_salary_inversion — inversion cases
# ---------------------------------------------------------------------------


def test_inversion_swaps_min_and_max() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    corrected, issue = fix_salary_inversion(candidate)
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.min == 18.0
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.max == 35.0


def test_inversion_returns_cleaning_issue() -> None:
    CleaningIssue, _, _, _, _, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue is not None
    assert isinstance(issue, CleaningIssue)


def test_inversion_issue_field_name() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue.field == "expected_salary_range_inr_lpa"


def test_inversion_issue_candidate_id_matches() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate("CAND_0042000", salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue.candidate_id == "CAND_0042000"


def test_inversion_issue_records_original_values() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue.original_value["min"] == 35.0
    assert issue.original_value["max"] == 18.0


def test_inversion_issue_records_corrected_values() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert issue.corrected_value["min"] == 18.0
    assert issue.corrected_value["max"] == 35.0


def test_inversion_issue_description_is_non_empty_string() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issue = fix_salary_inversion(candidate)
    assert isinstance(issue.description, str)
    assert len(issue.description) > 0


def test_inversion_corrected_candidate_is_valid_candidate() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    corrected, _ = fix_salary_inversion(candidate)
    assert isinstance(corrected, Candidate)


def test_inversion_candidate_id_preserved_after_fix() -> None:
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate("CAND_0042000", salary_min=35.0, salary_max=18.0)
    corrected, _ = fix_salary_inversion(candidate)
    assert corrected.candidate_id == "CAND_0042000"


# ---------------------------------------------------------------------------
# Immutability: cleaning must not mutate the original object
# ---------------------------------------------------------------------------


def test_fix_salary_inversion_does_not_mutate_original() -> None:
    """fix_salary_inversion must return a new Candidate; the input must be unchanged."""
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    original_min = candidate.redrob_signals.expected_salary_range_inr_lpa.min
    original_max = candidate.redrob_signals.expected_salary_range_inr_lpa.max

    corrected, _ = fix_salary_inversion(candidate)

    # Input object is unchanged
    assert candidate.redrob_signals.expected_salary_range_inr_lpa.min == original_min
    assert candidate.redrob_signals.expected_salary_range_inr_lpa.max == original_max


def test_fix_salary_inversion_returns_new_object() -> None:
    """The corrected candidate must be a different object, not the same instance."""
    *_, fix_salary_inversion = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    corrected, _ = fix_salary_inversion(candidate)
    assert corrected is not candidate


# ---------------------------------------------------------------------------
# clean_candidate
# ---------------------------------------------------------------------------


def test_clean_candidate_returns_tuple() -> None:
    _, _, _, clean_candidate, _, _ = _imports()
    candidate = _make_candidate(salary_min=15.0, salary_max=30.0)
    result = clean_candidate(candidate)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_clean_candidate_returns_candidate_and_list() -> None:
    _, _, _, clean_candidate, _, _ = _imports()
    candidate = _make_candidate(salary_min=15.0, salary_max=30.0)
    corrected, issues = clean_candidate(candidate)
    assert isinstance(corrected, Candidate)
    assert isinstance(issues, list)


def test_clean_candidate_no_issues_when_no_fix_needed() -> None:
    _, _, _, clean_candidate, _, _ = _imports()
    candidate = _make_candidate(salary_min=15.0, salary_max=30.0)
    _, issues = clean_candidate(candidate)
    assert issues == []


def test_clean_candidate_one_issue_when_salary_inverted() -> None:
    CleaningIssue, _, _, clean_candidate, _, _ = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    _, issues = clean_candidate(candidate)
    assert len(issues) == 1
    assert isinstance(issues[0], CleaningIssue)


def test_clean_candidate_fixes_salary_inversion() -> None:
    _, _, _, clean_candidate, _, _ = _imports()
    candidate = _make_candidate(salary_min=35.0, salary_max=18.0)
    corrected, _ = clean_candidate(candidate)
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.min == 18.0
    assert corrected.redrob_signals.expected_salary_range_inr_lpa.max == 35.0


# ---------------------------------------------------------------------------
# clean_candidates — core entry point
# ---------------------------------------------------------------------------


def test_clean_candidates_empty_list() -> None:
    _, CleaningResult, CleaningSummary, _, clean_candidates, _ = _imports()
    result = clean_candidates([])
    assert isinstance(result, CleaningResult)
    assert result.candidates == []
    assert result.summary.total_candidates == 0
    assert result.summary.candidates_modified == 0
    assert result.summary.issues == []


def test_clean_candidates_returns_cleaning_result() -> None:
    _, CleaningResult, _, _, clean_candidates, _ = _imports()
    candidates = [_make_candidate(salary_min=15.0, salary_max=30.0)]
    result = clean_candidates(candidates)
    assert isinstance(result, CleaningResult)


def test_clean_candidates_returns_same_count_as_input() -> None:
    """No candidate is ever dropped — output length always equals input length."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [_make_candidate(f"CAND_{i:07d}") for i in range(1, 6)]
    result = clean_candidates(candidates)
    assert len(result.candidates) == 5


def test_clean_candidates_no_modification_when_all_clean() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [_make_candidate(f"CAND_{i:07d}", salary_min=10.0, salary_max=20.0)
                  for i in range(1, 4)]
    result = clean_candidates(candidates)
    assert result.summary.candidates_modified == 0
    assert result.summary.issues == []


def test_clean_candidates_correct_modified_count() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    clean = _make_candidate("CAND_0000001", salary_min=10.0, salary_max=20.0)
    inverted1 = _make_candidate("CAND_0000002", salary_min=30.0, salary_max=15.0)
    inverted2 = _make_candidate("CAND_0000003", salary_min=25.0, salary_max=12.0)
    result = clean_candidates([clean, inverted1, inverted2])
    assert result.summary.candidates_modified == 2


def test_clean_candidates_correct_total_count() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [_make_candidate(f"CAND_{i:07d}") for i in range(1, 6)]
    result = clean_candidates(candidates)
    assert result.summary.total_candidates == 5


def test_clean_candidates_issues_list_has_one_entry_per_fix() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    inverted1 = _make_candidate("CAND_0000001", salary_min=30.0, salary_max=15.0)
    inverted2 = _make_candidate("CAND_0000002", salary_min=25.0, salary_max=12.0)
    result = clean_candidates([inverted1, inverted2])
    assert len(result.summary.issues) == 2


def test_clean_candidates_fixes_are_applied_in_output() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    inverted = _make_candidate("CAND_0000001", salary_min=35.0, salary_max=18.0)
    result = clean_candidates([inverted])
    fixed = result.candidates[0]
    assert fixed.redrob_signals.expected_salary_range_inr_lpa.min == 18.0
    assert fixed.redrob_signals.expected_salary_range_inr_lpa.max == 35.0


def test_clean_candidates_preserves_order() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    ids = [f"CAND_{i:07d}" for i in range(1, 6)]
    candidates = [_make_candidate(cid) for cid in ids]
    result = clean_candidates(candidates)
    assert [c.candidate_id for c in result.candidates] == ids


def test_clean_candidates_summary_issues_reference_correct_candidate_ids() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    inverted = _make_candidate("CAND_0042000", salary_min=35.0, salary_max=18.0)
    result = clean_candidates([inverted])
    assert result.summary.issues[0].candidate_id == "CAND_0042000"


# ---------------------------------------------------------------------------
# Integration with real sample fixtures
# ---------------------------------------------------------------------------


def test_clean_candidates_on_sample_fixture_returns_same_count(
    sample_candidates_data: list[dict],
) -> None:
    """Cleaning the real-data fixture slice never drops a candidate."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [Candidate.model_validate(r) for r in sample_candidates_data]
    result = clean_candidates(candidates)
    assert len(result.candidates) == len(candidates)


def test_clean_candidates_on_sample_fixture_all_outputs_are_candidates(
    sample_candidates_data: list[dict],
) -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [Candidate.model_validate(r) for r in sample_candidates_data]
    result = clean_candidates(candidates)
    assert all(isinstance(c, Candidate) for c in result.candidates)


def test_clean_candidates_on_sample_fixture_salary_inversions_corrected(
    sample_candidates_data: list[dict],
) -> None:
    """After cleaning, every candidate's salary min <= max."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidates = [Candidate.model_validate(r) for r in sample_candidates_data]
    result = clean_candidates(candidates)
    for candidate in result.candidates:
        salary = candidate.redrob_signals.expected_salary_range_inr_lpa
        assert salary.min <= salary.max, (
            f"{candidate.candidate_id}: min={salary.min} > max={salary.max} after cleaning"
        )


def test_synthetic_salary_inversion_fixture_is_fixed(
    synthetic_edge_cases_data: dict[str, dict],
) -> None:
    """The salary_inversion synthetic fixture (min=35.0, max=18.0) is corrected."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = Candidate.model_validate(synthetic_edge_cases_data["salary_inversion"])
    result = clean_candidates([candidate])
    fixed = result.candidates[0]
    assert fixed.redrob_signals.expected_salary_range_inr_lpa.min == 18.0
    assert fixed.redrob_signals.expected_salary_range_inr_lpa.max == 35.0
    assert result.summary.candidates_modified == 1


# ---------------------------------------------------------------------------
# Guardrail tests: cleaning must NOT touch sentinel values
# ---------------------------------------------------------------------------


def test_github_activity_score_negative_one_sentinel_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(github_activity_score=-1.0)
    result = clean_candidates([candidate])
    assert result.candidates[0].redrob_signals.github_activity_score == -1.0


def test_offer_acceptance_rate_negative_one_sentinel_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(offer_acceptance_rate=-1.0)
    result = clean_candidates([candidate])
    assert result.candidates[0].redrob_signals.offer_acceptance_rate == -1.0


def test_negative_sentinel_fixture_sentinels_survive(
    synthetic_edge_cases_data: dict[str, dict],
) -> None:
    """The negative_sentinel_values fixture has both sentinels; both must survive."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = Candidate.model_validate(
        synthetic_edge_cases_data["negative_sentinel_values"]
    )
    result = clean_candidates([candidate])
    signals = result.candidates[0].redrob_signals
    assert signals.github_activity_score == -1.0
    assert signals.offer_acceptance_rate == -1.0


# ---------------------------------------------------------------------------
# Guardrail tests: cleaning must NOT touch career_history
# ---------------------------------------------------------------------------


def test_career_history_duration_months_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(career_duration_months=99)
    result = clean_candidates([candidate])
    assert result.candidates[0].career_history[0].duration_months == 99


def test_career_history_description_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate()
    original_description = candidate.career_history[0].description
    result = clean_candidates([candidate])
    assert result.candidates[0].career_history[0].description == original_description


def test_career_history_length_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate()
    original_len = len(candidate.career_history)
    result = clean_candidates([candidate])
    assert len(result.candidates[0].career_history) == original_len


# ---------------------------------------------------------------------------
# Guardrail tests: cleaning must NOT touch skills
# ---------------------------------------------------------------------------


def test_skill_proficiency_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(skill_proficiency="expert")
    result = clean_candidates([candidate])
    assert result.candidates[0].skills[0].proficiency.value == "expert"


def test_skill_duration_months_survives_cleaning() -> None:
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(skill_duration_months=3)
    result = clean_candidates([candidate])
    assert result.candidates[0].skills[0].duration_months == 3


def test_skill_duration_months_zero_survives_cleaning() -> None:
    """duration_months=0 with advanced/expert proficiency is honeypot evidence — never touched."""
    _, _, _, _, clean_candidates, _ = _imports()
    candidate = _make_candidate(skill_proficiency="expert", skill_duration_months=0)
    result = clean_candidates([candidate])
    skill = result.candidates[0].skills[0]
    assert skill.duration_months == 0
    assert skill.proficiency.value == "expert"