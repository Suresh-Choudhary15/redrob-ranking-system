"""Tests for core/features/behavioral.py.

Tests verify ordering relationships and hard invariants only.
No test asserts an exact score value or depends on a specific
weighting scheme — the implementation algorithm may evolve and
these tests must remain valid across any future reweighting.

Conventions
-----------
- REFERENCE_DATE is a fixed constant so time-sensitive assertions
  never drift.
- _make_candidate() exposes every behaviorally-relevant signal field
  as a named parameter with a safe default.
- _score() is a thin wrapper that calls compute() with REFERENCE_DATE
  so individual tests don't repeat the keyword argument.
- Candidates are described by their signal values in test names, not
  by opaque fixture indices, so failures are self-explanatory.
"""

import datetime

import pytest

from core.features import EvidenceRecord, FeatureResult
from core.features.behavioral import compute
from core.schema import Candidate

# Fixed reference date — all recency calculations are relative to this.
REFERENCE_DATE = datetime.date(2026, 6, 25)

# Documented modifier bounds from the architecture spec.
MODIFIER_FLOOR = 0.5
MODIFIER_CEIL = 1.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _days_ago(days: int) -> str:
    """Return an ISO-format date string ``days`` before REFERENCE_DATE."""
    return (REFERENCE_DATE - datetime.timedelta(days=days)).isoformat()


def _make_candidate(
    *,
    candidate_id: str = "CAND_0000001",
    last_active_days_ago: int = 7,
    open_to_work: bool = True,
    response_rate: float = 0.7,
    response_time_hours: float = 24.0,
    verified_email: bool = True,
    verified_phone: bool = True,
    profile_completeness: float = 80.0,
    github_activity_score: float = 30.0,
    offer_acceptance_rate: float = 0.5,
) -> Candidate:
    """Return a minimal but fully valid Candidate with configurable signals."""
    return Candidate.model_validate(
        {
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
                    "duration_months": 60,
                    "is_current": True,
                    "industry": "Software",
                    "company_size": "51-200",
                    "description": "Built systems.",
                }
            ],
            "education": [],
            "skills": [],
            "redrob_signals": {
                "profile_completeness_score": profile_completeness,
                "signup_date": "2024-01-01",
                "last_active_date": _days_ago(last_active_days_ago),
                "open_to_work_flag": open_to_work,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 1,
                "recruiter_response_rate": response_rate,
                "avg_response_time_hours": response_time_hours,
                "skill_assessment_scores": {},
                "connection_count": 100,
                "endorsements_received": 5,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": False,
                "github_activity_score": github_activity_score,
                "search_appearance_30d": 20,
                "saved_by_recruiters_30d": 2,
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": offer_acceptance_rate,
                "verified_email": verified_email,
                "verified_phone": verified_phone,
                "linkedin_connected": False,
            },
        }
    )


def _score(candidate: Candidate) -> float:
    """Call compute() with the fixed REFERENCE_DATE and return the score."""
    return compute(candidate, reference_date=REFERENCE_DATE).score


def _result(candidate: Candidate) -> FeatureResult:
    """Call compute() with the fixed REFERENCE_DATE and return the full result."""
    return compute(candidate, reference_date=REFERENCE_DATE)


# ---------------------------------------------------------------------------
# Return type and structure invariants
# ---------------------------------------------------------------------------


def test_returns_feature_result() -> None:
    candidate = _make_candidate()
    result = _result(candidate)
    assert isinstance(result, FeatureResult)


def test_feature_type_is_modifier() -> None:
    candidate = _make_candidate()
    result = _result(candidate)
    assert result.feature_type == "modifier"


def test_score_is_float() -> None:
    candidate = _make_candidate()
    result = _result(candidate)
    assert isinstance(result.score, float)


def test_evidence_is_non_empty_list() -> None:
    """At least one evidence record must always be emitted."""
    candidate = _make_candidate()
    result = _result(candidate)
    assert isinstance(result.evidence, list)
    assert len(result.evidence) >= 1


def test_evidence_items_are_evidence_records() -> None:
    candidate = _make_candidate()
    result = _result(candidate)
    assert all(isinstance(e, EvidenceRecord) for e in result.evidence)


# ---------------------------------------------------------------------------
# Score range invariants
# ---------------------------------------------------------------------------


def test_score_at_floor_for_worst_inputs() -> None:
    """Worst-case signals must not produce a score below MODIFIER_FLOOR."""
    worst = _make_candidate(
        last_active_days_ago=365,
        open_to_work=False,
        response_rate=0.0,
        response_time_hours=9999.0,
        verified_email=False,
        verified_phone=False,
        profile_completeness=0.0,
    )
    assert _score(worst) >= MODIFIER_FLOOR


def test_score_at_ceiling_for_best_inputs() -> None:
    """Best-case signals must not produce a score above MODIFIER_CEIL."""
    best = _make_candidate(
        last_active_days_ago=0,
        open_to_work=True,
        response_rate=1.0,
        response_time_hours=0.0,
        verified_email=True,
        verified_phone=True,
        profile_completeness=100.0,
    )
    assert _score(best) <= MODIFIER_CEIL


@pytest.mark.parametrize(
    "last_active_days_ago, response_rate, response_time_hours, "
    "verified_email, verified_phone, profile_completeness",
    [
        (0, 1.0, 0.0, True, True, 100.0),       # ideal
        (365, 0.0, 9999.0, False, False, 0.0),   # worst
        (30, 0.5, 48.0, True, False, 60.0),      # average
        (90, 0.3, 120.0, False, True, 40.0),     # mixed
        (7, 0.8, 12.0, True, True, 85.0),        # strong recent
    ],
)
def test_score_always_within_valid_range(
    last_active_days_ago: int,
    response_rate: float,
    response_time_hours: float,
    verified_email: bool,
    verified_phone: bool,
    profile_completeness: float,
) -> None:
    """Score must be in [MODIFIER_FLOOR, MODIFIER_CEIL] for any valid input."""
    candidate = _make_candidate(
        last_active_days_ago=last_active_days_ago,
        response_rate=response_rate,
        response_time_hours=response_time_hours,
        verified_email=verified_email,
        verified_phone=verified_phone,
        profile_completeness=profile_completeness,
    )
    score = _score(candidate)
    assert MODIFIER_FLOOR <= score <= MODIFIER_CEIL, (
        f"score={score} is outside [{MODIFIER_FLOOR}, {MODIFIER_CEIL}]"
    )


# ---------------------------------------------------------------------------
# Ordering: recency
# ---------------------------------------------------------------------------


def test_recent_candidate_scores_higher_than_stale() -> None:
    """A candidate active 3 days ago must score higher than one active 300 days ago."""
    recent = _make_candidate(last_active_days_ago=3)
    stale = _make_candidate(last_active_days_ago=300)
    assert _score(recent) > _score(stale)


def test_score_decreases_monotonically_with_staleness() -> None:
    """Score must be non-increasing as days since last activity increases,
    all other signals held constant."""
    scores = [
        _score(_make_candidate(last_active_days_ago=d))
        for d in [1, 30, 90, 180, 365]
    ]
    for earlier, later in zip(scores, scores[1:]):
        assert earlier >= later, (
            f"Score did not decrease (or stay equal) as staleness increased: "
            f"{scores}"
        )


def test_later_reference_date_produces_lower_or_equal_score() -> None:
    """Holding last_active_date fixed, a later reference date means more
    staleness and must produce a lower or equal score."""
    candidate = _make_candidate(last_active_days_ago=30)
    score_now = compute(candidate, reference_date=REFERENCE_DATE).score
    score_later = compute(
        candidate,
        reference_date=REFERENCE_DATE + datetime.timedelta(days=90),
    ).score
    assert score_now >= score_later


# ---------------------------------------------------------------------------
# Ordering: responsiveness
# ---------------------------------------------------------------------------


def test_high_response_rate_scores_higher_than_low() -> None:
    """A candidate who responds to 90% of recruiter messages must score higher
    than one who responds to 5%, all else equal."""
    responsive = _make_candidate(response_rate=0.9)
    unresponsive = _make_candidate(response_rate=0.05)
    assert _score(responsive) > _score(unresponsive)


def test_fast_response_time_scores_higher_than_slow() -> None:
    """A candidate who responds in 1 hour must score higher than one who
    takes 200 hours, all else equal."""
    fast = _make_candidate(response_time_hours=1.0)
    slow = _make_candidate(response_time_hours=200.0)
    assert _score(fast) > _score(slow)


# ---------------------------------------------------------------------------
# Ordering: verification
# ---------------------------------------------------------------------------


def test_both_verified_scores_higher_than_both_unverified() -> None:
    """email + phone verified must score higher than neither verified."""
    verified = _make_candidate(verified_email=True, verified_phone=True)
    unverified = _make_candidate(verified_email=False, verified_phone=False)
    assert _score(verified) > _score(unverified)


# ---------------------------------------------------------------------------
# Ordering: combined
# ---------------------------------------------------------------------------


def test_ideal_candidate_scores_higher_than_worst_candidate() -> None:
    """An ideal candidate on all dimensions must substantially outscore
    the worst-case candidate."""
    ideal = _make_candidate(
        last_active_days_ago=0,
        open_to_work=True,
        response_rate=1.0,
        response_time_hours=0.0,
        verified_email=True,
        verified_phone=True,
        profile_completeness=100.0,
    )
    worst = _make_candidate(
        last_active_days_ago=365,
        open_to_work=False,
        response_rate=0.0,
        response_time_hours=9999.0,
        verified_email=False,
        verified_phone=False,
        profile_completeness=0.0,
    )
    assert _score(ideal) > _score(worst)


# ---------------------------------------------------------------------------
# Evidence quality invariants
# ---------------------------------------------------------------------------


def test_stale_candidate_evidence_includes_negative_direction() -> None:
    """A candidate inactive for 300 days must have at least one evidence
    record flagged as negative — this is what the reasoning column uses."""
    stale = _make_candidate(last_active_days_ago=300)
    result = _result(stale)
    directions = [e.direction for e in result.evidence]
    assert "negative" in directions, (
        f"No negative evidence for a 300-day-stale candidate. "
        f"Directions found: {directions}"
    )


def test_highly_engaged_candidate_evidence_includes_positive_direction() -> None:
    """A recently active, responsive, verified candidate must have at least
    one positive evidence record."""
    engaged = _make_candidate(
        last_active_days_ago=2,
        response_rate=0.95,
        response_time_hours=3.0,
        verified_email=True,
        verified_phone=True,
    )
    result = _result(engaged)
    directions = [e.direction for e in result.evidence]
    assert "positive" in directions, (
        f"No positive evidence for a highly engaged candidate. "
        f"Directions found: {directions}"
    )


def test_all_evidence_source_fields_reference_redrob_signals() -> None:
    """Every source_field must point into redrob_signals — behavioral
    features must not fabricate fields outside their data source."""
    candidate = _make_candidate()
    result = _result(candidate)
    for record in result.evidence:
        assert record.source_field.startswith("redrob_signals."), (
            f"source_field {record.source_field!r} does not start with "
            f"'redrob_signals.' — behavioral may only reference its own data."
        )


def test_all_evidence_directions_are_valid_literal_values() -> None:
    """direction must always be one of the three allowed Literal values."""
    candidate = _make_candidate()
    result = _result(candidate)
    valid = {"positive", "negative", "neutral"}
    for record in result.evidence:
        assert record.direction in valid, (
            f"Invalid direction {record.direction!r} in evidence record."
        )


def test_all_evidence_magnitudes_are_none_or_non_negative_float() -> None:
    """magnitude must be None or a non-negative float — never negative."""
    candidate = _make_candidate()
    result = _result(candidate)
    for record in result.evidence:
        if record.magnitude is not None:
            assert isinstance(record.magnitude, float)
            assert record.magnitude >= 0.0, (
                f"Negative magnitude {record.magnitude} in evidence record."
            )


def test_all_evidence_feature_names_are_non_empty_strings() -> None:
    candidate = _make_candidate()
    result = _result(candidate)
    for record in result.evidence:
        assert isinstance(record.feature_name, str)
        assert len(record.feature_name) > 0