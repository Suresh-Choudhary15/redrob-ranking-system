"""Candidate data cleaning for the Redrob ranking pipeline.

This module performs the single, narrowly-scoped normalization step that
sits between loading and feature extraction: correcting known data-generation
artifacts so that downstream feature modules receive internally consistent
input.

Permitted fix
-------------
**Salary range inversion** — ``expected_salary_range_inr_lpa.min > max``.
Observed directly in the sample dataset across several candidates.  Treated
as a data-generation noise artifact, not a candidate-quality signal.  The
values are swapped and an audit record is emitted.

What this module must never touch
----------------------------------
The following fields are deliberately left unmodified, even when they appear
anomalous, because they are **honeypot-detector evidence** required by
``core/features/honeypot.py`` in Phase 2:

- ``career_history[*].duration_months`` — tenure inconsistency is a
  honeypot signal.
- ``career_history[*]`` dates and descriptions — all career-history content
  is honeypot evidence.
- ``skills[*].proficiency`` and ``skills[*].duration_months`` — expert
  proficiency with near-zero duration is the spec's own named honeypot
  example.
- ``redrob_signals.github_activity_score`` — ``-1`` is a documented
  missing-data sentinel (no GitHub linked), not an error value.
- ``redrob_signals.offer_acceptance_rate`` — ``-1`` is a documented
  missing-data sentinel (no prior offers), not an error value.

No candidate is ever removed from the pool.  Input length always equals
output length.  ``Candidate`` objects are never mutated in place; Pydantic
v2 ``model_copy(update={...})`` is used to produce new instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from core.schema import Candidate, SalaryRange


# ---------------------------------------------------------------------------
# Audit types
# ---------------------------------------------------------------------------


@dataclass
class CleaningIssue:
    """A single corrective action applied to one candidate's data.

    Every fix emits exactly one ``CleaningIssue`` so that callers have a
    complete, ordered audit trail of what was changed and why.

    Attributes:
        candidate_id:    The ``CAND_XXXXXXX`` identifier of the affected
                         candidate.
        field:           Dot-notation path to the field that was corrected,
                         e.g. ``"expected_salary_range_inr_lpa"``.
        original_value:  The value as it existed in the loaded data, before
                         correction.  Stored as a plain Python object so it
                         can be serialised to JSON for logging.
        corrected_value: The value after correction.
        description:     A human-readable sentence explaining what was wrong
                         and how it was fixed.
    """

    candidate_id: str
    field: str
    original_value: Any
    corrected_value: Any
    description: str


@dataclass
class CleaningSummary:
    """Aggregate statistics for one ``clean_candidates()`` run.

    Attributes:
        total_candidates:    Total number of candidates processed.
        candidates_modified: Number of candidates where at least one field
                             was corrected.
        issues:              Flat list of every ``CleaningIssue`` emitted
                             during this run, in processing order.
    """

    total_candidates: int
    candidates_modified: int
    issues: list[CleaningIssue] = dc_field(default_factory=list)


@dataclass
class CleaningResult:
    """Return value of ``clean_candidates()``.

    Attributes:
        candidates: Cleaned ``Candidate`` objects in the same order as the
                    input.  Length always equals the input length — no
                    candidate is ever dropped.
        summary:    Aggregate statistics and the full issue audit trail.
    """

    candidates: list[Candidate]
    summary: CleaningSummary


# ---------------------------------------------------------------------------
# Individual fix functions
# ---------------------------------------------------------------------------


def fix_salary_inversion(
    candidate: Candidate,
) -> tuple[Candidate, CleaningIssue | None]:
    """Correct a salary range where ``min > max`` by swapping the two values.

    ``min == max`` is valid (a single-point expectation) and is left
    unchanged.  ``-1`` sentinels on other signal fields are never touched by
    this function.

    Args:
        candidate: A validated ``Candidate`` instance.

    Returns:
        A 2-tuple ``(corrected_candidate, issue_or_none)``.  When no
        inversion is detected the returned candidate is the same object as
        the input and the second element is ``None``.  When an inversion is
        corrected, the returned candidate is a **new** object produced via
        ``model_copy``; the input is not mutated.
    """
    salary = candidate.redrob_signals.expected_salary_range_inr_lpa

    if salary.min <= salary.max:
        return candidate, None

    # Swap the two values.
    original_min, original_max = salary.min, salary.max
    corrected_salary = SalaryRange(min=original_max, max=original_min)

    corrected_signals = candidate.redrob_signals.model_copy(
        update={"expected_salary_range_inr_lpa": corrected_salary}
    )
    corrected_candidate = candidate.model_copy(
        update={"redrob_signals": corrected_signals}
    )

    issue = CleaningIssue(
        candidate_id=candidate.candidate_id,
        field="expected_salary_range_inr_lpa",
        original_value={"min": original_min, "max": original_max},
        corrected_value={"min": corrected_salary.min, "max": corrected_salary.max},
        description=(
            f"Salary range inversion: min ({original_min}) > max ({original_max}); "
            f"swapped to min={corrected_salary.min}, max={corrected_salary.max}."
        ),
    )
    return corrected_candidate, issue


# ---------------------------------------------------------------------------
# Candidate-level orchestration
# ---------------------------------------------------------------------------


def clean_candidate(
    candidate: Candidate,
) -> tuple[Candidate, list[CleaningIssue]]:
    """Apply all permitted cleaning fixes to a single candidate.

    Currently applies only ``fix_salary_inversion``.  Additional fix
    functions should be added here in Phase 2+ if new data-generation
    artifacts are identified — never by modifying the individual fix
    functions to do more than their named job.

    Args:
        candidate: A validated ``Candidate`` instance.

    Returns:
        A 2-tuple ``(corrected_candidate, issues)``.  ``issues`` is an empty
        list when no fix was necessary.  The returned candidate is either the
        original object (no changes) or a new object produced via
        ``model_copy`` (one or more changes applied).
    """
    issues: list[CleaningIssue] = []

    candidate, salary_issue = fix_salary_inversion(candidate)
    if salary_issue is not None:
        issues.append(salary_issue)

    return candidate, issues


# ---------------------------------------------------------------------------
# Pool-level entry point
# ---------------------------------------------------------------------------


def clean_candidates(candidates: list[Candidate]) -> CleaningResult:
    """Apply all permitted cleaning fixes to a pool of candidates.

    Iterates the full pool, applies ``clean_candidate()`` to each record,
    and returns a ``CleaningResult`` containing the cleaned candidates plus
    a ``CleaningSummary`` with aggregate statistics and a flat issue audit
    trail.

    No candidate is ever removed from the pool regardless of how many issues
    are found.  The output list is always the same length as the input list
    and preserves input order.

    Args:
        candidates: A list of validated ``Candidate`` instances, as returned
                    by ``core.loading.load_candidates()``.

    Returns:
        A ``CleaningResult`` with the cleaned candidate pool and a
        ``CleaningSummary`` describing every corrective action taken.
    """
    cleaned: list[Candidate] = []
    all_issues: list[CleaningIssue] = []
    candidates_modified: int = 0

    for candidate in candidates:
        corrected, issues = clean_candidate(candidate)
        cleaned.append(corrected)
        if issues:
            candidates_modified += 1
            all_issues.extend(issues)

    summary = CleaningSummary(
        total_candidates=len(candidates),
        candidates_modified=candidates_modified,
        issues=all_issues,
    )
    return CleaningResult(candidates=cleaned, summary=summary)