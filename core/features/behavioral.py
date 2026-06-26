"""Behavioral availability modifier for the Redrob ranking pipeline.

Answers the question: is this candidate actually reachable and available
right now, or are they theoretically available but practically dormant?

A perfect-on-paper candidate who hasn't logged in for six months and has
a 5% recruiter response rate is, for hiring purposes, not actually
available.  This feature down-weights such candidates multiplicatively
rather than additively, so their base skill-fit score is suppressed without
being zeroed out — suppressed is more honest than invisible.

Modifier range: [0.5, 1.05]
  - 0.5: floor — the candidate is maximally unavailable on every signal.
  - 1.0: neutral — solid availability, no special penalty or boost.
  - 1.05: ceiling — exceptional availability across all signals (slight
    positive signal for a candidate who is demonstrably in active search).

Sub-scores (all in [0, 1] before weighting)
--------------------------------------------
recency         weight 0.50
    Exponential decay from last_active_date.  Half-life is 90 days:
    a candidate active 90 days ago scores ≈ 0.37; one active 180 days
    ago (≈ 6 months, the JD's "not actually available" threshold) scores
    ≈ 0.14.

responsiveness  weight 0.35
    Blend of recruiter_response_rate (0–1) and a normalized inverse of
    avg_response_time_hours (capped at 240 h before it equals "never").
    open_to_work_flag contributes a small bonus (0.1).

verification    weight 0.15
    Fraction of {verified_email, verified_phone} that are True, blended
    lightly with profile_completeness_score.  Low-variance but meaningful
    as a trust signal — verified contact details mean the recruiter can
    actually reach the candidate.

Formula
-------
    weighted  = 0.50 × recency + 0.35 × responsiveness + 0.15 × verification
    modifier  = 0.5 + 0.55 × weighted
    modifier  = clamp(modifier, 0.5, 1.05)

Public interface
----------------
    compute(candidate, *, reference_date) -> FeatureResult

``reference_date`` is always an explicit parameter — never ``datetime.date.today()``
inside the function — so the pipeline is deterministic and tests are hermetic.
``rank.py`` passes ``datetime.date.today()`` at runtime.
"""

from __future__ import annotations

import datetime
import math

from core.features import EvidenceRecord, FeatureResult
from core.schema import Candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECENCY_HALF_LIFE_DAYS: float = 90.0
_RESPONSE_TIME_CAP_HOURS: float = 240.0
_OPEN_TO_WORK_BONUS: float = 0.10

_MODIFIER_FLOOR: float = 0.50
_MODIFIER_CEIL: float = 1.05
_MODIFIER_RANGE: float = _MODIFIER_CEIL - _MODIFIER_FLOOR  # 0.55

_RECENCY_WEIGHT: float = 0.50
_RESPONSIVENESS_WEIGHT: float = 0.35
_VERIFICATION_WEIGHT: float = 0.15

# Thresholds that drive evidence direction labels.
_RECENCY_NEGATIVE_THRESHOLD_DAYS: int = 90   # beyond this → "negative"
_RECENCY_POSITIVE_THRESHOLD_DAYS: int = 30   # within this → "positive"
_RESPONSE_RATE_NEGATIVE_THRESHOLD: float = 0.30
_RESPONSE_RATE_POSITIVE_THRESHOLD: float = 0.70


# ---------------------------------------------------------------------------
# Sub-score functions (each returns a float in [0, 1])
# ---------------------------------------------------------------------------


def _recency_sub_score(days_since_active: int) -> float:
    """Exponential decay: score = exp(-days / half_life).

    Half-life of 90 days means a candidate active exactly 90 days ago
    retains ≈ 37% of the recency score.  At 180 days (≈ 6 months, the
    JD's stated "not actually available" threshold) the score is ≈ 14%.
    Never negative; never exceeds 1.0.
    """
    return math.exp(-max(0, days_since_active) / _RECENCY_HALF_LIFE_DAYS)


def _responsiveness_sub_score(
    response_rate: float,
    response_time_hours: float,
    open_to_work: bool,
) -> float:
    """Blend response rate, normalized inverse response time, and open-to-work flag.

    response_time_hours is capped at _RESPONSE_TIME_CAP_HOURS before inversion
    so that arbitrarily large values map cleanly to 0.0 rather than going
    negative.  open_to_work adds a small bonus that can push the raw sum
    above 1.0; the result is clamped at 1.0.
    """
    capped_time = min(response_time_hours, _RESPONSE_TIME_CAP_HOURS)
    normalized_time = 1.0 - capped_time / _RESPONSE_TIME_CAP_HOURS  # in [0, 1]
    base = 0.70 * response_rate + 0.30 * normalized_time
    bonus = _OPEN_TO_WORK_BONUS if open_to_work else 0.0
    return min(1.0, base + bonus)


def _verification_sub_score(
    verified_email: bool,
    verified_phone: bool,
    completeness_score: float,
) -> float:
    """Fraction of booleans verified, blended with profile completeness.

    Low-variance but directionally correct: a candidate who has not
    verified either contact channel is marginally less reachable, and a
    sparse profile is a mild trust signal.
    """
    verification_fraction = (int(verified_email) + int(verified_phone)) / 2.0
    completeness_fraction = max(0.0, min(1.0, completeness_score / 100.0))
    return 0.60 * verification_fraction + 0.40 * completeness_fraction


# ---------------------------------------------------------------------------
# Evidence builders (each returns an EvidenceRecord or None)
# ---------------------------------------------------------------------------


def _recency_evidence(
    days_since_active: int,
    last_active_date: datetime.date,
    sub_score: float,
) -> EvidenceRecord:
    """Always emitted — recency is the most important availability signal."""
    if days_since_active <= _RECENCY_POSITIVE_THRESHOLD_DAYS:
        direction: str = "positive"
    elif days_since_active <= _RECENCY_NEGATIVE_THRESHOLD_DAYS:
        direction = "neutral"
    else:
        direction = "negative"

    return EvidenceRecord(
        feature_name="behavioral",
        raw_value=last_active_date.isoformat(),
        source_field="redrob_signals.last_active_date",
        direction=direction,  # type: ignore[arg-type]
        magnitude=sub_score,
    )


def _responsiveness_evidence(
    response_rate: float,
    sub_score: float,
) -> EvidenceRecord | None:
    """Emitted only when the response rate is notably high or low."""
    if response_rate < _RESPONSE_RATE_NEGATIVE_THRESHOLD:
        direction = "negative"
    elif response_rate >= _RESPONSE_RATE_POSITIVE_THRESHOLD:
        direction = "positive"
    else:
        return None  # middle range — not noteworthy enough to surface

    return EvidenceRecord(
        feature_name="behavioral",
        raw_value=response_rate,
        source_field="redrob_signals.recruiter_response_rate",
        direction=direction,  # type: ignore[arg-type]
        magnitude=sub_score,
    )


def _verification_evidence(
    verified_email: bool,
    verified_phone: bool,
    sub_score: float,
) -> EvidenceRecord | None:
    """Emitted only when both channels are verified (positive) or
    neither is (negative) — mixed states are not notable enough."""
    both_true = verified_email and verified_phone
    both_false = (not verified_email) and (not verified_phone)

    if both_true:
        direction = "positive"
    elif both_false:
        direction = "negative"
    else:
        return None

    return EvidenceRecord(
        feature_name="behavioral",
        raw_value={"verified_email": verified_email, "verified_phone": verified_phone},
        source_field="redrob_signals.verified_email,redrob_signals.verified_phone",
        direction=direction,  # type: ignore[arg-type]
        magnitude=sub_score,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute(candidate: Candidate, *, reference_date: datetime.date) -> FeatureResult:
    """Compute the behavioral availability modifier for a single candidate.

    Args:
        candidate:      A validated ``Candidate`` instance.
        reference_date: The date treated as "today" for recency calculations.
                        Pass ``datetime.date.today()`` from ``rank.py``;
                        pass a fixed date from tests for hermeticity.

    Returns:
        A ``FeatureResult`` with ``feature_type="modifier"`` and score in
        [0.5, 1.05].  Evidence list is always non-empty (recency record is
        always emitted); additional records are emitted when the response
        rate or verification status is notably high or low.
    """
    signals = candidate.redrob_signals

    # --- parse last_active_date ----------------------------------------- #
    last_active_date = signals.last_active_date
    days_since_active = max(0, (reference_date - last_active_date).days)

    # --- compute sub-scores --------------------------------------------- #
    recency = _recency_sub_score(days_since_active)
    responsiveness = _responsiveness_sub_score(
        signals.recruiter_response_rate,
        signals.avg_response_time_hours,
        signals.open_to_work_flag,
    )
    verification = _verification_sub_score(
        signals.verified_email,
        signals.verified_phone,
        signals.profile_completeness_score,
    )

    # --- combine into modifier ------------------------------------------ #
    weighted = (
        _RECENCY_WEIGHT * recency
        + _RESPONSIVENESS_WEIGHT * responsiveness
        + _VERIFICATION_WEIGHT * verification
    )
    modifier = _MODIFIER_FLOOR + _MODIFIER_RANGE * weighted
    modifier = max(_MODIFIER_FLOOR, min(_MODIFIER_CEIL, modifier))

    # --- build evidence records ----------------------------------------- #
    evidence: list[EvidenceRecord] = []

    evidence.append(_recency_evidence(days_since_active, signals.last_active_date, recency))

    resp_record = _responsiveness_evidence(signals.recruiter_response_rate, responsiveness)
    if resp_record is not None:
        evidence.append(resp_record)

    verif_record = _verification_evidence(
        signals.verified_email, signals.verified_phone, verification
    )
    if verif_record is not None:
        evidence.append(verif_record)

    return FeatureResult(
        score=modifier,
        feature_type="modifier",
        evidence=evidence,
    )