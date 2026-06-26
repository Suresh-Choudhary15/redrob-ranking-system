"""Shared types for the recruiter-intelligence feature layer.

Every module in ``core/features/`` returns a ``FeatureResult`` that carries:

- A numeric score whose meaning depends on ``feature_type``.
- An ordered list of ``EvidenceRecord`` objects that trace the score back
  to specific, named fields in the candidate's profile.

Feature type semantics
-----------------------
``"component"``
    An additive input to ``base_fit`` in ``core/scoring.py``.
    Score is in **[0, 1]** — higher is a stronger fit signal.
    Examples: ``skill_fit``, ``domain_specificity``, ``experience``.

``"modifier"``
    A multiplicative factor applied on top of ``base_fit``.
    Score is in **[0.05, 1.05]** — values below 1.0 suppress the base
    score; 1.0 is neutral; values above 1.0 give a small boost.
    Examples: ``behavioral``, ``product_vs_services``, ``honeypot_gate``.

Evidence record contract
-------------------------
``source_field`` is a dot-notation path to the literal field value that
drove this piece of evidence — e.g. ``"redrob_signals.last_active_date"``.
It must be a real field path traceable to the candidate's raw profile so
that Stage 4 reviewers can verify any claim against the source JSON in
seconds.  It is never a synthesised summary.

``raw_value`` is the literal value read from that field, stored as-is.
It is never paraphrased or processed before storage.

``magnitude`` is optional.  When present it is a non-negative float
indicating how much this evidence item moved the feature score, used by
``core/explain.py`` to select the top-N strongest strengths and risks
for the ``reasoning`` column.

Both types use ``slots=True`` for memory efficiency across the 3,000–5,000
candidate shortlist that the scoring layer processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["EvidenceRecord", "FeatureResult"]


@dataclass(slots=True)
class EvidenceRecord:
    """One atomic piece of evidence linking a feature score to a profile field.

    Attributes:
        feature_name:  Name of the feature module that emitted this record,
                       e.g. ``"behavioral"``, ``"skill_fit"``.
        raw_value:     The literal value read from the candidate's profile at
                       ``source_field``.  Never a paraphrase or derivation.
        source_field:  Dot-notation path to the originating field, e.g.
                       ``"redrob_signals.recruiter_response_rate"``.  Must be
                       a real, traceable field path.
        direction:     Whether this evidence helps (``"positive"``), hurts
                       (``"negative"``), or is contextual (``"neutral"``).
        magnitude:     Optional non-negative float indicating how much weight
                       this evidence carried in the final score.  ``None``
                       when the feature module does not compute a per-record
                       magnitude.
    """

    feature_name: str
    raw_value: object
    source_field: str
    direction: Literal["positive", "negative", "neutral"]
    magnitude: float | None


@dataclass(slots=True)
class FeatureResult:
    """The return value of every ``compute()`` function in ``core/features/``.

    Attributes:
        score:        The feature's numeric output.  Range and semantics
                      depend on ``feature_type`` — see module docstring.
        feature_type: ``"component"`` for additive base-fit inputs;
                      ``"modifier"`` for multiplicative post-base factors.
        evidence:     Ordered list of evidence records produced during
                      computation.  Must be non-empty — at minimum one
                      record must always be emitted so that ``explain.py``
                      can always produce a grounded reasoning sentence.
    """

    score: float
    feature_type: Literal["component", "modifier"]
    evidence: list[EvidenceRecord]