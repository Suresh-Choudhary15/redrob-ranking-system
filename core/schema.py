"""Pydantic v2 models mirroring ``candidate_schema.json``.

This module is the single source of truth for what a *valid* candidate
record looks like. It performs structural and type validation only — no
normalization, correction, cleaning, or scoring. In particular:

- ``SalaryRange`` does not enforce ``min <= max``. The raw dataset contains
  inverted ranges; correcting them is a data-cleaning concern handled
  elsewhere, not a schema-validation concern.
- The ``-1`` sentinel values permitted on ``github_activity_score`` and
  ``offer_acceptance_rate`` are modeled as ordinary valid values within
  range. Their special meaning ("no data") is a feature-engineering
  concern for a later phase, not something this module interprets.

Field and model ordering follows ``candidate_schema.json``'s own
``properties``/``required`` ordering wherever a dependency doesn't force
otherwise, so this file can be read side by side with the schema.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field

CANDIDATE_ID_PATTERN = r"^CAND_[0-9]{7}$"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CompanySize(StrEnum):
    """Employee-count bracket, shared by Profile and CareerHistoryEntry."""

    SIZE_1_10 = "1-10"
    SIZE_11_50 = "11-50"
    SIZE_51_200 = "51-200"
    SIZE_201_500 = "201-500"
    SIZE_501_1000 = "501-1000"
    SIZE_1001_5000 = "1001-5000"
    SIZE_5001_10000 = "5001-10000"
    SIZE_10001_PLUS = "10001+"


class ProficiencyLevel(StrEnum):
    """Self-reported skill proficiency."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class WorkMode(StrEnum):
    """Candidate's preferred work arrangement."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"


class LanguageProficiency(StrEnum):
    """Spoken/written language proficiency."""

    BASIC = "basic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    NATIVE = "native"


class EducationTier(StrEnum):
    """Internal institution-prestige tiering."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Profile and career/education/skill building blocks
# ---------------------------------------------------------------------------


class Profile(BaseModel):
    """``candidate.profile`` — all fields required per schema."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str


class CareerHistoryEntry(BaseModel):
    """One entry in ``candidate.career_history``. All fields required.

    ``end_date`` is a required key but its value may be ``None`` (current
    role) — it has no default, so the key must still be present.
    """

    company: str
    title: str
    start_date: date
    end_date: date | None
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str


class EducationEntry(BaseModel):
    """One entry in ``candidate.education``.

    ``grade`` and ``tier`` are not in the schema's ``required`` list for
    this object and default to ``None``.
    """

    institution: str
    degree: str
    field_of_study: str
    start_year: int = Field(ge=1970, le=2030)
    end_year: int = Field(ge=1970, le=2035)
    grade: str | None = None
    tier: EducationTier | None = None


class SkillEntry(BaseModel):
    """One entry in ``candidate.skills``.

    ``duration_months`` is not in the schema's ``required`` list for this
    object and defaults to ``None``.
    """

    name: str
    proficiency: ProficiencyLevel
    endorsements: int = Field(ge=0)
    duration_months: int | None = Field(default=None, ge=0)


class CertificationEntry(BaseModel):
    """One entry in the optional ``candidate.certifications`` list."""

    name: str
    issuer: str
    year: int


class LanguageEntry(BaseModel):
    """One entry in the optional ``candidate.languages`` list."""

    language: str
    proficiency: LanguageProficiency


# ---------------------------------------------------------------------------
# Redrob platform signals
# ---------------------------------------------------------------------------


class SalaryRange(BaseModel):
    """``redrob_signals.expected_salary_range_inr_lpa``, in INR lakhs/year.

    Deliberately permissive: no cross-field validation enforcing
    ``min <= max``. Inverted ranges exist in the raw data; correcting them
    is a data-cleaning concern, not a schema-validation concern.
    """

    min: float = Field(ge=0)
    max: float = Field(ge=0)


class RedrobSignals(BaseModel):
    """``candidate.redrob_signals``.

    All 23 fields are required, matching ``candidate_schema.json``'s
    ``redrob_signals.required`` list exactly, in the same order.
    ``github_activity_score`` (-1 to 100) and ``offer_acceptance_rate``
    (-1 to 1) use ``-1`` as a documented "no data" sentinel; it is accepted
    here as an ordinary in-range value, not specially handled.
    """

    profile_completeness_score: float = Field(ge=0, le=100)
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1)
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: dict[str, float]
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: WorkMode
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1, le=100)
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0, le=1)
    offer_acceptance_rate: float = Field(ge=-1, le=1)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


# ---------------------------------------------------------------------------
# Top-level candidate
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """A single candidate profile — ``candidate_schema.json``'s root object.

    ``candidate_id``, ``profile``, ``career_history``, ``education``,
    ``skills``, and ``redrob_signals`` are required, matching the schema's
    top-level ``required`` list exactly. ``certifications`` and
    ``languages`` are not in that list and default to an empty list.
    """

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    profile: Profile
    career_history: list[CareerHistoryEntry] = Field(min_length=1, max_length=10)
    education: list[EducationEntry] = Field(max_length=5)
    skills: list[SkillEntry]
    certifications: list[CertificationEntry] = Field(default_factory=list)
    languages: list[LanguageEntry] = Field(default_factory=list)
    redrob_signals: RedrobSignals