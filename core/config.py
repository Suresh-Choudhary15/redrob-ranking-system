"""Runtime configuration for the Redrob candidate-ranking pipeline.

This module is the ONLY place in ``core/`` that is permitted to read
environment variables or a ``.env`` file.  All other modules receive a
``Settings`` instance via dependency injection — they never call
``load_settings()`` themselves and never touch ``os.environ`` directly.

Design rules enforced here:
- No I/O of any kind at import time.  Models are defined as pure classes;
  the environment is only read when ``load_settings()`` is called explicitly
  by application entry points (``apps/cli/rank.py``,
  ``apps/platform/main.py``, or test helpers that need non-default config).
- ``ScoringWeights`` is intentionally empty in Phase 1.  It is reserved for
  Phase 4 (feature-weight tuning) so that the field exists on ``Settings``
  from the start and callers can already pass it around without refactoring.
- Path fields carry ``pathlib.Path`` types throughout; no raw strings are
  used for filesystem locations.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class Paths(BaseSettings):
    """Filesystem locations used by the ranking pipeline.

    All paths default to sensible project-relative values so the pipeline
    runs without any configuration on a freshly cloned repository.  Override
    any field via environment variable or ``.env`` file when the default
    location does not match your deployment layout.

    Environment variable names are uppercased field names prefixed with
    ``REDROB_``, e.g. ``REDROB_CANDIDATES_FILE``.
    """

    model_config = SettingsConfigDict(
        env_prefix="REDROB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    candidates_file: Path = Field(
        default=Path("data/candidates.jsonl"),
        description=(
            "Path to the candidates dataset.  "
            "Accepts both JSONL (one JSON object per line) and a JSON array."
        ),
    )
    job_description_file: Path = Field(
        default=Path("data/job_description.txt"),
        description="Path to the plain-text or Markdown job description.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description=(
            "Directory where the ranked submission CSV and any intermediate "
            "artefacts (embeddings cache, BM25 index) are written."
        ),
    )
    submission_file: Path = Field(
        default=Path("output/submission.csv"),
        description="Final path of the submission CSV produced by the ranking step.",
    )
    embeddings_cache_file: Path = Field(
        default=Path("output/embeddings.npy"),
        description=(
            "NumPy ``.npy`` file caching precomputed candidate embeddings.  "
            "If this file exists and ``Settings.force_recompute`` is False, "
            "the embedding step is skipped."
        ),
    )
    bm25_index_file: Path = Field(
        default=Path("output/bm25_index.pkl"),
        description=(
            "Pickle file caching the precomputed BM25 index over candidate "
            "text.  Invalidated and rebuilt whenever ``force_recompute`` is True."
        ),
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for structured pipeline logs.",
    )


# ---------------------------------------------------------------------------
# ScoringWeights  (Phase 4 placeholder — intentionally empty)
# ---------------------------------------------------------------------------


class ScoringWeights(BaseSettings):
    """Feature weights for the final scoring formula.

    **Reserved for Phase 4.**  This class is empty in Phase 1 so that
    ``Settings`` can already expose a ``scoring_weights`` field without
    requiring downstream callers to be refactored later.  Weights will be
    added here once the feature modules (``core/features/``) are complete
    and empirically validated against the proxy eval set.

    Do not add fields here until Phase 4 begins.
    """

    model_config = SettingsConfigDict(
        env_prefix="REDROB_WEIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Settings  (root configuration object)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration object for the Redrob ranking pipeline.

    Aggregates ``Paths`` and ``ScoringWeights`` sub-models alongside
    top-level pipeline knobs.  Application entry points call
    ``load_settings()`` once, then pass the resulting ``Settings`` instance
    through the call stack.  No module below ``apps/`` should ever
    instantiate ``Settings`` or call ``load_settings()`` directly.

    Environment variable names are uppercased field names prefixed with
    ``REDROB_``, e.g. ``REDROB_LOG_LEVEL``.
    """

    model_config = SettingsConfigDict(
        env_prefix="REDROB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- sub-models ---------------------------------------------------------

    paths: Paths = Field(
        default_factory=Paths,
        description="Filesystem locations used by the pipeline.",
    )
    scoring_weights: ScoringWeights = Field(
        default_factory=ScoringWeights,
        description="Feature weights for the final scoring formula (Phase 4 placeholder).",
    )

    # --- pipeline knobs -----------------------------------------------------

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description=(
            "HuggingFace model ID for the sentence-transformer used to embed "
            "candidate profiles and the job description.  Must be a model "
            "compatible with ``sentence-transformers``.  BGE-small is the "
            "default: asymmetric-retrieval trained, 384-dim, smallest artifact."
        ),
    )
    embedding_batch_size: int = Field(
        default=256,
        ge=1,
        description="Number of candidate texts encoded per forward pass.",
    )
    retrieval_top_k: int = Field(
        default=2000,
        ge=100,
        description=(
            "Number of candidates surfaced by hybrid retrieval (dense + BM25) "
            "before the recruiter-intelligence re-ranker reduces them to 100."
        ),
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "Constant ``k`` in the Reciprocal Rank Fusion formula: "
            "score = 1 / (k + rank).  Higher values smooth rank differences; "
            "lower values amplify them.  60 is the conventional default."
        ),
    )
    force_recompute: bool = Field(
        default=False,
        description=(
            "When True, discard any cached embeddings or BM25 index and "
            "recompute from scratch.  Useful after dataset or model changes."
        ),
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )
    random_seed: int = Field(
        default=42,
        description=(
            "Global random seed for any non-deterministic steps "
            "(tie-breaking, sampling).  Fixing this ensures the submission "
            "CSV is fully reproducible across runs."
        ),
    )
    submission_top_n: int = Field(
        default=100,
        ge=1,
        le=100,
        description=(
            "Number of candidates to include in the final submission CSV. "
            "The spec requires exactly 100; this field exists so integration "
            "tests can set it to a smaller value for speed."
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_settings(env_file: Path | None = None) -> Settings:
    """Construct and return a fully validated ``Settings`` instance.

    This is the **only** function in ``core/`` permitted to trigger
    environment-variable or ``.env`` file reads.  Call it once at process
    start from an application entry point, then pass the result down the
    call stack.

    Args:
        env_file: Optional explicit path to a ``.env`` file.  When provided,
            it is passed to pydantic-settings as the ``_env_file`` override,
            taking precedence over any ``.env`` discovered in the working
            directory.  When ``None``, pydantic-settings uses the
            ``env_file`` value declared on each model's ``model_config``
            (defaulting to ``".env"`` in the current working directory, or
            silently skipping if absent).

    Returns:
        A fully validated ``Settings`` instance with all sub-models
        populated.  Raises ``pydantic.ValidationError`` if any environment
        variable is present but fails type validation.

    Example::

        # Application entry point
        from core.config import load_settings
        settings = load_settings()

        # Integration test with an isolated .env
        settings = load_settings(env_file=Path("tests/.env.test"))
    """
    if env_file is not None:
        return Settings(
            _env_file=env_file,
            paths=Paths(_env_file=env_file),
            scoring_weights=ScoringWeights(_env_file=env_file),
        )
    return Settings(
        paths=Paths(),
        scoring_weights=ScoringWeights(),
    )