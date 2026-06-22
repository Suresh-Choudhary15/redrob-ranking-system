"""Tests for core/config.py — Settings, Paths, ScoringWeights, load_settings().

Covers:
- Default values load correctly with no .env file present.
- load_settings() returns a fully populated Settings instance.
- Sub-models (Paths, ScoringWeights) are accessible on Settings.
- Path fields are pathlib.Path instances, not raw strings.
- Numeric field constraints are enforced (ge/le bounds).
- Environment variable overrides work for key fields.
- An explicit env_file path is honoured by load_settings().
- No I/O happens at import time (importing config never reads the env).

No .env file is written to disk — env overrides are injected via
monkeypatch.setenv() so tests remain hermetic and side-effect-free.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import Paths, Settings, ScoringWeights, load_settings


# ---------------------------------------------------------------------------
# load_settings() — default values, no .env
# ---------------------------------------------------------------------------


def test_load_settings_returns_settings_instance() -> None:
    """load_settings() with no env file returns a Settings object."""
    settings = load_settings()
    assert isinstance(settings, Settings)


def test_load_settings_sub_models_are_populated() -> None:
    """Settings.paths and Settings.scoring_weights are present and typed."""
    settings = load_settings()
    assert isinstance(settings.paths, Paths)
    assert isinstance(settings.scoring_weights, ScoringWeights)


def test_default_embedding_model() -> None:
    settings = load_settings()
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_default_retrieval_top_k() -> None:
    settings = load_settings()
    assert settings.retrieval_top_k == 2000


def test_default_rrf_k() -> None:
    settings = load_settings()
    assert settings.rrf_k == 60


def test_default_force_recompute_is_false() -> None:
    settings = load_settings()
    assert settings.force_recompute is False


def test_default_log_level() -> None:
    settings = load_settings()
    assert settings.log_level == "INFO"


def test_default_random_seed() -> None:
    settings = load_settings()
    assert settings.random_seed == 42


def test_default_submission_top_n() -> None:
    settings = load_settings()
    assert settings.submission_top_n == 100


def test_default_embedding_batch_size() -> None:
    settings = load_settings()
    assert settings.embedding_batch_size == 256


# ---------------------------------------------------------------------------
# Paths sub-model — default values and types
# ---------------------------------------------------------------------------


def test_paths_candidates_file_default() -> None:
    settings = load_settings()
    assert settings.paths.candidates_file == Path("data/candidates.jsonl")


def test_paths_job_description_file_default() -> None:
    settings = load_settings()
    assert settings.paths.job_description_file == Path("data/job_description.txt")


def test_paths_output_dir_default() -> None:
    settings = load_settings()
    assert settings.paths.output_dir == Path("output")


def test_paths_submission_file_default() -> None:
    settings = load_settings()
    assert settings.paths.submission_file == Path("output/submission.csv")


def test_paths_embeddings_cache_file_default() -> None:
    settings = load_settings()
    assert settings.paths.embeddings_cache_file == Path("output/embeddings.npy")


def test_paths_bm25_index_file_default() -> None:
    settings = load_settings()
    assert settings.paths.bm25_index_file == Path("output/bm25_index.pkl")


def test_paths_log_dir_default() -> None:
    settings = load_settings()
    assert settings.paths.log_dir == Path("logs")


def test_all_path_fields_are_pathlib_path_instances() -> None:
    """Every path field must be a pathlib.Path, never a raw string."""
    paths = load_settings().paths
    assert isinstance(paths.candidates_file, Path)
    assert isinstance(paths.job_description_file, Path)
    assert isinstance(paths.output_dir, Path)
    assert isinstance(paths.submission_file, Path)
    assert isinstance(paths.embeddings_cache_file, Path)
    assert isinstance(paths.bm25_index_file, Path)
    assert isinstance(paths.log_dir, Path)


# ---------------------------------------------------------------------------
# ScoringWeights — intentionally empty in Phase 1
# ---------------------------------------------------------------------------


def test_scoring_weights_instantiates_with_no_fields() -> None:
    """ScoringWeights is a valid, empty model — no fields to set or validate."""
    sw = ScoringWeights()
    assert isinstance(sw, ScoringWeights)


def test_scoring_weights_has_no_unexpected_fields() -> None:
    """Phase 1: ScoringWeights exposes no public fields yet."""
    sw = ScoringWeights()
    fields = ScoringWeights.model_fields
    assert len(fields) == 0, (
        f"ScoringWeights should be empty in Phase 1 but has fields: {list(fields)}"
    )


# ---------------------------------------------------------------------------
# Numeric constraint enforcement
# ---------------------------------------------------------------------------


def test_retrieval_top_k_below_minimum_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(retrieval_top_k=99)
    assert "retrieval_top_k" in str(exc_info.value)


def test_rrf_k_below_minimum_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(rrf_k=0)
    assert "rrf_k" in str(exc_info.value)


def test_embedding_batch_size_below_minimum_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(embedding_batch_size=0)
    assert "embedding_batch_size" in str(exc_info.value)


def test_submission_top_n_above_maximum_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(submission_top_n=101)
    assert "submission_top_n" in str(exc_info.value)


def test_submission_top_n_below_minimum_raises() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(submission_top_n=0)
    assert "submission_top_n" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


def test_env_override_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    settings = load_settings()
    assert settings.embedding_model == "BAAI/bge-base-en-v1.5"


def test_env_override_force_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_FORCE_RECOMPUTE", "true")
    settings = load_settings()
    assert settings.force_recompute is True


def test_env_override_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_LOG_LEVEL", "DEBUG")
    settings = load_settings()
    assert settings.log_level == "DEBUG"


def test_env_override_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_RANDOM_SEED", "7")
    settings = load_settings()
    assert settings.random_seed == 7


def test_env_override_submission_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """submission_top_n can be lowered for integration tests via env var."""
    monkeypatch.setenv("REDROB_SUBMISSION_TOP_N", "10")
    settings = load_settings()
    assert settings.submission_top_n == 10


def test_env_override_candidates_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_CANDIDATES_FILE", "/tmp/candidates.jsonl")
    settings = load_settings()
    assert settings.paths.candidates_file == Path("/tmp/candidates.jsonl")


def test_env_override_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDROB_OUTPUT_DIR", "/tmp/redrob_output")
    settings = load_settings()
    assert settings.paths.output_dir == Path("/tmp/redrob_output")


# ---------------------------------------------------------------------------
# Explicit env_file argument to load_settings()
# ---------------------------------------------------------------------------


def test_load_settings_with_explicit_env_file(tmp_path: Path) -> None:
    """load_settings(env_file=...) reads from the given file, not from .env."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "REDROB_LOG_LEVEL=WARNING\n"
        "REDROB_RANDOM_SEED=99\n"
        "REDROB_FORCE_RECOMPUTE=true\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_file)
    assert settings.log_level == "WARNING"
    assert settings.random_seed == 99
    assert settings.force_recompute is True


def test_load_settings_explicit_env_file_overrides_path(tmp_path: Path) -> None:
    """Path fields in Paths sub-model are also overridden by an explicit env_file."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "REDROB_CANDIDATES_FILE=/data/full_candidates.jsonl\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_file)
    assert settings.paths.candidates_file == Path("/data/full_candidates.jsonl")


def test_load_settings_nonexistent_env_file_uses_defaults(tmp_path: Path) -> None:
    """Pointing load_settings() at a nonexistent .env file falls back to defaults
    rather than raising — consistent with pydantic-settings behaviour."""
    missing = tmp_path / "does_not_exist.env"
    settings = load_settings(env_file=missing)
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.random_seed == 42


# ---------------------------------------------------------------------------
# Import-time safety
# ---------------------------------------------------------------------------


def test_importing_config_does_not_mutate_os_environ() -> None:
    """Importing core.config must not add or remove keys from os.environ."""
    before = set(os.environ.keys())
    import importlib

    import core.config  # noqa: F401 — intentional re-import check
    importlib.reload(core.config)
    after = set(os.environ.keys())
    assert before == after, (
        f"import/reload of core.config mutated os.environ: "
        f"added={after - before}, removed={before - after}"
    )