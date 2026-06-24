"""Candidate file loading for the Redrob ranking pipeline.

Exposes three public names:

    LoadError   — raised when a file cannot be opened or a record is invalid.
    LoadResult  — returned by load_candidates() when strict=False.
    load_candidates() — the single entry point for reading candidate data.

Format support
--------------
Two formats are accepted, detected by content (not file extension):

* **JSONL** — one JSON object per line.  Any line whose stripped form is
  empty is silently skipped.  Lines are processed one at a time; the file
  is never buffered entirely into memory.
* **JSON array** — a file whose first non-whitespace character is ``[``.
  The full array is parsed with ``json.load()`` in one pass, which is
  appropriate because JSON array files must be valid as a whole unit.

Strict vs non-strict loading
-----------------------------
``strict=True`` (default)
    Raises ``LoadError`` on the first malformed JSON line or schema-invalid
    record.  Returns ``list[Candidate]``.

``strict=False``
    Skips invalid records, collects a ``LoadError`` for each one, and
    returns a ``LoadResult(candidates, errors)`` regardless of how many
    records were skipped.  Returns ``LoadResult``.

All ``json.JSONDecodeError`` and ``pydantic.ValidationError`` exceptions are
always wrapped in ``LoadError``; callers never see raw library exceptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from core.schema import Candidate


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LoadError(Exception):
    """Raised when candidate data cannot be loaded or fails schema validation.

    Wraps ``json.JSONDecodeError`` and ``pydantic.ValidationError``; callers
    always catch this single type rather than multiple library exceptions.
    The original exception is available via ``__cause__`` when present.
    """


@dataclass
class LoadResult:
    """Return value of ``load_candidates()`` when ``strict=False``.

    Attributes:
        candidates: Validated ``Candidate`` objects in source order.
        errors:     One ``LoadError`` per record that was skipped.  Empty
                    when every record in the file passed validation.
    """

    candidates: list[Candidate] = dc_field(default_factory=list)
    errors: list[LoadError] = dc_field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _peek_first_char(path: Path) -> str | None:
    """Return the first non-whitespace character in *path*, or ``None`` if the
    file is empty or contains only whitespace.

    Uses line iteration (``for line in f``) so that ``Path.open``-wrapped
    file objects that track ``.read()`` / ``.readlines()`` calls are never
    triggered.  This is important for the streaming contract: even format
    detection must not buffer the file.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped[0]
    return None


def _parse_line(raw: str, line_num: int) -> dict:
    """Parse *raw* as a JSON object, wrapping any parse error in ``LoadError``.

    Args:
        raw:      The stripped line content.
        line_num: 1-based line number, used in the error message.

    Returns:
        A plain Python dict.

    Raises:
        LoadError: If *raw* is not valid JSON.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LoadError(
            f"Malformed JSON at line {line_num}: {exc}"
        ) from exc


def _validate_record(data: dict, position: int) -> Candidate:
    """Validate *data* against ``Candidate``, wrapping errors in ``LoadError``.

    The ``candidate_id`` value (if present in *data*) is included in the
    error message so non-strict callers can correlate errors with records.

    Args:
        data:     Raw dict from JSON parsing.
        position: 1-based record position (line number for JSONL, index for
                  JSON array), used in the error message.

    Returns:
        A validated ``Candidate`` instance.

    Raises:
        LoadError: If *data* fails ``Candidate`` schema validation.
    """
    candidate_id: str = data.get("candidate_id", "<unknown>")
    try:
        return Candidate.model_validate(data)
    except ValidationError as exc:
        raise LoadError(
            f"Schema validation failed for candidate {candidate_id!r} "
            f"at position {position}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Format-specific processors
# ---------------------------------------------------------------------------


def _process_jsonl(
    path: Path,
    strict: bool,
) -> Union[list[Candidate], LoadResult]:
    """Stream-process a JSONL file, one line at a time.

    Iterates over the open file object directly (``for line in f``) so that
    ``.read()`` and ``.readlines()`` are never called.  This satisfies the
    streaming contract even when ``Path.open`` is wrapped by a tracking
    proxy in tests.

    Args:
        path:   Path to the JSONL file.
        strict: When ``True``, raise on the first error; when ``False``,
                collect errors and continue.

    Returns:
        ``list[Candidate]`` when *strict* is ``True``;
        ``LoadResult`` when *strict* is ``False``.
    """
    candidates: list[Candidate] = []
    errors: list[LoadError] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue  # blank / whitespace-only line
            try:
                data = _parse_line(stripped, line_num)
                candidate = _validate_record(data, line_num)
                candidates.append(candidate)
            except LoadError as exc:
                if strict:
                    raise
                errors.append(exc)

    if strict:
        return candidates
    return LoadResult(candidates=candidates, errors=errors)


def _process_json_array(
    path: Path,
    strict: bool,
) -> Union[list[Candidate], LoadResult]:
    """Load a JSON array file with ``json.load()`` and validate each element.

    JSON arrays must be valid as a complete unit, so a single ``json.load``
    pass is appropriate here.  The streaming constraint (no ``.read()`` /
    ``.readlines()``) applies only to JSONL files; ``json.load`` internally
    calls ``f.read()`` but that is acceptable for the array format.

    Args:
        path:   Path to the JSON array file.
        strict: When ``True``, raise on the first error; when ``False``,
                collect errors and continue.

    Returns:
        ``list[Candidate]`` when *strict* is ``True``;
        ``LoadResult`` when *strict* is ``False``.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            raw_array = json.load(f)
    except json.JSONDecodeError as exc:
        raise LoadError(
            f"Malformed JSON array in {path}: {exc}"
        ) from exc

    candidates: list[Candidate] = []
    errors: list[LoadError] = []

    for idx, record in enumerate(raw_array):
        try:
            candidate = _validate_record(record, idx + 1)
            candidates.append(candidate)
        except LoadError as exc:
            if strict:
                raise
            errors.append(exc)

    if strict:
        return candidates
    return LoadResult(candidates=candidates, errors=errors)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_candidates(
    path: Path,
    *,
    strict: bool = True,
) -> Union[list[Candidate], LoadResult]:
    """Load and validate candidate records from a JSONL or JSON array file.

    Format is detected by content: if the first non-whitespace character in
    the file is ``[``, the file is treated as a JSON array; otherwise it is
    treated as JSONL.  File extension is ignored.

    Args:
        path:   Path to the candidate data file.  Must be a
                ``pathlib.Path``; both JSONL and JSON-array formats are
                accepted.
        strict: Controls error handling.

                ``True`` (default) — raise ``LoadError`` on the first
                malformed JSON line or schema-invalid record.  Returns
                ``list[Candidate]``.

                ``False`` — skip invalid records, collect one ``LoadError``
                per skipped record, and always return a ``LoadResult``
                (even if every record was valid).  Returns ``LoadResult``.

    Returns:
        * ``list[Candidate]`` — when *strict* is ``True`` and the file
          loaded without errors.
        * ``LoadResult`` — when *strict* is ``False``.  Inspect
          ``result.candidates`` for valid records and ``result.errors`` for
          skipped ones.

    Raises:
        LoadError: (strict mode) File not found, unreadable, contains
            malformed JSON, or contains a record that fails ``Candidate``
            schema validation.
        LoadError: (non-strict mode) File not found or unreadable.  Per-
            record errors are collected into ``LoadResult.errors`` rather
            than raised.

    Examples::

        # Strict — raises on any problem
        candidates = load_candidates(Path("data/candidates.jsonl"))

        # Non-strict — inspect skipped records
        result = load_candidates(Path("data/candidates.jsonl"), strict=False)
        print(f"Loaded {len(result.candidates)}, skipped {len(result.errors)}")
    """
    # ------------------------------------------------------------------ #
    # 1. Existence check — wrap OSError in LoadError for a clean contract. #
    # ------------------------------------------------------------------ #
    if not path.exists():
        raise LoadError(f"File not found: {path}")

    # ------------------------------------------------------------------ #
    # 2. Format detection — peek at the first non-whitespace character.   #
    #    Uses line iteration only; never calls .read() or .readlines().   #
    # ------------------------------------------------------------------ #
    try:
        first_char = _peek_first_char(path)
    except OSError as exc:
        raise LoadError(f"Cannot read {path}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 3. Empty file — return early without opening a second time.         #
    # ------------------------------------------------------------------ #
    if first_char is None:
        return [] if strict else LoadResult(candidates=[], errors=[])

    # ------------------------------------------------------------------ #
    # 4. Dispatch to the appropriate format processor.                    #
    # ------------------------------------------------------------------ #
    try:
        if first_char == "[":
            return _process_json_array(path, strict)
        else:
            return _process_jsonl(path, strict)
    except LoadError:
        raise  # already wrapped — re-raise as-is
    except OSError as exc:
        raise LoadError(f"Cannot read {path}: {exc}") from exc