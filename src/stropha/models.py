"""Shared pydantic models used across modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SourceFile(BaseModel):
    """A file discovered by the walker, ready to be chunked."""

    path: Path
    rel_path: str
    language: str
    size_bytes: int


@dataclass(frozen=True)
class FileDelta:
    """One file-level change emitted by the incremental walker (Phase B).

    Maps to the four ``git diff --name-status`` actions:

    - ``add``     — file appeared since ``since_sha``
    - ``modify``  — file existed and was edited
    - ``delete``  — file removed from the working tree
    - ``rename``  — file renamed (possibly with edits — ``similarity`` is
      the git-reported percentage; identical-content renames are 100)

    ``rel_path`` is the path *as of HEAD* (i.e. post-rename). For renames
    ``old_rel_path`` is the pre-rename path so the pipeline can
    ``rename_chunks(old, new)`` instead of re-embedding.
    """

    action: Literal["add", "modify", "delete", "rename"]
    rel_path: str
    old_rel_path: str | None = None
    similarity: int | None = None


class Chunk(BaseModel):
    """An indexable unit. Phase 0 = whole file. Phase 1 adds AST kinds.

    Phase 1 (pipeline-adapters) augments the row with the enricher's output
    (``embedding_text``) and the adapter id that produced it
    (``enricher_id``) so the pipeline can detect drift (config change →
    re-enrich + re-embed automatically, no ``--rebuild`` needed).
    """

    chunk_id: str  # sha256 of (rel_path + start_line + end_line + content_hash)
    rel_path: str
    language: str
    kind: str = "file"  # file | class | method | component | scenario | section
    symbol: str | None = None
    parent_chunk_id: str | None = None
    start_line: int
    end_line: int
    content: str
    content_hash: str  # sha256 of content
    # Embedding metadata is set after the embedder runs.
    embedding_model: str | None = None
    embedding_dim: int | None = None
    # Enricher metadata is set after the enricher runs (pipeline-adapters Phase 1).
    embedding_text: str | None = None
    """Exact text that was fed to the embedder. Equals ``content`` for
    ``enricher=noop``. Persisted so drift detection can compare against
    a fresh enricher's output."""
    enricher_id: str | None = None
    """``adapter_id`` of the enricher that produced ``embedding_text``."""


class RepoRef(BaseModel):
    """Stable identifier of the git repository a chunk belongs to.

    Returned alongside every ``SearchHit`` so an MCP client can run
    ``git clone <url>`` to retrieve the source. See
    ``ingest/git_meta.py:RepoIdentity`` for the producer.
    """

    normalized_key: str = Field(
        description=(
            "Stable cross-user key (lowercased host/path, no scheme/auth/.git). "
            "'local:<path>' for git repos without a remote; "
            "'path:<path>' for non-git directories."
        )
    )
    url: str | None = Field(
        default=None,
        description="HTTPS URL suitable for `git clone`. None for local-only.",
    )
    default_branch: str | None = Field(
        default=None,
        description="Default branch for checkout (e.g. 'main').",
    )
    head_commit: str | None = Field(
        default=None,
        description="HEAD SHA at last index time.",
    )


class SearchHit(BaseModel):
    """A single result from retrieval."""

    rank: int
    score: float
    rel_path: str
    language: str
    kind: str
    symbol: str | None = None
    start_line: int
    end_line: int
    snippet: str = Field(description="Truncated content preview.")
    chunk_id: str
    repo: RepoRef | None = Field(
        default=None,
        description="Source repository identity. None for legacy chunks "
        "indexed before schema v2.",
    )


class RepoStats(BaseModel):
    """Per-repo counters reported by Storage.stats() and `list_repos`."""

    normalized_key: str
    url: str | None
    default_branch: str | None
    head_commit: str | None
    files: int
    chunks: int
    last_indexed_at: str | None
