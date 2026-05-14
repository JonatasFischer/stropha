"""Shared pydantic models used across modules."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SourceFile(BaseModel):
    """A file discovered by the walker, ready to be chunked."""

    path: Path
    rel_path: str
    language: str
    size_bytes: int


class Chunk(BaseModel):
    """An indexable unit. Phase 0 = whole file. Phase 1 adds AST kinds."""

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
