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
    """An indexable unit. Phase 0 = whole file. Phase 1 will add AST kinds."""

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
