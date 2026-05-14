"""Structured exceptions. Per CLAUDE.md, never raise bare Exception."""

from __future__ import annotations


class RagError(Exception):
    """Base class for all errors raised by mimoria-rag."""


class ConfigError(RagError):
    """Invalid or missing configuration."""


class WalkerError(RagError):
    """Failure while discovering source files."""


class ChunkerError(RagError):
    """Failure while splitting a file into chunks."""


class EmbeddingError(RagError):
    """Failure while computing embeddings (network, auth, model)."""


class StorageError(RagError):
    """Failure in the SQLite/vec layer (schema, IO, corruption)."""


class RetrievalError(RagError):
    """Failure during query execution."""
