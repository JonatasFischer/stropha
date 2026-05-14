"""Shared contract + helpers for language-specific chunkers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ...models import Chunk, SourceFile

# Spec §3.3.4: chunks > 8K tokens overflow reranker context. At ~4 chars/token,
# that is ~32 KB. We keep margin so even verbose Java methods + docs fit.
MAX_CHARS_PER_CHUNK = 24_000

# Anything shorter than this is too small to embed usefully (spec §3.3.4).
MIN_CHARS_PER_CHUNK = 20


@runtime_checkable
class LanguageChunker(Protocol):
    """Splits a single file into one or more Chunks."""

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]: ...


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_chunk_id(rel_path: str, start: int, end: int, content_hash: str) -> str:
    """Deterministic chunk id per spec §3.4."""
    payload = f"{rel_path}:{start}-{end}:{content_hash}"
    return f"sha256:{sha256_hex(payload)}"


def slice_lines(content: str, start_line: int, end_line: int) -> str:
    """Return lines `[start_line, end_line]` inclusive, 1-indexed."""
    lines = content.splitlines(keepends=True)
    # Clamp; tree-sitter occasionally returns end_line == len(lines).
    s = max(0, start_line - 1)
    e = min(len(lines), end_line)
    return "".join(lines[s:e])


def count_lines(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)
