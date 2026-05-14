"""File-level chunker. Used when no AST chunker is registered for a language."""

from __future__ import annotations

from collections.abc import Iterator

from ...models import Chunk, SourceFile
from .base import (
    MAX_CHARS_PER_CHUNK,
    MIN_CHARS_PER_CHUNK,
    count_lines,
    make_chunk_id,
    sha256_hex,
)


class FallbackChunker:
    """One chunk per file; line-based split when file exceeds MAX_CHARS_PER_CHUNK."""

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        if len(content.strip()) < MIN_CHARS_PER_CHUNK:
            return
        line_count = count_lines(content)
        if len(content) <= MAX_CHARS_PER_CHUNK:
            h = sha256_hex(content)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, 1, line_count, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="file",
                start_line=1,
                end_line=line_count,
                content=content,
                content_hash=h,
            )
            return
        yield from self._split_oversized(file, content)

    def _split_oversized(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        lines = content.splitlines(keepends=True)
        buf: list[str] = []
        buf_len = 0
        start_line = 1
        cur_line = 0
        for line in lines:
            cur_line += 1
            if buf_len + len(line) > MAX_CHARS_PER_CHUNK and buf:
                text = "".join(buf)
                h = sha256_hex(text)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, start_line, cur_line - 1, h),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind="file_part",
                    start_line=start_line,
                    end_line=cur_line - 1,
                    content=text,
                    content_hash=h,
                )
                buf = []
                buf_len = 0
                start_line = cur_line
            buf.append(line)
            buf_len += len(line)
        if buf:
            text = "".join(buf)
            h = sha256_hex(text)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, start_line, cur_line, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="file_part",
                start_line=start_line,
                end_line=cur_line,
                content=text,
                content_hash=h,
            )
