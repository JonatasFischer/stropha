"""Markdown chunker: split by ATX/setext headings.

Per spec §3.3.1: each section H2/H3 is a chunk. The full document is treated
as the macro level (we do not currently emit a separate file chunk; the
section chunks together cover it).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ...models import Chunk, SourceFile
from .base import (
    MAX_CHARS_PER_CHUNK,
    MIN_CHARS_PER_CHUNK,
    count_lines,
    make_chunk_id,
    sha256_hex,
)

# Match ATX headings: # … through ###### …. Captures level + title.
_ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _split_sections(content: str) -> list[tuple[int, int, int, str]]:
    """Return [(start_line, end_line, level, title), ...] for every heading block.

    Each section spans from its heading line to the line BEFORE the next heading
    of equal-or-shallower level, or EOF.
    """
    lines = content.splitlines()
    # Collect heading positions: (line_idx_0based, level, title)
    headings: list[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _ATX_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((i, level, title))

    sections: list[tuple[int, int, int, str]] = []
    for idx, (line_idx, level, title) in enumerate(headings):
        # End at line before next heading of level <= current; clamp to EOF.
        end_idx = len(lines)
        for j in range(idx + 1, len(headings)):
            next_line_idx, next_level, _ = headings[j]
            if next_level <= level:
                end_idx = next_line_idx
                break
        sections.append((line_idx + 1, end_idx, level, title))
    return sections


class MarkdownChunker:
    """Section-per-chunk. Parent chain encoded via parent_chunk_id."""

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        total_lines = count_lines(content)
        sections = _split_sections(content)

        if not sections:
            # No headings — single file chunk.
            if len(content.strip()) < MIN_CHARS_PER_CHUNK:
                return
            h = sha256_hex(content)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, 1, total_lines, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="document",
                start_line=1,
                end_line=total_lines,
                content=content,
                content_hash=h,
            )
            return

        # Map level → most recent ancestor chunk_id.
        ancestors: dict[int, str] = {}
        lines = content.splitlines(keepends=True)

        # Optional prelude: content before the first heading.
        first_start = sections[0][0]
        if first_start > 1:
            prelude_text = "".join(lines[: first_start - 1]).strip()
            if len(prelude_text) >= MIN_CHARS_PER_CHUNK:
                h = sha256_hex(prelude_text)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, 1, first_start - 1, h),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind="section",
                    symbol="(prelude)",
                    start_line=1,
                    end_line=first_start - 1,
                    content=prelude_text + "\n",
                    content_hash=h,
                )

        for start_line, end_line, level, title in sections:
            text = "".join(lines[start_line - 1 : end_line])
            if len(text.strip()) < MIN_CHARS_PER_CHUNK:
                continue
            parent_id = next(
                (ancestors[lvl] for lvl in sorted(ancestors) if lvl < level),
                None,
            )
            # Reset deeper ancestors.
            for lvl in [k for k in ancestors if k >= level]:
                del ancestors[lvl]

            if len(text) <= MAX_CHARS_PER_CHUNK:
                h = sha256_hex(text)
                cid = make_chunk_id(file.rel_path, start_line, end_line, h)
                ancestors[level] = cid
                yield Chunk(
                    chunk_id=cid,
                    rel_path=file.rel_path,
                    language=file.language,
                    kind="section",
                    symbol=title,
                    parent_chunk_id=parent_id,
                    start_line=start_line,
                    end_line=end_line,
                    content=text,
                    content_hash=h,
                )
            else:
                # Oversized section: split on blank-line boundaries.
                for sub in _line_split(text, start_line):
                    sub_text, sub_start, sub_end = sub
                    h = sha256_hex(sub_text)
                    cid = make_chunk_id(file.rel_path, sub_start, sub_end, h)
                    yield Chunk(
                        chunk_id=cid,
                        rel_path=file.rel_path,
                        language=file.language,
                        kind="section_part",
                        symbol=title,
                        parent_chunk_id=parent_id,
                        start_line=sub_start,
                        end_line=sub_end,
                        content=sub_text,
                        content_hash=h,
                    )
                # No ancestor registered for split section (would orphan children).


def _line_split(
    text: str, base_line: int
) -> Iterator[tuple[str, int, int]]:
    """Yield (chunk_text, start_line, end_line) splitting on blank lines."""
    lines = text.splitlines(keepends=True)
    buf: list[str] = []
    buf_len = 0
    s = base_line
    cur = base_line - 1
    for line in lines:
        cur += 1
        if buf_len + len(line) > MAX_CHARS_PER_CHUNK and buf:
            yield "".join(buf), s, cur - 1
            buf = []
            buf_len = 0
            s = cur
        buf.append(line)
        buf_len += len(line)
    if buf:
        yield "".join(buf), s, cur
