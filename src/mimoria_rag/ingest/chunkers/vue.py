"""Vue SFC chunker — splits a single-file component into script/template/style.

Per spec §3.3.1: three chunks per Vue SFC (script, template, style). For
components smaller than ~150 total lines, we emit a single chunk because the
embedding gain from splitting is dwarfed by extra storage.
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

_BLOCK_RE = re.compile(
    r"<(script|template|style)\b([^>]*)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)

SFC_SPLIT_THRESHOLD_LINES = 150


def _line_of_offset(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


class VueChunker:
    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        total_lines = count_lines(content)
        # Component name from filename (e.g. StudyView.vue → "StudyView").
        component = file.path.stem

        if total_lines <= SFC_SPLIT_THRESHOLD_LINES:
            if len(content.strip()) < MIN_CHARS_PER_CHUNK:
                return
            h = sha256_hex(content)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, 1, total_lines, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="component",
                symbol=component,
                start_line=1,
                end_line=total_lines,
                content=content,
                content_hash=h,
            )
            return

        emitted = 0
        for match in _BLOCK_RE.finditer(content):
            tag = match.group(1).lower()
            attrs = match.group(2)
            inner = match.group(3)
            if len(inner.strip()) < MIN_CHARS_PER_CHUNK:
                continue
            start_line = _line_of_offset(content, match.start())
            end_line = _line_of_offset(content, match.end())
            full = match.group(0)  # include the wrapping tags so context is preserved
            kind_map = {"script": "component_script", "template": "component_template", "style": "component_style"}
            kind = kind_map.get(tag, "component_block")
            # Annotate symbol with sub-block for clarity in search hits.
            symbol = f"{component}.{tag}"
            if "setup" in attrs.lower():
                symbol = f"{component}.{tag}.setup"

            if len(full) <= MAX_CHARS_PER_CHUNK:
                h = sha256_hex(full)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, start_line, end_line, h),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind=kind,
                    symbol=symbol,
                    start_line=start_line,
                    end_line=end_line,
                    content=full,
                    content_hash=h,
                )
                emitted += 1
            else:
                # Oversized block — line-based split.
                yield from _split_block(file, full, start_line, kind, symbol)
                emitted += 1

        if emitted == 0:
            # No matchable blocks — fall back to a single file chunk so we
            # never drop a file silently.
            h = sha256_hex(content)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, 1, total_lines, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="component",
                symbol=component,
                start_line=1,
                end_line=total_lines,
                content=content,
                content_hash=h,
            )


def _split_block(
    file: SourceFile, text: str, base_line: int, kind: str, symbol: str
) -> Iterator[Chunk]:
    lines = text.splitlines(keepends=True)
    buf: list[str] = []
    buf_len = 0
    s = base_line
    cur = base_line - 1
    for line in lines:
        cur += 1
        if buf_len + len(line) > MAX_CHARS_PER_CHUNK and buf:
            chunk_text = "".join(buf)
            h = sha256_hex(chunk_text)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, s, cur - 1, h),
                rel_path=file.rel_path,
                language=file.language,
                kind=f"{kind}_part",
                symbol=symbol,
                start_line=s,
                end_line=cur - 1,
                content=chunk_text,
                content_hash=h,
            )
            buf = []
            buf_len = 0
            s = cur
        buf.append(line)
        buf_len += len(line)
    if buf:
        chunk_text = "".join(buf)
        h = sha256_hex(chunk_text)
        yield Chunk(
            chunk_id=make_chunk_id(file.rel_path, s, cur, h),
            rel_path=file.rel_path,
            language=file.language,
            kind=f"{kind}_part",
            symbol=symbol,
            start_line=s,
            end_line=cur,
            content=chunk_text,
            content_hash=h,
        )
