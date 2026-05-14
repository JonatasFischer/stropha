"""Generic AST chunker built on tree-sitter-language-pack's `process()`.

Strategy (spec §3.3.1 hierarchical):
- Each top-level container (Class/Interface/Enum/Struct/Module) → meso chunk.
- Each function / method inside a container or at module level → micro chunk.
- Each micro chunk references its container via `parent_chunk_id` and `parent_symbol`.

We skip emitting a container body chunk if it is identical to the file (i.e.
the file is a single class) — would only duplicate content with the methods.

Files smaller than `SIMPLE_FILE_LINES` lines just emit one file-level chunk
(no point in AST decomposition for utility files).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import tree_sitter_language_pack as tslp

from ...logging import get_logger
from ...models import Chunk, SourceFile
from .base import (
    MAX_CHARS_PER_CHUNK,
    MIN_CHARS_PER_CHUNK,
    count_lines,
    make_chunk_id,
    sha256_hex,
    slice_lines,
)

log = get_logger(__name__)

# Below this line count, just emit one file-level chunk.
SIMPLE_FILE_LINES = 15

# StructureKind values we treat as "containers" (emit a container chunk + recurse).
_CONTAINER_KINDS = {
    "Class", "Interface", "Enum", "Struct", "Trait", "Impl", "Module",
    "Namespace", "Object", "Protocol", "Record", "Union",
}

# StructureKind values we treat as "leaves" (emit one micro chunk, no recursion).
_LEAF_KINDS = {
    "Function", "Method", "Constructor", "Getter", "Setter",
    "AssociatedFunction",
}


def _kind_str(item: Any) -> str:
    """Normalize StructureKind enum to a comparable string."""
    return str(item.kind).split(".")[-1]


def _qualified_name(stack: list[str], name: str | None) -> str | None:
    if not name:
        return None
    return ".".join([*stack, name]) if stack else name


class AstGenericChunker:
    """Uses tslp.process() for languages with structure extraction support."""

    def __init__(self, language: str) -> None:
        self._language = language

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        line_total = count_lines(content)
        if line_total <= SIMPLE_FILE_LINES or len(content) < MIN_CHARS_PER_CHUNK:
            yield from self._file_chunk(file, content, line_total)
            return

        try:
            result = tslp.process(
                content,
                tslp.ProcessConfig(
                    language=self._language,
                    structure=True,
                    imports=False,
                    exports=False,
                    symbols=False,
                ),
            )
        except Exception as exc:
            log.warning(
                "ast_chunker.fallback",
                language=self._language,
                path=file.rel_path,
                error=str(exc),
            )
            yield from self._file_chunk(file, content, line_total)
            return

        if not result.structure:
            yield from self._file_chunk(file, content, line_total)
            return

        # Walk structure: emit container chunks + leaf chunks; track parent.
        emitted = 0
        for item in result.structure:
            for chunk in self._walk(file, content, item, parent_stack=[], parent_chunk_id=None):
                emitted += 1
                yield chunk

        if emitted == 0:
            yield from self._file_chunk(file, content, line_total)

    # ---- internals ----

    def _file_chunk(self, file: SourceFile, content: str, line_total: int) -> Iterator[Chunk]:
        """Emit a single file-level chunk (or split if oversized)."""
        if len(content) <= MAX_CHARS_PER_CHUNK:
            h = sha256_hex(content)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, 1, line_total, h),
                rel_path=file.rel_path,
                language=file.language,
                kind="file",
                symbol=None,
                start_line=1,
                end_line=line_total,
                content=content,
                content_hash=h,
            )
        else:
            yield from self._slice_split(file, content, 1, line_total, kind="file_part")

    def _slice_split(
        self,
        file: SourceFile,
        content: str,
        start_line: int,
        end_line: int,
        *,
        kind: str,
        symbol: str | None = None,
        parent_chunk_id: str | None = None,
    ) -> Iterator[Chunk]:
        """Split an oversized span on line boundaries."""
        lines = content.splitlines(keepends=True)[start_line - 1 : end_line]
        buf: list[str] = []
        buf_len = 0
        s = start_line
        cur = start_line - 1
        for line in lines:
            cur += 1
            if buf_len + len(line) > MAX_CHARS_PER_CHUNK and buf:
                text = "".join(buf)
                h = sha256_hex(text)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, s, cur - 1, h),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind=kind,
                    symbol=symbol,
                    parent_chunk_id=parent_chunk_id,
                    start_line=s,
                    end_line=cur - 1,
                    content=text,
                    content_hash=h,
                )
                buf = []
                buf_len = 0
                s = cur
            buf.append(line)
            buf_len += len(line)
        if buf:
            text = "".join(buf)
            h = sha256_hex(text)
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, s, cur, h),
                rel_path=file.rel_path,
                language=file.language,
                kind=kind,
                symbol=symbol,
                parent_chunk_id=parent_chunk_id,
                start_line=s,
                end_line=cur,
                content=text,
                content_hash=h,
            )

    def _walk(
        self,
        file: SourceFile,
        content: str,
        item: Any,
        *,
        parent_stack: list[str],
        parent_chunk_id: str | None,
    ) -> Iterator[Chunk]:
        kind = _kind_str(item)
        span = item.span
        if span is None:
            return
        start = max(1, int(span.start_line) + 1)
        end = max(start, int(span.end_line) + 1)
        name = item.name
        qualified = _qualified_name(parent_stack, name)
        body = slice_lines(content, start, end).rstrip("\n") + "\n"

        if kind in _LEAF_KINDS:
            if len(body.strip()) < MIN_CHARS_PER_CHUNK:
                return
            chunk_kind = "method" if kind in {"Method", "Constructor", "Getter", "Setter"} else "function"
            if len(body) <= MAX_CHARS_PER_CHUNK:
                h = sha256_hex(body)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, start, end, h),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind=chunk_kind,
                    symbol=qualified,
                    parent_chunk_id=parent_chunk_id,
                    start_line=start,
                    end_line=end,
                    content=body,
                    content_hash=h,
                )
            else:
                yield from self._slice_split(
                    file, content, start, end,
                    kind=f"{chunk_kind}_part",
                    symbol=qualified,
                    parent_chunk_id=parent_chunk_id,
                )
            return

        if kind in _CONTAINER_KINDS:
            container_kind = {
                "Class": "class", "Interface": "interface", "Enum": "enum",
                "Struct": "struct", "Trait": "trait", "Impl": "impl",
                "Module": "module", "Namespace": "namespace", "Object": "object",
                "Protocol": "protocol", "Record": "record", "Union": "union",
            }.get(kind, "container")

            container_chunk_id: str | None = None
            # Decide whether to emit a container "skeleton" chunk. We skip Module
            # entirely (it just wraps the file). For classes/interfaces/etc. we
            # emit a compact summary that ALWAYS includes the qualified name,
            # so BM25 can match the type by identifier even when the body lives
            # in separate method chunks.
            if kind != "Module" and qualified is not None:
                signature = (getattr(item, "signature", None) or "").strip()
                doc = (getattr(item, "doc_comment", None) or "").strip()
                child_names = [
                    f"  {_kind_str(c).lower()} {c.name}"
                    for c in (item.children or [])
                    if c.name
                ]
                header = f"{container_kind} {qualified}"
                parts: list[str] = []
                if doc:
                    parts.append(doc)
                parts.append(header)
                if signature and signature != header:
                    parts.append(signature)
                if child_names:
                    parts.append("Members:\n" + "\n".join(child_names))
                skeleton = "\n".join(parts).strip()

                if len(skeleton) >= MIN_CHARS_PER_CHUNK:
                    h = sha256_hex(skeleton)
                    container_chunk_id = make_chunk_id(file.rel_path, start, end, h)
                    yield Chunk(
                        chunk_id=container_chunk_id,
                        rel_path=file.rel_path,
                        language=file.language,
                        kind=container_kind,
                        symbol=qualified,
                        parent_chunk_id=parent_chunk_id,
                        start_line=start,
                        end_line=end,
                        content=skeleton + "\n",
                        content_hash=h,
                    )

            new_stack = parent_stack + ([name] if name else [])
            for child in item.children or []:
                yield from self._walk(
                    file, content, child,
                    parent_stack=new_stack,
                    parent_chunk_id=container_chunk_id or parent_chunk_id,
                )
            return

        # Unknown kind — recurse into children but do not emit a chunk for `item`.
        new_stack = parent_stack + ([name] if name else [])
        for child in item.children or []:
            yield from self._walk(
                file, content, child,
                parent_stack=new_stack,
                parent_chunk_id=parent_chunk_id,
            )
