"""Gherkin chunker (custom — tree-sitter-language-pack does not ship a grammar).

Per spec §3.3.1: one chunk per Scenario / Scenario Outline. A parent chunk per
Feature carries the Background so context is preserved when individual
scenarios are returned.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ...models import Chunk, SourceFile
from .base import MIN_CHARS_PER_CHUNK, count_lines, make_chunk_id, sha256_hex

_FEATURE_RE = re.compile(r"^\s*Feature:\s*(.+?)\s*$", re.MULTILINE)
_SCENARIO_RE = re.compile(
    r"^\s*(Scenario(?: Outline)?|Background|Rule|Example):\s*(.*?)\s*$",
    re.MULTILINE,
)


class GherkinChunker:
    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        total_lines = count_lines(content)
        lines = content.splitlines(keepends=True)
        feature_match = _FEATURE_RE.search(content)
        feature_name = (feature_match.group(1).strip() if feature_match else file.path.stem)

        # Find all section starts.
        starts: list[tuple[int, str, str]] = []  # (line_1based, kind, title)
        for m in _SCENARIO_RE.finditer(content):
            line = content.count("\n", 0, m.start()) + 1
            kind_raw = m.group(1).strip()
            title = (m.group(2) or "").strip()
            starts.append((line, kind_raw, title))

        # Feature-level chunk: from line 1 to start of first section (or EOF).
        first_section_line = starts[0][0] if starts else total_lines + 1
        feature_text = "".join(lines[: max(0, first_section_line - 1)])
        if len(feature_text.strip()) >= MIN_CHARS_PER_CHUNK:
            h = sha256_hex(feature_text)
            feature_chunk_id: str | None = make_chunk_id(
                file.rel_path, 1, max(1, first_section_line - 1), h
            )
            yield Chunk(
                chunk_id=feature_chunk_id,
                rel_path=file.rel_path,
                language=file.language,
                kind="feature",
                symbol=feature_name,
                start_line=1,
                end_line=max(1, first_section_line - 1),
                content=feature_text,
                content_hash=h,
            )
        else:
            feature_chunk_id = None

        # Scenario / Background / Rule chunks.
        for idx, (start_line, kind_raw, title) in enumerate(starts):
            end_line = (
                starts[idx + 1][0] - 1 if idx + 1 < len(starts) else total_lines
            )
            text = "".join(lines[start_line - 1 : end_line])
            if len(text.strip()) < MIN_CHARS_PER_CHUNK:
                continue
            h = sha256_hex(text)
            kind = {
                "Scenario": "scenario",
                "Scenario Outline": "scenario_outline",
                "Background": "background",
                "Rule": "rule",
                "Example": "scenario",
            }.get(kind_raw, "scenario")
            symbol = f"{feature_name} :: {title}" if title else feature_name
            yield Chunk(
                chunk_id=make_chunk_id(file.rel_path, start_line, end_line, h),
                rel_path=file.rel_path,
                language=file.language,
                kind=kind,
                symbol=symbol,
                parent_chunk_id=feature_chunk_id,
                start_line=start_line,
                end_line=end_line,
                content=text,
                content_hash=h,
            )
