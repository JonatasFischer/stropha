"""Cost dashboard (Phase 3 §11.4).

Aggregates the structured-log trail produced by the indexer/server into
a per-adapter / per-repo / per-day summary. No external dependencies —
just walks the log file and builds buckets.

What we count:
- ``pipeline.repo_done`` events → files, chunks_seen, chunks_embedded
  per repo per run.
- ``local_embedder.ready`` / ``pipeline.adapter.built`` → adapter
  identities seen.
- Hook log refresh markers → number of background refresh cycles.
- ``graph_vec_loader.done`` → graph nodes embedded per run.

Because stropha is local-first there is no per-token money cost to
report. The "cost" is computed time: total embeddings, total chunks,
wall-clock estimates derived from event timestamps. That's enough to
spot regressions and budget large indexing runs.

Output formats:
- Rich table (default for terminal) via :func:`render_table`
- JSON via :func:`render_json` for programmatic consumption
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_HOOK_LOG = Path.home() / ".cache" / "stropha-hook.log"


# --------------------------------------------------------------------- model


@dataclass
class RepoCost:
    repo: str
    runs: int = 0
    files_visited: int = 0
    chunks_seen: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    enricher_cache_hits: int = 0
    last_event: str | None = None  # ISO timestamp


@dataclass
class AdapterCost:
    adapter_id: str
    stage: str
    build_count: int = 0


@dataclass
class GraphCost:
    nodes_loaded: int = 0
    edges_loaded: int = 0
    edges_filtered: int = 0
    embed_runs: int = 0
    embedded_nodes: int = 0
    last_load: str | None = None


@dataclass
class HookCost:
    refresh_launched: int = 0
    refresh_done: int = 0
    refresh_failed: int = 0
    graphify_updates: int = 0


@dataclass
class CostReport:
    repos: dict[str, RepoCost] = field(default_factory=dict)
    adapters: dict[str, AdapterCost] = field(default_factory=dict)
    graph: GraphCost = field(default_factory=GraphCost)
    hook: HookCost = field(default_factory=HookCost)
    sources_scanned: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repos": {k: vars(v) for k, v in sorted(self.repos.items())},
            "adapters": {k: vars(v) for k, v in sorted(self.adapters.items())},
            "graph": vars(self.graph),
            "hook": vars(self.hook),
            "sources_scanned": self.sources_scanned,
        }


# --------------------------------------------------------------------- parser


def _iter_log_lines(paths: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    for p in paths:
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                yield p, line.strip()
        except OSError:
            continue


def _try_json(line: str) -> dict[str, Any] | None:
    if not line.startswith("{"):
        return None
    try:
        d = json.loads(line)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def aggregate(paths: Iterable[Path]) -> CostReport:
    """Walk the given log paths and return aggregated counters."""
    report = CostReport()
    for src, line in _iter_log_lines(paths):
        if str(src) not in report.sources_scanned:
            report.sources_scanned.append(str(src))

        # Hook log entries are plain text, not JSON. Match by substring
        # because the "launching refresh" line is the date-prefixed banner
        # (no `stropha-hook` marker), while the other lines use the
        # `[stropha-hook]` prefix from the hook script.
        if not line.startswith("{"):
            matched = False
            if "launching refresh" in line:
                report.hook.refresh_launched += 1
                matched = True
            elif "[stropha-hook] refresh done" in line:
                report.hook.refresh_done += 1
                matched = True
            elif "[stropha-hook] failed" in line or "[stropha-hook] graphify update failed" in line:
                report.hook.refresh_failed += 1
                matched = True
            if "graphify update (commit" in line:
                report.hook.graphify_updates += 1
                matched = True
            if matched:
                continue

        d = _try_json(line)
        if d is None:
            continue
        event = d.get("event")
        if event == "pipeline.repo_done":
            repo = str(d.get("repo") or "(unknown)")
            rc = report.repos.setdefault(repo, RepoCost(repo=repo))
            rc.runs += 1
            rc.files_visited += int(d.get("files") or 0)
            rc.chunks_seen += int(d.get("chunks") or 0)
            rc.chunks_embedded += int(d.get("embedded") or 0)
            rc.chunks_reused += int(d.get("reused") or 0)
            rc.enricher_cache_hits += int(d.get("enricher_cache_hits") or 0)
            ts = d.get("timestamp")
            if ts and (rc.last_event is None or ts > rc.last_event):
                rc.last_event = ts
        elif event == "pipeline.adapter.built":
            aid = str(d.get("adapter_id") or "?")
            stage = str(d.get("stage") or "?")
            ac = report.adapters.setdefault(aid, AdapterCost(adapter_id=aid, stage=stage))
            ac.build_count += 1
        elif event == "graphify_loader.done":
            g = report.graph
            g.nodes_loaded += int(d.get("nodes") or 0)
            g.edges_loaded += int(d.get("edges_loaded") or 0)
            g.edges_filtered += int(d.get("edges_filtered") or 0)
            ts = d.get("timestamp")
            if ts and (g.last_load is None or ts > g.last_load):
                g.last_load = ts
        elif event == "graph_vec_loader.done":
            report.graph.embed_runs += 1
            report.graph.embedded_nodes += int(d.get("embedded") or 0)
    return report


def default_log_paths() -> list[Path]:
    """Standard locations stropha writes structured logs to.

    Honours ``STROPHA_HOOK_LOG`` env var for the post-commit hook log. We
    also scan ``~/.cache/stropha-hook-*.log`` to pick up per-repo trails
    created by the v=3 cross-repo install.
    """
    main = Path(os.environ.get("STROPHA_HOOK_LOG", str(DEFAULT_HOOK_LOG)))
    candidates: list[Path] = []
    if main.is_file():
        candidates.append(main)
    cache_dir = main.parent
    if cache_dir.is_dir():
        for p in sorted(cache_dir.glob("stropha-hook-*.log")):
            if p not in candidates:
                candidates.append(p)
    return candidates


# --------------------------------------------------------------------- render


def render_table(report: CostReport):
    """Build a :class:`rich.console.RenderableType` table summary."""
    from rich.console import Group
    from rich.table import Table

    renders = []

    if report.repos:
        repos_t = Table(title="Per-repo activity", show_lines=False)
        repos_t.add_column("Repo", style="cyan", overflow="fold")
        repos_t.add_column("Runs", justify="right")
        repos_t.add_column("Files", justify="right")
        repos_t.add_column("Chunks", justify="right")
        repos_t.add_column("Embedded", justify="right")
        repos_t.add_column("Reused", justify="right")
        repos_t.add_column("Enricher cache", justify="right")
        repos_t.add_column("Last", overflow="fold")
        for rc in sorted(report.repos.values(), key=lambda r: -r.chunks_embedded):
            repos_t.add_row(
                rc.repo, str(rc.runs), str(rc.files_visited), str(rc.chunks_seen),
                str(rc.chunks_embedded), str(rc.chunks_reused),
                str(rc.enricher_cache_hits), rc.last_event or "—",
            )
        renders.append(repos_t)

    if report.adapters:
        adp = Table(title="Adapters seen", show_lines=False)
        adp.add_column("Stage", style="cyan")
        adp.add_column("Adapter ID", overflow="fold")
        adp.add_column("Builds", justify="right")
        for ac in sorted(
            report.adapters.values(), key=lambda a: (a.stage, -a.build_count),
        ):
            adp.add_row(ac.stage, ac.adapter_id, str(ac.build_count))
        renders.append(adp)

    if report.graph.nodes_loaded or report.graph.embed_runs:
        gt = Table(title="Graphify activity", show_lines=False)
        gt.add_column("Field", style="cyan")
        gt.add_column("Value")
        gt.add_row("Total nodes loaded (cumulative)", str(report.graph.nodes_loaded))
        gt.add_row("EXTRACTED edges loaded", str(report.graph.edges_loaded))
        gt.add_row("INFERRED edges filtered", str(report.graph.edges_filtered))
        gt.add_row("Vec-embed runs", str(report.graph.embed_runs))
        gt.add_row("Total nodes embedded", str(report.graph.embedded_nodes))
        gt.add_row("Last load", report.graph.last_load or "—")
        renders.append(gt)

    if report.hook.refresh_launched or report.hook.refresh_done or report.hook.refresh_failed:
        ht = Table(title="Post-commit hook", show_lines=False)
        ht.add_column("Event", style="cyan")
        ht.add_column("Count", justify="right")
        ht.add_row("Refresh launched", str(report.hook.refresh_launched))
        ht.add_row("Refresh done", str(report.hook.refresh_done))
        ht.add_row("Refresh failed", str(report.hook.refresh_failed))
        ht.add_row("graphify update invocations", str(report.hook.graphify_updates))
        renders.append(ht)

    if not renders:
        return None
    return Group(*renders)


def render_json(report: CostReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
