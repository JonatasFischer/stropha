"""Typer CLI — composition root.

Subcommands:
- ``index``    : walk → chunk → enrich → embed → store (uses pipeline-adapters).
- ``search``   : embed query → hybrid top-k.
- ``stats``    : print index metadata.
- ``pipeline`` : introspect the resolved pipeline (``show``, ``validate``).
- ``adapters`` : list available adapters per stage.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, get_config
from .errors import StrophaError
from .logging import configure_logging, get_logger
from .pipeline import (
    Pipeline,
    all_adapters,
    build_stages,
    load_pipeline_config,
)

# Note: We don't call load_dotenv() here anymore. The get_config() singleton
# handles .env loading with proper precedence (env vars > .env file).
app = typer.Typer(
    name="stropha",
    help="stropha — index a codebase and serve it to LLM clients via MCP.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


def _load_config() -> Config:
    """Get the singleton config instance."""
    try:
        return get_config()
    except Exception as exc:
        raise typer.BadParameter(f"Config error: {exc}") from exc


@app.callback()
def _root(
    log_level: str = typer.Option(
        None, "--log-level", help="Override STROPHA_LOG_LEVEL (DEBUG/INFO/WARNING)."
    ),
) -> None:
    """Initialize logging before any subcommand runs."""
    cfg = _load_config()
    configure_logging(log_level or cfg.log_level)


@app.command()
def index(
    repo: list[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help=(
            "Repository to index. Repeat for multi-repo indexing "
            "(e.g. `--repo /a --repo /b`). Falls back to STROPHA_TARGET_REPO "
            "when omitted."
        ),
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest", "-m",
        help=(
            "Path to a YAML manifest with `repos: [{path: ...}]`. "
            "Mutually exclusive with --repo."
        ),
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Clear the index before reindexing."
    ),
    full: bool = typer.Option(
        False, "--full",
        help=(
            "Force a full walk (every file checked). "
            "Useful when --incremental defaults would otherwise apply."
        ),
    ),
    incremental: bool = typer.Option(
        False, "--incremental",
        help=(
            "Force git-diff incremental mode. Errors if no last_indexed_sha "
            "checkpoint exists (run a full index first, or pass --since)."
        ),
    ),
    since: str | None = typer.Option(
        None, "--since",
        help=(
            "Override last_indexed_sha for incremental mode. "
            "Useful to re-run against a specific checkpoint."
        ),
    ),
    enricher: str | None = typer.Option(
        None,
        "--enricher",
        help="Override pipeline.enricher.adapter (e.g. `noop`, `hierarchical`).",
    ),
    embedder: str | None = typer.Option(
        None,
        "--embedder",
        help="Override pipeline.embedder.adapter (e.g. `local`, `voyage`).",
    ),
) -> None:
    """Walk one or more repos, chunk every file, enrich, embed, and store."""
    cfg = _load_config()
    if manifest and repo:
        console.print("[red]--manifest and --repo are mutually exclusive[/red]")
        raise typer.Exit(code=1)

    targets: list[Path] = []
    if manifest:
        from .ingest.manifest import ManifestError, load_manifest
        try:
            entries = load_manifest(manifest)
        except ManifestError as exc:
            console.print(f"[red]Manifest error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        targets = [e.path for e in entries]
        if not targets:
            console.print("[yellow]Manifest has no enabled repos[/yellow]")
            raise typer.Exit(code=1)
    else:
        targets = [p.resolve() for p in (repo or [])]
        if not targets:
            targets = [cfg.target_repo.resolve()]
    for t in targets:
        if not t.is_dir():
            console.print(f"[red]Target not a directory:[/red] {t}")
            raise typer.Exit(code=1)

    overrides: dict = {}
    if enricher:
        overrides.setdefault("enricher", {})["adapter"] = enricher
    if embedder:
        overrides.setdefault("embedder", {})["adapter"] = embedder
    resolved = load_pipeline_config(overrides=overrides or None)

    console.print(
        "[bold]Target"
        + ("s" if len(targets) > 1 else "")
        + ":[/bold] "
        + ", ".join(str(t) for t in targets)
    )
    console.print(f"[bold]Index :[/bold] {cfg.resolve_index_path()}")

    try:
        built = build_stages(resolved)
    except StrophaError as exc:
        console.print(f"[red]Pipeline build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[bold]Embedder:[/bold] {built.embedder.adapter_id} "
        f"(dim={built.embedder.dim})"
    )
    console.print(f"[bold]Enricher:[/bold] {built.enricher.adapter_id}")

    if full and incremental:
        console.print("[red]--full and --incremental are mutually exclusive[/red]")
        raise typer.Exit(code=1)
    if full:
        mode = "full"
    elif incremental:
        mode = "incremental"
    else:
        mode = "auto"

    try:
        with built.storage as storage:  # type: ignore[union-attr]
            pipeline = Pipeline(
                storage=storage,
                embedder=built.embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=targets,
            )
            stats = pipeline.run(rebuild=rebuild, mode=mode, since_sha=since)
        # Surface file-level cache hits prominently — they are the
        # biggest perf win of Phase A incremental.
        skipped_part = (
            f"{stats.files_skipped_fresh} files fresh · "
            if stats.files_skipped_fresh
            else ""
        )
        evicted_part = (
            f"{stats.files_evicted} evicted · "
            if stats.files_evicted
            else ""
        )
        console.print(
            f"[green]Done.[/green] {stats.files_visited} files · "
            f"{skipped_part}{evicted_part}"
            f"{stats.chunks_seen} chunks · "
            f"{stats.chunks_embedded} embedded · "
            f"{stats.chunks_skipped_fresh} reused · "
            f"{stats.chunks_enriched_from_cache} enrich-cache hits · "
            f"{stats.chunks_enriched_fresh} enrich-fresh "
            f"across {len(stats.repos)} repo"
            + ("s" if len(stats.repos) != 1 else "")
        )
        for r in stats.repos:
            extras = []
            if r.files_skipped_fresh:
                extras.append(f"{r.files_skipped_fresh} fresh")
            if r.files_evicted:
                extras.append(f"{r.files_evicted} evicted")
            extra = f" ({', '.join(extras)})" if extras else ""
            console.print(
                f"  [dim]·[/dim] {r.normalized_key}"
                + (f"  ([blue]{r.url}[/blue])" if r.url else "")
                + f"  — {r.files_visited} files, {r.chunks_embedded} embedded{extra}"
            )
    except StrophaError as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Invalid input:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or symbol query."),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=50),
) -> None:
    """Hybrid retrieval (dense + BM25 + symbol-token fused via RRF)."""
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage:  # type: ignore[union-attr]
            hits = built.retrieval.search(query, top_k=top_k)
    except StrophaError as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title=f"Top {len(hits)} for {query!r}", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Repo", overflow="fold", style="dim")
    table.add_column("Path:Lines", overflow="fold")
    table.add_column("Kind")
    table.add_column("Symbol", overflow="fold")
    table.add_column("Snippet", overflow="fold")
    for h in hits:
        repo_label = h.repo.normalized_key if h.repo else "—"
        # Truncate to last two path segments for compactness.
        if "/" in repo_label:
            parts = repo_label.split("/")
            if len(parts) > 2:
                repo_label = ".../" + "/".join(parts[-2:])
        table.add_row(
            str(h.rank),
            f"{h.score:.3f}",
            repo_label,
            f"{h.rel_path}:{h.start_line}-{h.end_line}",
            h.kind,
            h.symbol or "—",
            h.snippet.replace("\n", " ⏎ ")[:140],
        )
    console.print(table)


@app.command()
def stats() -> None:
    """Print index metadata."""
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            info = storage.stats()
    except StrophaError as exc:
        console.print(f"[red]Stats failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]stropha[/bold] v{__version__}")
    console.print(f"DB         : {info['db_path']}")
    console.print(f"DB size    : {info['size_bytes'] / 1024:.1f} KB")
    console.print(f"Chunks     : {info['chunks']}")
    console.print(f"Files      : {info['files']}")
    console.print(f"Index dim  : {info['index_dim']}")
    if info["models"]:
        table = Table(title="Embedding models in index")
        table.add_column("Model")
        table.add_column("Dim", justify="right")
        table.add_column("Chunks", justify="right")
        for m in info["models"]:
            table.add_row(m["embedding_model"], str(m["embedding_dim"]), str(m["n"]))
        console.print(table)
    if len(info["models"]) > 1:
        console.print(
            "[yellow]Warning:[/yellow] index contains multiple embedding models. "
            "Run `stropha index --rebuild` to normalize."
        )
    if info.get("enrichers"):
        enrichers_table = Table(title="Enrichers in index (drift detection)")
        enrichers_table.add_column("Enricher adapter_id")
        enrichers_table.add_column("Chunks", justify="right")
        for e in info["enrichers"]:
            enrichers_table.add_row(e["enricher_id"], str(e["n"]))
        console.print(enrichers_table)
        cache = info.get("enrichment_cache_size", 0)
        console.print(f"[dim]Enrichment cache rows: {cache}[/dim]")
    if info["repos"]:
        repos_table = Table(title="Repositories in index")
        repos_table.add_column("Repo", overflow="fold")
        repos_table.add_column("Clone URL", overflow="fold")
        repos_table.add_column("Branch")
        repos_table.add_column("Files", justify="right")
        repos_table.add_column("Chunks", justify="right")
        repos_table.add_column("HEAD")
        for r in info["repos"]:
            head = (r["head_commit"] or "")[:8] or "—"
            repos_table.add_row(
                r["normalized_key"],
                r["url"] or "—",
                r["default_branch"] or "—",
                str(r["files"]),
                str(r["chunks"]),
                head,
            )
        console.print(repos_table)


# ---------------------------------------------------------------------------
# Pipeline introspection (Phase 1 — embedder + enricher are adapter-aware;
# other stages still appear as "legacy" until later phases migrate them).
# ---------------------------------------------------------------------------

pipeline_app = typer.Typer(
    name="pipeline",
    help="Inspect the resolved pipeline composition.",
    no_args_is_help=True,
)
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("show")
def pipeline_show(
    enricher: str | None = typer.Option(None, "--enricher"),
    embedder: str | None = typer.Option(None, "--embedder"),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not open the storage backend (skip storage + retrieval probes).",
    ),
) -> None:
    """Print the fully-resolved pipeline composition (YAML + env + CLI)."""
    overrides: dict = {}
    if enricher:
        overrides.setdefault("enricher", {})["adapter"] = enricher
    if embedder:
        overrides.setdefault("embedder", {})["adapter"] = embedder
    resolved = load_pipeline_config(overrides=overrides or None)
    try:
        built = build_stages(resolved, open_storage=not no_open)
    except StrophaError as exc:
        console.print(f"[red]Build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Pipeline composition", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Adapter", style="bold")
    table.add_column("adapter_id", overflow="fold")
    table.add_column("Status")
    table.add_column("Notes", overflow="fold")

    rows: list[tuple[str, object | None]] = [
        ("walker", built.walker),
        ("chunker", built.chunker),
        ("enricher", built.enricher),
        ("embedder", built.embedder),
        ("storage", built.storage),
        ("retrieval", built.retrieval),
    ]
    for stage_name, instance in rows:
        if instance is None:
            table.add_row(stage_name, "(skipped)", "—", "—", "use without --no-open to probe")
            continue
        h = instance.health()  # type: ignore[union-attr]
        marker = {"ready": "✓", "warning": "⚠", "error": "✗"}.get(h.status, "?")
        table.add_row(
            stage_name,
            instance.adapter_name,  # type: ignore[union-attr]
            instance.adapter_id,  # type: ignore[union-attr]
            f"{marker} {h.status}",
            h.message,
        )
    console.print(table)
    if built.storage is not None:
        built.storage.close()  # type: ignore[union-attr]


@pipeline_app.command("validate")
def pipeline_validate() -> None:
    """Run a lightweight health probe on every adapter. Exit non-zero on error."""
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
    except StrophaError as exc:
        console.print(f"[red]Build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    worst_status = "ready"
    for instance in (
        built.walker,
        built.chunker,
        built.enricher,
        built.embedder,
        built.storage,
        built.retrieval,
    ):
        if instance is None:
            continue
        h = instance.health()
        marker = {"ready": "✓", "warning": "⚠", "error": "✗"}.get(h.status, "?")
        console.print(f"  {marker} {instance.stage_name}/{instance.adapter_name}: {h.message}")
        if h.status == "error":
            worst_status = "error"
        elif h.status == "warning" and worst_status == "ready":
            worst_status = "warning"
    if built.storage is not None:
        built.storage.close()
    raise typer.Exit(code=0 if worst_status != "error" else 1)


adapters_app = typer.Typer(
    name="adapters",
    help="Enumerate adapters registered for each pipeline stage.",
    no_args_is_help=True,
)
app.add_typer(adapters_app, name="adapters")


@adapters_app.command("list")
def adapters_list(
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage."),
) -> None:
    """List every adapter the registry knows about."""
    registry = all_adapters()
    if stage:
        if stage not in registry:
            console.print(f"[yellow]No adapters registered for stage {stage!r}.[/yellow]")
            raise typer.Exit(code=1)
        registry = {stage: registry[stage]}

    table = Table(title="Available adapters", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Adapter", style="bold")
    for s, names in sorted(registry.items()):
        for n in names:
            table.add_row(s, n)
    console.print(table)


# ---------------------------------------------------------------------------
# Cost dashboard (Phase 3 §11.4 — aggregate structlog by adapter / repo)
# ---------------------------------------------------------------------------


@app.command()
def cost(
    log_path: Path = typer.Option(
        None, "--log-path",
        help="Override the hook log path. Defaults to ~/.cache/stropha-hook*.log",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit JSON instead of the rich table.",
    ),
) -> None:
    """Aggregate the structured log trail into a per-repo / per-adapter view."""
    from .cost import aggregate, default_log_paths, render_json, render_table

    paths = [log_path] if log_path else default_log_paths()
    if not paths:
        console.print("[yellow]No logs found. Has the hook ever run?[/yellow]")
        raise typer.Exit(code=1)

    report = aggregate(paths)
    if json_out:
        import sys
        sys.stdout.write(render_json(report))
        sys.stdout.write("\n")
        raise typer.Exit(code=0)
    rendered = render_table(report)
    if rendered is None:
        console.print(f"[yellow]Log present but contained no recognisable events:[/yellow] "
                      f"{', '.join(str(p) for p in paths)}")
        raise typer.Exit(code=1)
    console.print(f"[dim]Scanned: {', '.join(str(p) for p in paths)}[/dim]")
    console.print(rendered)


# ---------------------------------------------------------------------------
# Soft index (Phase 3 §8 — file watcher with debounce)
# ---------------------------------------------------------------------------


@app.command()
def watch(
    repo: Path = typer.Option(
        None, "--repo", "-r",
        help="Repo to watch. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
    interval: float = typer.Option(
        1.0, "--interval", min=0.1, max=60.0,
        help="Polling interval in seconds.",
    ),
    debounce: float = typer.Option(
        2.0, "--debounce", min=0.0, max=60.0,
        help="Wait this many seconds of no further changes before re-indexing.",
    ),
    full_refresh: bool = typer.Option(
        False, "--full-refresh",
        help="Run a full Pipeline pass on every cycle (slower but covers graphify reload).",
    ),
) -> None:
    """Watch a working tree and re-index on file changes.

    Press Ctrl-C to stop.
    """
    from .watch import watch_repo

    target = repo or _load_config().target_repo
    console.print(
        f"[bold]Watching:[/bold] {target.resolve()}\n"
        f"[dim]interval={interval}s · debounce={debounce}s · full_refresh={full_refresh}[/dim]\n"
        "[dim]Ctrl-C to stop.[/dim]"
    )
    try:
        watch_repo(
            target, interval_s=interval, debounce_s=debounce,
            full_refresh=full_refresh,
        )
    except StrophaError as exc:
        console.print(f"[red]Watch failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Evaluation harness (Phase 2 §16 — golden set + Recall@K + MRR)
# ---------------------------------------------------------------------------


@app.command()
def eval(
    golden: Path = typer.Option(
        Path("tests/eval/golden.jsonl"),
        "--golden", "-g",
        help="Path to JSONL golden file (one case per line).",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100),
    tag: str | None = typer.Option(
        None, "--tag",
        help="Run only cases with this tag (e.g. `--tag retrieval`).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of the rich table.",
    ),
) -> None:
    """Run the offline eval harness against the golden dataset."""
    from .eval import load_golden, run_eval

    if not golden.is_file():
        console.print(f"[red]Golden file not found:[/red] {golden}")
        raise typer.Exit(code=1)

    try:
        cases = load_golden(golden)
    except ValueError as exc:
        console.print(f"[red]Golden file invalid:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if tag:
        cases = [c for c in cases if tag in c.tags]
        if not cases:
            console.print(f"[yellow]No cases with tag {tag!r}.[/yellow]")
            raise typer.Exit(code=1)

    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
    except StrophaError as exc:
        console.print(f"[red]Pipeline build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        with built.storage:  # type: ignore[union-attr]
            report = run_eval(cases, built.retrieval, top_k=top_k)
    except StrophaError as exc:
        console.print(f"[red]Eval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        import json as _json

        payload = {
            "summary": report.summary(),
            "results": [
                {
                    "id": r.case.id, "query": r.case.query,
                    "tags": list(r.case.tags),
                    "hit": r.hit, "rank": r.rank_of_first_hit,
                    "matched_path": r.matched_path,
                    "matched_symbol": r.matched_symbol,
                }
                for r in report.results
            ],
            "by_tag": {
                t: sub.summary() for t, sub in report.by_tag().items()
            },
        }
        console.print_json(_json.dumps(payload))
        raise typer.Exit(code=0 if report.recall_at_k >= 0.85 else 2)

    table = Table(
        title=f"Evaluation — {report.n} cases (top_k={report.top_k})",
        show_lines=False,
    )
    table.add_column("Case", style="cyan")
    table.add_column("Query", overflow="fold")
    table.add_column("Hit", justify="center")
    table.add_column("Rank", justify="right")
    table.add_column("Tags", overflow="fold")
    for r in report.results:
        marker = "✓" if r.hit else "✗"
        rank_s = str(r.rank_of_first_hit) if r.rank_of_first_hit else "—"
        table.add_row(r.case.id, r.case.query[:80], marker, rank_s, ",".join(r.case.tags))
    console.print(table)

    summary = report.summary()
    console.print(
        f"[bold]Summary:[/bold] {summary['hits']}/{summary['cases']} hits · "
        f"recall@{report.top_k}={summary[f'recall@{report.top_k}']} · "
        f"mrr={summary['mrr']}"
    )
    by_tag = report.by_tag()
    if len(by_tag) > 1:
        tag_table = Table(title="By tag", show_lines=False)
        tag_table.add_column("Tag", style="cyan")
        tag_table.add_column("Cases", justify="right")
        tag_table.add_column(f"Recall@{report.top_k}", justify="right")
        tag_table.add_column("MRR", justify="right")
        for t, sub in sorted(by_tag.items()):
            tag_table.add_row(
                t, str(sub.n),
                f"{sub.recall_at_k:.2f}", f"{sub.mrr:.2f}",
            )
        console.print(tag_table)

    # Non-zero exit when below the spec threshold so CI catches regressions.
    raise typer.Exit(code=0 if report.recall_at_k >= 0.85 else 2)


# ---------------------------------------------------------------------------
# Hook management (RFC §7.1 — Trilha A 1.5c)
# ---------------------------------------------------------------------------

hook_app = typer.Typer(
    name="hook",
    help="Manage the post-commit hook (refreshes graphify graph + stropha index).",
    no_args_is_help=True,
)
app.add_typer(hook_app, name="hook")


def _resolve_hook_target(target: Path | None) -> Path:
    """Default to the configured target_repo, then cwd. Validate ``.git/`` exists."""
    if target is None:
        cfg = _load_config()
        target = cfg.target_repo or Path.cwd()
    target = target.expanduser().resolve()
    if not (target / ".git").exists():
        console.print(f"[red]Not a git repo:[/red] {target}")
        raise typer.Exit(code=1)
    return target


@hook_app.command("install")
def hook_install_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to install into. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Silence the warning when a graphify-hook coexists.",
    ),
    project_dir: Path = typer.Option(
        None, "--project-dir",
        help=(
            "Where stropha is installed (the directory with the `uv` venv). "
            "Bake into the hook so cross-repo installs work without manual "
            "env vars. Defaults to --target (dogfooding case)."
        ),
    ),
    index_path: Path = typer.Option(
        None, "--index-path",
        help=(
            "Bake STROPHA_INDEX_PATH into the hook so the index step doesn't "
            "inherit from --project-dir/.env. Recommended when --project-dir "
            "differs from --target."
        ),
    ),
    log_path: Path = typer.Option(
        None, "--log-path",
        help=(
            "Bake the hook log file path. Useful for separating per-repo "
            "refresh trails. Defaults to ~/.cache/stropha-hook.log."
        ),
    ),
) -> None:
    """Install or refresh the post-commit hook for the target repo."""
    from .tools.hook_install import install

    repo = _resolve_hook_target(target)

    # Light validation: warn (don't block) when --project-dir lacks the
    # usual project marker. Keeps the hook flexible while flagging obvious
    # typos.
    if project_dir is not None:
        resolved_project = project_dir.expanduser().resolve()
        if not (resolved_project / "pyproject.toml").is_file():
            console.print(
                f"[yellow]⚠ --project-dir {resolved_project} has no "
                f"pyproject.toml — `uv run` may fail at commit time.[/yellow]"
            )

    try:
        result = install(
            repo, force=force,
            project_dir=project_dir,
            index_path=index_path,
            log_path=log_path,
        )
    except (ValueError, OSError) as exc:
        console.print(f"[red]Install failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    marker = {"created": "✓", "updated": "✓", "appended": "+"}.get(result["action"], "?")
    console.print(
        f"{marker} Hook {result['action']}: [cyan]{result['hook_path']}[/cyan]"
        f"  (v={result['version']})"
    )
    # Surface baked defaults so the user knows what got injected.
    baked_lines: list[str] = []
    if result.get("project_dir"):
        baked_lines.append(f"  • project_dir: [cyan]{result['project_dir']}[/cyan]")
    if result.get("index_path"):
        baked_lines.append(f"  • index_path:  [cyan]{result['index_path']}[/cyan]")
    if log_path is not None:
        baked_lines.append(f"  • log_path:    [cyan]{result['log_path']}[/cyan]")
    if baked_lines:
        console.print("Baked defaults:")
        for ln in baked_lines:
            console.print(ln)
    if result["coexist_warning"]:
        console.print(
            "[yellow]⚠ A `graphify hook` block was detected in the same file.[/yellow]\n"
            "  This stropha hook already calls `graphify update`; keeping both "
            "will duplicate work.\n"
            "  Either run `graphify hook uninstall` or pass `--force` to silence."
        )
    console.print(
        f"[dim]Logs: {result['log_path']}[/dim]\n"
        "Test with: [cyan]git commit --allow-empty -m 'hook test'[/cyan] then "
        f"[cyan]tail -f {result['log_path']}[/cyan]"
    )


@hook_app.command("uninstall")
def hook_uninstall_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to uninstall from. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
) -> None:
    """Remove the stropha block from the post-commit hook."""
    from .tools.hook_install import uninstall

    repo = _resolve_hook_target(target)
    result = uninstall(repo)
    if result["action"] == "noop":
        console.print(f"[yellow]Nothing to do:[/yellow] {result['reason']}")
        raise typer.Exit(code=0)
    if result["action"] == "removed":
        console.print(f"[green]✓ Hook file removed:[/green] {result['hook_path']}")
    else:
        console.print(
            f"[green]✓ stropha block removed; rest of hook left intact:[/green] "
            f"{result['hook_path']}"
        )


@hook_app.command("status")
def hook_status_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to inspect. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
) -> None:
    """Report whether the post-commit hook is installed."""
    from .tools.hook_install import HOOK_VERSION, status

    repo = _resolve_hook_target(target)
    s = status(repo)
    table = Table(title=f"Hook status — {repo}", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Hook file", str(s.hook_path))
    if s.installed:
        ver_marker = "✓" if s.version == HOOK_VERSION else "⚠ outdated"
        table.add_row("Installed", f"yes (v={s.version}) {ver_marker}")
    else:
        table.add_row("Installed", "no")
    table.add_row("Graphify cohabit", "yes" if s.graphify_cohabit else "no")
    # v=3 baked defaults — show only the ones actually populated.
    if s.project_dir:
        table.add_row("Baked project_dir", str(s.project_dir))
    if s.index_path:
        table.add_row("Baked index_path", str(s.index_path))
    if s.log_path_default:
        table.add_row("Baked log_path", str(s.log_path_default))
    # Effective log path: baked default wins, else legacy fallback.
    effective_log = s.log_path_default or s.log_path
    table.add_row("Log file", str(effective_log))
    if effective_log.is_file():
        try:
            size = effective_log.stat().st_size
            table.add_row("Log size", f"{size} bytes")
        except OSError:
            pass
    console.print(table)
    raise typer.Exit(code=0 if s.installed else 1)


if __name__ == "__main__":
    app()
