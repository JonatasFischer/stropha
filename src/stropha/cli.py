"""Typer CLI — composition root for Phase 0.

Subcommands:
- `index` : walk → chunk → embed → store.
- `search`: embed query → dense top-k.
- `stats` : print index metadata.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config
from .embeddings import build_embedder
from .errors import StrophaError
from .ingest.pipeline import IndexPipeline
from .logging import configure_logging, get_logger
from .retrieval import SearchEngine
from .storage import Storage

load_dotenv()
app = typer.Typer(
    name="stropha",
    help="stropha — index a codebase and serve it to LLM clients via MCP.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


def _load_config() -> Config:
    try:
        return Config()  # type: ignore[call-arg]
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
    repo: Path = typer.Option(
        None, "--repo", help="Target repo to index (default: STROPHA_TARGET_REPO)."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Clear the index before reindexing."
    ),
) -> None:
    """Walk the target repo, chunk every file, embed, and store."""
    cfg = _load_config()
    target = (repo or cfg.target_repo).resolve()
    console.print(f"[bold]Target:[/bold] {target}")
    console.print(f"[bold]Index :[/bold] {cfg.resolve_index_path()}")

    try:
        embedder = build_embedder(cfg)
    except StrophaError as exc:
        console.print(f"[red]Embedder error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[bold]Embedder:[/bold] {embedder.model_name} (dim={embedder.dim})"
    )

    try:
        with Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim) as storage:
            pipeline = IndexPipeline(
                repo=target, storage=storage, embedder=embedder
            )
            stats = pipeline.run(rebuild=rebuild)
        console.print(
            f"[green]Done.[/green] {stats.files_visited} files · "
            f"{stats.chunks_seen} chunks · "
            f"{stats.chunks_embedded} embedded · "
            f"{stats.chunks_skipped_fresh} reused"
        )
    except StrophaError as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or symbol query."),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=50),
) -> None:
    """Run a Phase 0 dense search and pretty-print results."""
    cfg = _load_config()
    try:
        embedder = build_embedder(cfg)
        with Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim) as storage:
            engine = SearchEngine(storage, embedder)
            hits = engine.search(query, top_k=top_k)
    except StrophaError as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title=f"Top {len(hits)} for {query!r}", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Path:Lines", overflow="fold")
    table.add_column("Kind")
    table.add_column("Symbol", overflow="fold")
    table.add_column("Snippet", overflow="fold")
    for h in hits:
        table.add_row(
            str(h.rank),
            f"{h.score:.3f}",
            f"{h.rel_path}:{h.start_line}-{h.end_line}",
            h.kind,
            h.symbol or "—",
            h.snippet.replace("\n", " ⏎ ")[:160],
        )
    console.print(table)


@app.command()
def stats() -> None:
    """Print index metadata."""
    cfg = _load_config()
    try:
        embedder = build_embedder(cfg)
        with Storage(cfg.resolve_index_path(), embedding_dim=embedder.dim) as storage:
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


if __name__ == "__main__":
    app()
