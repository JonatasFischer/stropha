# MCP Integration Guide

> **stropha** exposes 11 MCP tools for code search and graph traversal.
> This guide shows how to integrate stropha with popular AI coding assistants.

---

## Quick Start

### 1. Install stropha

```bash
# Clone the repository
git clone https://github.com/your-org/stropha.git
cd stropha

# Install dependencies
uv sync
```

### 2. Index your codebase

```bash
# Index the current repo
uv run stropha index

# Or specify a different repo
STROPHA_TARGET_REPO=/path/to/your/project uv run stropha index

# Test it works
uv run stropha search "where is the main entry point"
```

### 3. Configure your MCP client

Copy the example config and adjust paths:

```bash
cp .mcp.example.json /path/to/your/project/.mcp.json
# Edit the paths in .mcp.json
```

---

## Client-Specific Instructions

### OpenCode

OpenCode natively supports MCP. Add stropha to your `opencode.json`:

```json
{
  "mcpServers": {
    "stropha_rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/stropha", "run", "stropha-mcp"],
      "env": {
        "STROPHA_TARGET_REPO": "${workspaceFolder}",
        "STROPHA_INDEX_PATH": "${HOME}/.stropha/index.db"
      }
    }
  }
}
```

The tools will appear as `mcp_Stropha_rag_*` in OpenCode's tool palette.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "stropha_rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/stropha", "run", "stropha-mcp"],
      "env": {
        "STROPHA_TARGET_REPO": "/path/to/your/project",
        "STROPHA_INDEX_PATH": "/path/to/index.db"
      }
    }
  }
}
```

### Continue (VS Code Extension)

Add to your Continue config (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "stropha_rag",
      "command": "uv",
      "args": ["--directory", "/path/to/stropha", "run", "stropha-mcp"],
      "env": {
        "STROPHA_TARGET_REPO": "${workspaceFolder}"
      }
    }
  ]
}
```

### Cursor

Cursor supports MCP via its settings. Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "stropha_rag": {
      "command": "uv",
      "args": ["--directory", "/path/to/stropha", "run", "stropha-mcp"],
      "env": {
        "STROPHA_TARGET_REPO": "."
      }
    }
  }
}
```

### Zed

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "assistant": {
    "mcp_servers": {
      "stropha_rag": {
        "command": "uv",
        "args": ["--directory", "/path/to/stropha", "run", "stropha-mcp"],
        "env": {
          "STROPHA_TARGET_REPO": "."
        }
      }
    }
  }
}
```

### Generic MCP Client

Any MCP-compatible client can use stropha via stdio transport:

```bash
# Start the server (connects via stdin/stdout)
uv run stropha-mcp
```

The server speaks the MCP protocol over stdio.

---

## Available Tools

### Search Tools

| Tool | Description |
|------|-------------|
| `search_code` | Hybrid semantic + lexical search. Use for "where is X?" or "how does Y work?" |
| `get_symbol` | Exact symbol lookup. Cheaper than `search_code` when you know the name. |
| `get_file_outline` | Get all symbols in a file. Plan reads before consuming context. |
| `list_repos` | List all indexed repositories. |

### Graph Traversal Tools

These tools require the graphify mirror to be loaded (run `graphify .` in your repo first).

| Tool | Description |
|------|-------------|
| `find_callers` | Who calls this symbol? BFS up `calls` edges. |
| `find_tests_for` | Find tests that cover a symbol. |
| `find_related` | Find symbols related to this one (any edge type). |
| `get_community` | Get all symbols in the same architectural community. |
| `find_rationale` | Find docs/ADRs explaining why this symbol exists. |
| `trace_feature` | Trace a feature description to all code that implements it. |

### Debug Tools

| Tool | Description |
|------|-------------|
| `get_config` | Show active configuration. Debug which index is being used. |

---

## Example Queries

### Finding Code

```
search_code("where is the FSRS calculator implemented")
search_code("how does authentication work", language=["typescript"])
search_code("API endpoints", path_prefix="src/api/", exclude_tests=True)
```

### Graph Traversal

```
find_callers("FsrsCalculator", depth=2)
find_tests_for("submitAnswer")
trace_feature("user submits an answer and mastery is updated")
get_community("StudyService")
```

### Architecture Understanding

```
search_code("entry point", kind=["class", "function"])
find_related("Database", relations=["implements", "references"])
find_rationale("FsrsCalculator")
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STROPHA_TARGET_REPO` | cwd | Repository to index/search |
| `STROPHA_INDEX_PATH` | `.stropha/index.db` | SQLite index location |
| `VOYAGE_API_KEY` | unset | Enable Voyage embeddings (optional, local fallback available) |
| `STROPHA_LOG_LEVEL` | `INFO` | Log verbosity |
| `STROPHA_RECURSIVE_RETRIEVAL` | `0` | Auto-merge sibling/adjacent chunks |
| `STROPHA_HYDE_ENABLED` | `0` | Hypothetical document rewriting |
| `STROPHA_MULTI_QUERY_ENABLED` | `0` | Multi-query expansion |
| `STROPHA_QUERY_CACHE_ENABLED` | `0` | Semantic query caching |

---

## Automatic Index Updates

stropha includes a post-commit hook that automatically updates the index after each commit:

```bash
# Install the hook in your project
cd /path/to/your/project
uv run --directory /path/to/stropha stropha hook install

# Check status
uv run --directory /path/to/stropha stropha hook status
```

The hook runs in the background (non-blocking) and updates both the stropha index and the graphify graph.

---

## Troubleshooting

### "No results" from search_code

1. Check the index exists: `ls .stropha/index.db`
2. Verify repo is indexed: `uv run stropha stats`
3. Try a simpler query to test basic functionality

### Graph tools return "graph_loaded: false"

1. Run graphify in your repo: `graphify .`
2. Verify graph exists: `ls graphify-out/graph.json`
3. Re-run index: `uv run stropha index`

### Slow indexing

1. Use local embeddings (default) instead of Voyage for large repos
2. Enable incremental indexing (default in hook v4)
3. Consider excluding large generated files via `.strophaignore`

### Hook not running

1. Check hook is installed: `cat .git/hooks/post-commit`
2. Verify permissions: `chmod +x .git/hooks/post-commit`
3. Check logs: `cat ~/.cache/stropha-hook.log`

---

## Advanced Configuration

### YAML Configuration

Create `stropha.yaml` in your repo root:

```yaml
walker:
  adapter: git-ls-files
  config:
    max_file_bytes: 1048576  # 1MB

enricher:
  adapter: contextual  # or: hierarchical, graph-aware, noop
  config:
    model: qwen2.5-coder:1.5b

embedder:
  adapter: local  # or: voyage, bge-m3
  config:
    model: mixedbread-ai/mxbai-embed-large-v1

retrieval:
  adapter: hybrid-rrf
  config:
    top_k: 10
    hyde_enabled: true
    multi_query_enabled: true
    recursive_enabled: true
```

### Multi-Repo Indexing

Index multiple repos into a single index:

```bash
# Via manifest file
cat > repos.yaml << EOF
repos:
  - path: ../frontend
  - path: ../backend
  - path: ../shared
EOF

uv run stropha index --manifest repos.yaml
```

Or via command line:

```bash
uv run stropha index --repo ../frontend --repo ../backend --repo ../shared
```

---

## Performance Tips

1. **Use local embeddings** for development (zero latency)
2. **Enable query cache** for repeated queries (`STROPHA_QUERY_CACHE_ENABLED=1`)
3. **Use incremental indexing** (default with hook v4)
4. **Exclude large files** via `.strophaignore`
5. **Use get_symbol instead of search_code** when you know the exact name
