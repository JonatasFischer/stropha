# Performance Tuning Guide

> Tips and best practices for optimizing stropha performance on large codebases.

---

## Quick Wins

### 1. Use Local Embeddings

Local embeddings via fastembed have zero network latency:

```bash
# Default: mixedbread-ai/mxbai-embed-large-v1 (1024 dims)
# No API key needed, weights download once

# For multilingual codebases:
stropha index --embedder bge-m3
```

### 2. Enable Query Cache

For repeated or similar queries, enable the semantic cache:

```bash
export STROPHA_QUERY_CACHE_ENABLED=1
export STROPHA_QUERY_CACHE_SIZE=500    # LRU entries
export STROPHA_QUERY_CACHE_TTL=3600    # 1 hour
```

The cache keys by quantized query embedding + filters, so similar queries hit the cache.

### 3. Use Incremental Indexing

The post-commit hook uses incremental indexing by default (v4+):

```bash
# Check hook version
stropha hook status

# Reinstall if needed (gets latest version)
stropha hook install
```

Incremental indexing only processes changed files, making re-indexing nearly instant.

---

## Large Codebase Strategies

### Repositories with 100k+ Files

1. **Exclude generated files** via `.strophaignore`:

```
# .strophaignore
**/node_modules/**
**/dist/**
**/build/**
**/*.min.js
**/*.bundle.js
**/vendor/**
**/__pycache__/**
```

2. **Use manifest for selective indexing**:

```yaml
# repos.yaml
repos:
  - path: src/
  - path: lib/
  - path: tests/
    enabled: false  # Skip tests for faster indexing
```

```bash
stropha index --manifest repos.yaml
```

3. **Increase file size limit if needed**:

```bash
export STROPHA_MAX_FILE_BYTES=2097152  # 2MB (default 1MB)
```

### Monorepos

For monorepos with multiple projects:

```bash
# Index specific subdirectories
stropha index --repo ./frontend --repo ./backend --repo ./shared

# Or use manifest
cat > repos.yaml << EOF
repos:
  - path: packages/frontend
  - path: packages/backend  
  - path: packages/shared
  - path: packages/deprecated
    enabled: false
EOF
stropha index --manifest repos.yaml
```

---

## Embedder Selection

| Embedder | Latency | Quality | Cost | Best For |
|----------|---------|---------|------|----------|
| `local` (mxbai) | ~5ms/chunk | High | Free | Development, most codebases |
| `bge-m3` | ~5ms/chunk | High | Free | Multilingual codebases |
| `voyage` | ~50ms/chunk | Highest | $$$ | Production with budget |

### Switching Embedders

Switching embedders requires re-indexing (embeddings are model-specific):

```bash
# Switch to bge-m3 for multilingual support
stropha index --embedder bge-m3 --rebuild
```

---

## Enricher Tradeoffs

| Enricher | Speed | Quality | Dependencies |
|----------|-------|---------|--------------|
| `noop` | Instant | Baseline | None |
| `hierarchical` | Fast | +10% recall | None |
| `graph-aware` | Fast | +15% recall | graphify graph |
| `contextual` | Slow | +35% recall | Ollama running |

### Recommendations

- **Development**: `hierarchical` (fast, no dependencies)
- **Production with Ollama**: `contextual` (best quality)
- **With graphify**: `graph-aware` (community context)

```bash
# Use contextual enricher (requires Ollama)
stropha index --enricher contextual

# Use graph-aware enricher (requires graphify graph)
graphify .
stropha index --enricher graph-aware
```

---

## Query-Time Optimization

### Disable Expensive Features for Interactive Use

```bash
# Fast search (all enhancements off)
export STROPHA_HYDE_ENABLED=0
export STROPHA_MULTI_QUERY_ENABLED=0
export STROPHA_RECURSIVE_RETRIEVAL=0
```

### Enable for Batch/Thorough Search

```bash
# Maximum recall (slower)
export STROPHA_HYDE_ENABLED=1
export STROPHA_MULTI_QUERY_ENABLED=1
export STROPHA_RECURSIVE_RETRIEVAL=1
```

### Feature Latency Impact

| Feature | Latency Impact | Recall Improvement |
|---------|----------------|-------------------|
| HyDE | +100-500ms | +10-20% |
| Multi-query | +200-800ms | +15-25% |
| Recursive | +10-50ms | +5-10% |
| Reranker | +50-200ms | +10-15% |

---

## Hook Optimization

### Fast Commit Experience

The hook should exit in <100ms (foreground). Heavy work runs in background:

```bash
# Check hook timing
time git commit --allow-empty -m "test"

# If slow, check for blocking calls
cat ~/.cache/stropha-hook.log | tail -20
```

### Skip Hook for Large Commits

During big refactors or rebases:

```bash
export STROPHA_HOOK_SKIP=1
git rebase main
unset STROPHA_HOOK_SKIP
stropha index  # Re-index manually
```

### Hook Timeout

Default timeout is 600s (10 min). Adjust for very large repos:

```bash
export STROPHA_HOOK_TIMEOUT=1200  # 20 minutes
```

---

## Database Optimization

### Index Size

Check your index size and composition:

```bash
stropha stats
```

Typical sizes:
- Small repo (<1k files): 10-50 MB
- Medium repo (1k-10k files): 50-200 MB
- Large repo (10k-100k files): 200 MB - 1 GB

### Database Location

Place the index on fast storage (SSD):

```bash
# Use local SSD instead of network storage
export STROPHA_INDEX_PATH=/fast/ssd/.stropha/index.db
```

### WAL Mode

SQLite WAL mode is enabled by default for concurrent reads. No action needed.

---

## Memory Usage

### Embedding Batch Size

For very large codebases, embeddings are batched. Memory scales with batch size:

- fastembed: ~500MB baseline + ~100MB per 1000 chunks
- Typical peak: 1-2 GB during indexing

### Graph Nodes

The graph-vec retrieval stream loads all node embeddings into memory:
- ~50 bytes per node embedding
- 50k nodes = ~2.5 MB (negligible)

---

## Profiling

### Enable Debug Logging

```bash
export STROPHA_LOG_LEVEL=DEBUG
stropha index 2>&1 | tee index.log
```

### Time Individual Stages

```bash
# Time chunking
time uv run python -c "from stropha.pipeline import Pipeline; ..."

# Check structlog output for per-stage timing
```

### Cost Dashboard

View aggregate timings across hook runs:

```bash
stropha cost
stropha cost --json  # Machine-readable
```

---

## Best Practices Summary

| Scenario | Recommendations |
|----------|-----------------|
| Development | Local embedder, hierarchical enricher, cache enabled |
| CI/CD | Incremental indexing, skip heavy enrichers |
| Production | Voyage embedder (if budget allows), contextual enricher |
| Large monorepo | Manifest-based indexing, .strophaignore, increased timeout |
| Interactive search | Disable HyDE/multi-query, enable cache |
| Thorough analysis | Enable all enhancements, reranker |

---

## Benchmarks

### Indexing Speed (Apple M2, local embedder)

| Repo Size | Files | Chunks | Time |
|-----------|-------|--------|------|
| Small | 500 | 2,000 | ~30s |
| Medium | 5,000 | 20,000 | ~5min |
| Large | 50,000 | 200,000 | ~50min |

### Query Latency (cached, local)

| Feature Set | p50 | p99 |
|-------------|-----|-----|
| Baseline | 15ms | 50ms |
| + HyDE | 150ms | 500ms |
| + Multi-query | 300ms | 800ms |
| + Reranker | 50ms | 200ms |

---

## Troubleshooting Performance

### Slow Indexing

1. Check for large binary files: `find . -size +1M -type f`
2. Add to `.strophaignore`
3. Check enricher: `stropha pipeline show`

### Slow Queries

1. Enable cache: `STROPHA_QUERY_CACHE_ENABLED=1`
2. Disable HyDE/multi-query for interactive use
3. Check index stats: `stropha stats`

### High Memory Usage

1. Reduce batch size (if customized)
2. Use local embedder instead of Voyage
3. Check for memory leaks: `stropha cost`
