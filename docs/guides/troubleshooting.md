# Troubleshooting Guide

> Solutions to common stropha issues.

---

## Installation Issues

### "sqlite3 was built without extension loading"

**Error:**
```
StorageError: Your Python sqlite3 was built without extension loading.
```

**Solution:**
Install Python via uv or pyenv (system Python on macOS often lacks this):

```bash
uv python install 3.12
uv sync
```

### "No module named 'sqlite_vec'"

**Error:**
```
ModuleNotFoundError: No module named 'sqlite_vec'
```

**Solution:**
```bash
uv sync  # Reinstall dependencies
```

---

## Indexing Issues

### "No files found to index"

**Causes:**
1. Not in a git repository
2. `.strophaignore` is too aggressive
3. All files exceed size limit

**Solutions:**
```bash
# Check you're in a git repo
git status

# Check ignore patterns
cat .strophaignore

# Try with verbose logging
STROPHA_LOG_LEVEL=DEBUG stropha index
```

### Indexing is extremely slow

**Causes:**
1. Using cloud embedder (Voyage) instead of local
2. Very large files being processed
3. Ollama enricher with slow model

**Solutions:**
```bash
# Use local embedder
stropha index --embedder local

# Check for large files
find . -size +1M -type f | head -20

# Add large files to .strophaignore
echo "*.min.js" >> .strophaignore
```

### "Embedding dim mismatch"

**Error:**
```
StorageError: Embedding dim mismatch: got 1024, expected 512
```

**Cause:** Index was created with a different embedder.

**Solution:**
```bash
# Rebuild with current embedder
stropha index --rebuild
```

---

## Search Issues

### "No results" for valid queries

**Causes:**
1. Index doesn't exist or is empty
2. Query doesn't match indexed content
3. Filters are too restrictive

**Solutions:**
```bash
# Check index exists and has content
stropha stats

# Try without filters
stropha search "your query"

# Check with debug logging
STROPHA_LOG_LEVEL=DEBUG stropha search "your query"
```

### Search returns irrelevant results

**Causes:**
1. Query is too generic
2. Index has stale content
3. Wrong embedder model

**Solutions:**
```bash
# Use more specific query terms
stropha search "FsrsCalculator class"

# Rebuild index
stropha index --rebuild

# Try exact symbol lookup instead
# (via MCP: get_symbol("FsrsCalculator"))
```

### Search is slow

**Causes:**
1. HyDE or multi-query enabled
2. Large index without cache
3. Ollama not running (for HyDE)

**Solutions:**
```bash
# Disable slow features
export STROPHA_HYDE_ENABLED=0
export STROPHA_MULTI_QUERY_ENABLED=0

# Enable query cache
export STROPHA_QUERY_CACHE_ENABLED=1

# Check Ollama status
curl http://localhost:11434/api/tags
```

---

## MCP Server Issues

### Server doesn't start

**Error:**
```
Error: Failed to start MCP server
```

**Solutions:**
```bash
# Check uv is available
which uv

# Test manual start
uv run stropha-mcp

# Check for port conflicts (stdio uses no ports, but check logs)
STROPHA_LOG_LEVEL=DEBUG uv run stropha-mcp
```

### Tools not appearing in client

**Causes:**
1. MCP config path is wrong
2. Server name doesn't match
3. Client needs restart

**Solutions:**
1. Verify config file location (varies by client)
2. Check server name is `stropha_rag`
3. Restart the MCP client

```json
{
  "mcpServers": {
    "stropha_rag": {
      "command": "uv",
      "args": ["--directory", "/correct/path/to/stropha", "run", "stropha-mcp"]
    }
  }
}
```

### "graph_loaded: false" from graph tools

**Cause:** graphify graph not loaded.

**Solution:**
```bash
# Generate the graphify graph
cd /path/to/your/repo
graphify .

# Re-run index to load graph
stropha index
```

---

## Hook Issues

### Hook not running

**Causes:**
1. Hook not installed
2. Hook not executable
3. STROPHA_HOOK_SKIP is set

**Solutions:**
```bash
# Check hook status
stropha hook status

# Reinstall
stropha hook install --force

# Check executable bit
ls -la .git/hooks/post-commit

# Unset skip variable
unset STROPHA_HOOK_SKIP
```

### Hook blocking commits

**Error:** Commits take several seconds.

**Cause:** Hook should fork immediately but might be blocking.

**Solutions:**
```bash
# Check hook version (should be v4+)
stropha hook status

# Reinstall to get latest version
stropha hook install --force

# Temporary workaround
export STROPHA_HOOK_SKIP=1
git commit -m "message"
unset STROPHA_HOOK_SKIP
```

### Hook errors in log

**Check the log:**
```bash
cat ~/.cache/stropha-hook.log | tail -50
```

**Common issues:**

1. **"uv not found"**: Set `STROPHA_HOOK_UV=/path/to/uv`
2. **"graphify not found"**: Set `STROPHA_HOOK_GRAPHIFY=/path/to/graphify` or `STROPHA_HOOK_NO_GRAPHIFY=1`
3. **Timeout**: Increase `STROPHA_HOOK_TIMEOUT=1200`

---

## Graph Issues

### Graph tools return empty results

**Causes:**
1. graphify hasn't been run
2. Graph file is outdated
3. Symbol name doesn't match

**Solutions:**
```bash
# Generate/update graph
cd /your/repo
graphify .

# Reload into stropha
stropha index

# Check graph stats
stropha stats | grep graph
```

### "community_id is None" warnings

**Cause:** graphify ran without community detection.

**Solution:**
```bash
# Re-run graphify with community detection
graphify . --communities
stropha index
```

---

## Database Issues

### "database is locked"

**Cause:** Multiple processes accessing the index.

**Solutions:**
```bash
# Wait for other processes to finish
sleep 5 && stropha index

# Check for zombie processes
ps aux | grep stropha

# Force unlock (last resort)
rm .stropha/index.db-shm .stropha/index.db-wal
```

### Corrupted index

**Symptoms:** Strange errors, missing data, crashes.

**Solution:**
```bash
# Rebuild from scratch
rm -rf .stropha/
stropha index
```

### Index too large

**Check size:**
```bash
du -h .stropha/index.db
stropha stats
```

**Solutions:**
1. Add more patterns to `.strophaignore`
2. Use manifest to exclude directories
3. Consider separate indexes for different projects

---

## Enricher Issues

### Ollama enricher fails silently

**Symptoms:** Index completes but search quality is poor.

**Check:**
```bash
# Is Ollama running?
curl http://localhost:11434/api/tags

# Is the model pulled?
ollama list | grep qwen2.5-coder

# Check enricher in use
stropha pipeline show
```

**Solution:**
```bash
# Start Ollama
ollama serve

# Pull the model
ollama pull qwen2.5-coder:1.5b

# Re-index
stropha index --enricher contextual
```

### MLX enricher not available

**Error:**
```
ImportError: mlx_lm not available
```

**Solution (macOS Apple Silicon only):**
```bash
uv pip install mlx-lm
```

---

## Environment Issues

### Environment variables not taking effect

**Check precedence:**
1. Command-line flags (highest)
2. Environment variables
3. YAML config file
4. `.env` file
5. Defaults (lowest)

```bash
# Verify env var is set
echo $STROPHA_INDEX_PATH

# Check resolved config
stropha pipeline show
```

### Wrong repository being indexed

**Check:**
```bash
# Which repo?
echo $STROPHA_TARGET_REPO
pwd

# Force specific repo
STROPHA_TARGET_REPO=/path/to/repo stropha index
```

---

## Getting Help

### Collect Debug Information

```bash
# System info
uv --version
python --version
uname -a

# stropha info
stropha --version
stropha stats
stropha pipeline show

# Logs
STROPHA_LOG_LEVEL=DEBUG stropha index 2>&1 | tee debug.log
```

### Report Issues

File issues at: https://github.com/your-org/stropha/issues

Include:
1. stropha version
2. OS and Python version
3. Error message (full traceback)
4. Steps to reproduce
5. Debug log output

---

## Quick Reference: Common Fixes

| Problem | Quick Fix |
|---------|-----------|
| No results | `stropha stats` to check index exists |
| Slow search | `STROPHA_HYDE_ENABLED=0` |
| Hook not running | `stropha hook install --force` |
| Graph tools empty | `graphify . && stropha index` |
| Embedding mismatch | `stropha index --rebuild` |
| Database locked | Wait or `rm .stropha/*.db-*` |
| Ollama errors | `ollama serve && ollama pull qwen2.5-coder:1.5b` |
