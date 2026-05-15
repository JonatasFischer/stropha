"""Per-language chunker sub-adapters.

Each module wraps an existing implementation under
``stropha.ingest.chunkers`` so the heavy AST/regex code stays in one
place; the wrapper supplies the ``Stage`` introspection surface and
the ``@register_adapter(stage='language-chunker', name=...)`` registration.
"""
