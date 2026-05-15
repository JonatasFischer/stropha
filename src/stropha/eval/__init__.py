"""Evaluation harness — golden dataset + Recall@K + MRR (Phase 2 §16).

The harness runs deterministic offline metrics against a JSONL golden
dataset. No LLM in the hot path — Recall@K is computed by checking
whether any expected ``rel_path`` (or ``symbol``) appears in the top-K
hits.

Why: lets us guard regressions when swapping adapters (drift detection,
new enricher, new retrieval stream) without paying for LLM-judge runs.
RAGAS-style answer-level evaluation (faithfulness, answer relevance) is
intentionally out of scope here — see §16.4 for the optional integration.
"""

from .harness import (
    EvalCase,
    EvalReport,
    EvalResult,
    load_golden,
    run_eval,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "load_golden",
    "run_eval",
]
