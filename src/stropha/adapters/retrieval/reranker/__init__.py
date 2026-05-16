"""Reranker sub-adapters for retrieval stage.

Cross-encoder rerankers re-score candidates from the RRF fusion step,
providing higher precision at the cost of additional latency.

Available adapters:
- noop: pass-through (default, zero cost)
- mxbai-rerank: mixedbread mxbai-rerank-large-v1 (ONNX, local)
"""
