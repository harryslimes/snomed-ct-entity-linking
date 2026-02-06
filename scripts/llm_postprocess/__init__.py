"""LLM post-processing agent for KIRI predictions.

This package implements a constrained "edit-script" interface that can delete
spans or shift their boundaries, with hard guardrails to avoid damaging macro
IoU (FP-sensitive) performance.
"""

