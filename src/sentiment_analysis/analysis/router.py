"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 4  HYBRID ROUTER
  Single threshold (default 0.60) — below → SLM,
  at/above → LLM.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from typing import Tuple

from sentiment_analysis.analysis.complexity import ComplexityScorer
from sentiment_analysis.config import COMPLEXITY_THRESHOLD
from sentiment_analysis.ingestion.chunker import RawChunk
from sentiment_analysis.schemas import InferenceRoute


class HybridRouter:
    """
    Decides inference path for each chunk.

    The router fires BEFORE any model call — it examines the raw text
    complexity and decides whether the chunk needs the full LLM or can
    be handled by the lighter SLM.
    """

    def __init__(self, threshold: float = COMPLEXITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._scorer = ComplexityScorer()

    def route(self, chunk: RawChunk) -> Tuple[InferenceRoute, float]:
        """Returns (route, complexity_score)."""
        score = self._scorer.score(chunk.text)
        route = InferenceRoute.LLM if score >= self.threshold else InferenceRoute.SLM
        return route, score
