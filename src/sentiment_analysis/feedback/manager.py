"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 13  FEEDBACK MANAGER
  MLOps correction API — analyst scores → Redis →
  nightly SetFit/LoRA fine-tune job.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sentiment_analysis.config import logger
from sentiment_analysis.orchestration.cache import SemanticCache
from sentiment_analysis.schemas import SentimentLabel


class FeedbackManager:
    """
    Collects human analyst corrections.
    These are written to Redis and consumed by an offline fine-tuning job.

    MLOps loop:
      1. Analyst reviews DocumentAnalysisResult
      2. Corrects scores/labels via this API
      3. Corrections stored in Redis (key pattern: sa:feedback:*)
      4. Nightly batch job:
          a. Reads all feedback records
          b. Creates labeled training examples
          c. Fine-tunes DeBERTa-v3 SLM via SetFit / LoRA
          d. Updates SLM checkpoint
    """

    def __init__(self, cache: SemanticCache) -> None:
        self._cache = cache

    async def submit_correction(
        self,
        document_id: str,
        corrected_overall_score: float,
        corrected_label: SentimentLabel,
        analyst_notes: str = "",
        chunk_corrections: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        feedback = {
            "document_id": document_id,
            "corrected_overall_score": corrected_overall_score,
            "corrected_label": corrected_label.value,
            "analyst_notes": analyst_notes,
            "chunk_corrections": chunk_corrections or [],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._cache.record_feedback(document_id, feedback)
        logger.info(f"Feedback submitted for document {document_id}")
