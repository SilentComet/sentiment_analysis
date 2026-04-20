"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 11  SEMANTIC CACHE
  Redis exact-hash tier (SHA-256, 7-day TTL) +
  in-process dict fallback. record_feedback() writes
  to sa:feedback:* keys for the offline fine-tuning loop.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

from sentiment_analysis.config import REDIS_AVAILABLE, aioredis, logger
from sentiment_analysis.schemas import DocumentAnalysisResult


class SemanticCache:
    """
    Two-tier caching strategy:

    Tier 1 — Exact hash cache (Redis):
        SHA-256 of the document → full DocumentAnalysisResult JSON.
        TTL: 7 days. Prevents reprocessing identical documents.

    Tier 2 — In-memory fallback:
        dict-based cache when Redis is unavailable.
        (Production: extend with FAISS / pgvector for semantic similarity.)
    """

    _EXACT_TTL = 60 * 60 * 24 * 7  # 7 days

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._redis: Optional[Any] = None
        self._memory: Dict[str, str] = {}  # fallback in-process cache

    async def _get_redis(self):
        if not REDIS_AVAILABLE:
            return None
        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    @staticmethod
    def _document_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, content: str) -> Optional[DocumentAnalysisResult]:
        key = f"sa:doc:{self._document_hash(content)}"
        redis = await self._get_redis()
        raw = None
        if redis:
            raw = await redis.get(key)
        else:
            raw = self._memory.get(key)
        if raw:
            logger.info(f"Cache HIT for document hash {key[-8:]}")
            return DocumentAnalysisResult.model_validate_json(raw)
        return None

    async def set(self, content: str, result: DocumentAnalysisResult) -> None:
        key = f"sa:doc:{self._document_hash(content)}"
        serialized = result.model_dump_json()
        redis = await self._get_redis()
        if redis:
            await redis.setex(key, self._EXACT_TTL, serialized)
        else:
            self._memory[key] = serialized
        logger.info(f"Cached result for document hash {key[-8:]}")

    async def record_feedback(
        self, document_id: str, feedback: Dict[str, Any]
    ) -> None:
        """
        MLOps feedback loop: stores human-corrected scores for continuous learning.
        These records are consumed by the offline fine-tuning pipeline.
        """
        key = f"sa:feedback:{document_id}:{int(time.time())}"
        redis = await self._get_redis()
        payload = json.dumps(feedback)
        if redis:
            await redis.setex(key, 60 * 60 * 24 * 30, payload)  # 30 days
        else:
            self._memory[key] = payload
        logger.info(f"Feedback recorded: {key}")
