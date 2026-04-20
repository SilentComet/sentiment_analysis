"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 7  LLM INFERENCE ENGINE
  anthropic.AsyncAnthropic, structured JSON system
  prompt, tenacity exponential backoff on 429s,
  asyncio.Semaphore(8).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict

import anthropic
import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentiment_analysis.config import LLM_MAX_CONCURRENCY, logger


_LLM_SYSTEM_PROMPT = """You are an expert-level NLP analyst specializing in sentiment analysis,
aspect-based sentiment analysis (ABSA), and implicit intent detection.

You MUST respond with a single valid JSON object (no markdown fences, no prose).
Schema:
{
  "sentiment_score": <float -1.0 to 1.0>,
  "sentiment_label": <"very_negative"|"negative"|"neutral"|"positive"|"very_positive"|"mixed">,
  "dominant_emotion": <"joy"|"sadness"|"anger"|"fear"|"surprise"|"disgust"|"anticipation"|"trust"|"neutral">,
  "emotion_scores": {"joy": 0.0, "sadness": 0.0, ...},
  "has_sarcasm": <bool>,
  "has_corporate_speak": <bool>,
  "has_idiom": <bool>,
  "detected_phrases": ["phrase1", "phrase2"],
  "verbatim_evidence": ["exact quote from text", ...],
  "reasoning_steps": ["step 1 ...", "step 2 ...", "step 3 ..."],
  "confidence": <float 0.0 to 1.0>,
  "implicit_intent": <null or "persuade"|"inform"|"warn"|"deflect"|"forecast"|"criticize">
}

Rules:
- verbatim_evidence items MUST be exact substrings from the provided text.
- reasoning_steps MUST be at least 3 steps linking evidence to score.
- If sarcasm is detected, adjust sentiment_score accordingly (sarcasm inverts surface sentiment).
- Decode corporate speak and idioms into their true meaning before scoring.
"""


class LLMInferenceEngine:
    """
    Calls Anthropic Claude claude-sonnet-4-20250514 for complex chunks.

    Features:
      - AsyncRetrying with exponential backoff (tenacity)
      - asyncio.Semaphore to respect API concurrency limits
      - Structured JSON extraction with Pydantic validation
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_concurrent: int = LLM_MAX_CONCURRENCY,
    ) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def infer(self, text: str) -> Dict[str, Any]:
        """
        Returns a dict matching the JSON schema in _LLM_SYSTEM_PROMPT.
        Retries on rate-limit (429) and transient errors up to 5 times.
        """
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (anthropic.RateLimitError, anthropic.APIConnectionError)
                ),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                stop=stop_after_attempt(5),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.messages.create(
                        model=self._model,
                        max_tokens=1024,
                        system=_LLM_SYSTEM_PROMPT,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Analyze the following text chunk:\n\n"
                                    f"```\n{text[:4000]}\n```"
                                ),
                            }
                        ],
                    )

        raw = response.content[0].text.strip()
        # Strip accidental markdown fences
        raw = re.sub(
            r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE
        ).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(f"LLM returned invalid JSON: {exc}\n{raw[:300]}")
            # Graceful fallback
            data = {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "dominant_emotion": "neutral",
                "emotion_scores": {},
                "has_sarcasm": False,
                "has_corporate_speak": False,
                "has_idiom": False,
                "detected_phrases": [],
                "verbatim_evidence": [],
                "reasoning_steps": [
                    "LLM output parse failure; defaulting to neutral."
                ],
                "confidence": 0.1,
                "implicit_intent": None,
            }

        # Clamp score to valid range
        data["sentiment_score"] = float(
            np.clip(data.get("sentiment_score", 0.0), -1.0, 1.0)
        )
        return data
