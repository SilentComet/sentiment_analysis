"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 6  SLM INFERENCE ENGINE
  HuggingFace text-classification pipeline
  (DeBERTa / RoBERTa); graceful lexicon fallback
  when GPU / transformers is unavailable.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from sentiment_analysis.config import HF_AVAILABLE, hf_pipeline, logger
from sentiment_analysis.schemas import EmotionLabel, SentimentLabel


# Emotion label mapping from DeBERTa SST-style outputs → our EmotionLabel
_SENTIMENT_TO_SCORE: Dict[str, float] = {
    "very_negative": -1.0,
    "negative": -0.5,
    "neutral": 0.0,
    "positive": 0.5,
    "very_positive": 1.0,
    # HuggingFace star-rating variants
    "1 star": -1.0,
    "2 stars": -0.5,
    "3 stars": 0.0,
    "4 stars": 0.5,
    "5 stars": 1.0,
    # Simple binary
    "label_0": -0.5,
    "label_1": 0.5,
}


class SLMInferenceEngine:
    """
    Wraps a local HuggingFace sentiment pipeline.
    Falls back to a lexicon-based heuristic when HF is unavailable.

    Production: replace model_name with your fine-tuned DeBERTa-v3 checkpoint,
    e.g. "cross-encoder/nli-deberta-v3-large" or a SetFit model.
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ) -> None:
        self._pipe = None
        if HF_AVAILABLE and hf_pipeline is not None:
            try:
                self._pipe = hf_pipeline(
                    "text-classification",
                    model=model_name,
                    truncation=True,
                    max_length=512,
                    top_k=None,
                )
                logger.info(f"SLM loaded: {model_name}")
            except Exception as exc:
                logger.warning(f"SLM load failed ({exc}); using lexicon fallback.")

    def infer(
        self, text: str
    ) -> Tuple[float, SentimentLabel, EmotionLabel, float, Dict[str, float]]:
        """
        Returns:
            (sentiment_score, sentiment_label, dominant_emotion,
             confidence, emotion_scores_dict)
        """
        if self._pipe is not None:
            results = self._pipe(text[:512])
            # results is a list of [{label, score}]
            label_scores: Dict[str, float] = {}
            if isinstance(results, list) and isinstance(results[0], list):
                label_scores = {r["label"].lower(): r["score"] for r in results[0]}
            elif isinstance(results, list):
                label_scores = {r["label"].lower(): r["score"] for r in results}

            best_label = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
            confidence = label_scores[best_label]
            score = _SENTIMENT_TO_SCORE.get(best_label, 0.0)
        else:
            # ── Lexicon heuristic fallback ──
            score, confidence = self._lexicon_score(text)
            best_label = self._score_to_label_str(score)
            label_scores = {best_label: confidence}

        sentiment_label = self._score_to_sentiment_label(score)
        emotion_label, emotion_scores = self._estimate_emotion(text, score)
        return score, sentiment_label, emotion_label, confidence, emotion_scores

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _lexicon_score(text: str) -> Tuple[float, float]:
        """
        VADER-inspired lexicon heuristic (no external dependency).
        Returns (score ∈ [-1,1], confidence ∈ [0,1]).
        """
        POS = {
            "good", "great", "excellent", "outstanding", "positive", "benefit",
            "growth", "strong", "improve", "success", "profit", "gain", "win",
            "love", "happy", "hope", "optimistic", "confident", "record", "best",
        }
        NEG = {
            "bad", "poor", "loss", "decline", "fail", "risk", "concern", "weak",
            "negative", "miss", "drop", "cut", "layoff", "debt", "crisis",
            "challenge", "problem", "difficult", "struggle", "fear", "warn",
        }
        tokens = set(text.lower().split())
        pos = len(tokens & POS)
        neg = len(tokens & NEG)
        total = pos + neg
        if total == 0:
            return 0.0, 0.4
        score = (pos - neg) / total
        confidence = min(0.3 + total * 0.05, 0.80)
        return float(np.clip(score, -1.0, 1.0)), confidence

    @staticmethod
    def _score_to_label_str(score: float) -> str:
        if score <= -0.6:
            return "very_negative"
        if score <= -0.2:
            return "negative"
        if score < 0.2:
            return "neutral"
        if score < 0.6:
            return "positive"
        return "very_positive"

    @staticmethod
    def _score_to_sentiment_label(score: float) -> SentimentLabel:
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        if score <= -0.2:
            return SentimentLabel.NEGATIVE
        if score < 0.2:
            return SentimentLabel.NEUTRAL
        if score < 0.6:
            return SentimentLabel.POSITIVE
        return SentimentLabel.VERY_POSITIVE

    @staticmethod
    def _estimate_emotion(
        text: str, score: float
    ) -> Tuple[EmotionLabel, Dict[str, float]]:
        """Heuristic emotion estimation from text + sentiment score."""
        text_lower = text.lower()
        base: Dict[str, float] = {e.value: 0.05 for e in EmotionLabel}

        # Keyword-based boost
        emotion_keywords: Dict[str, List[str]] = {
            "joy": ["happy", "delight", "celebrat", "excit", "thrilled", "proud"],
            "sadness": ["sad", "disappoint", "unfortun", "regret", "mourn", "miss"],
            "anger": ["angry", "frustrated", "outrag", "infuriat", "furious"],
            "fear": ["afraid", "worried", "concern", "risk", "threat", "danger"],
            "anticipation": [
                "expect", "look forward", "hope", "upcoming", "future", "plan",
            ],
            "trust": ["reliable", "partner", "commit", "confident", "secure"],
            "surprise": ["unexpected", "sudden", "shocking", "unbelievable"],
            "disgust": [
                "disappoint", "unacceptable", "terrible", "awful", "horrible",
            ],
        }
        for emotion, keywords in emotion_keywords.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            base[emotion] += hits * 0.15

        # Adjust with sentiment score
        if score > 0.3:
            base["joy"] += 0.2
            base["anticipation"] += 0.1
        elif score < -0.3:
            base["sadness"] += 0.15
            base["fear"] += 0.1

        # Normalize
        total = sum(base.values())
        normalized = {k: round(v / total, 4) for k, v in base.items()}
        dominant = max(normalized, key=normalized.get)  # type: ignore[arg-type]
        return EmotionLabel(dominant), normalized
