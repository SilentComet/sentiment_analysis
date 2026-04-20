"""
══════════════════════════════════════════════════════════════════════════════
  PHASE 3 — PYDANTIC SCHEMA
  Single source of truth for all structured outputs.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class SentimentLabel(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"
    MIXED = "mixed"


class IntentLabel(str, Enum):
    PERSUADE = "persuade"
    INFORM = "inform"
    WARN = "warn"
    CELEBRATE = "celebrate"
    CRITICIZE = "criticize"
    DEFLECT = "deflect"
    DENY = "deny"
    FORECAST = "forecast"


class EmotionLabel(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    NEUTRAL = "neutral"


class InferenceRoute(str, Enum):
    SLM = "slm"
    LLM = "llm"
    CACHED = "cached"


# ── Document Metadata ─────────────────────────────────────────────────────────


class DocumentMetadata(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: Optional[str] = None
    total_pages: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    word_count: int = Field(ge=0)
    language: str = "en"
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: float = Field(ge=0.0)
    inference_route_distribution: Dict[str, int] = Field(default_factory=dict)
    """e.g. {"slm": 42, "llm": 8, "cached": 3}"""


# ── Evidence & Grounding ──────────────────────────────────────────────────────


class VerbatimEvidence(BaseModel):
    """A single grounding quote extracted verbatim from the document."""

    quote: str = Field(min_length=5, max_length=500)
    chunk_id: str
    page_number: Optional[int] = None
    chunk_sequence: int = Field(ge=-1)
    character_offset_start: Optional[int] = None
    character_offset_end: Optional[int] = None
    relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_placeholder(cls, v: str) -> str:
        """Reject placeholder strings — evidence must be real document text."""
        if v.strip().lower() in {"n/a", "none", "null", "..."}:
            raise ValueError("Quote must contain actual document text.")
        return v


class ChainOfEvidence(BaseModel):
    """Full Chain-of-Evidence bundle for one aspect or claim."""

    claim_summary: str
    supporting_quotes: List[VerbatimEvidence] = Field(min_length=1)
    reasoning_steps: List[str] = Field(
        description="Step-by-step reasoning that links quotes → claim"
    )
    hallucination_risk_score: float = Field(ge=0.0, le=1.0)
    """0 = fully grounded, 1 = no evidence found"""
    grounding_coverage: float = Field(ge=0.0, le=1.0)
    """Fraction of claim tokens traceable to verbatim quotes"""


# ── Chunk-level Output ────────────────────────────────────────────────────────


class ChunkSentimentResult(BaseModel):
    chunk_id: str
    chunk_sequence: int = Field(ge=0)
    page_number: Optional[int] = None
    text_preview: str = Field(max_length=200)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    dominant_emotion: EmotionLabel
    emotion_scores: Dict[str, float] = Field(default_factory=dict)
    complexity_score: float = Field(ge=0.0, le=1.0)
    inference_route: InferenceRoute
    inference_latency_ms: float = Field(ge=0.0)
    has_sarcasm_signal: bool = False
    has_idiom_signal: bool = False
    has_corporate_speak: bool = False
    cultural_adjustments: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ── Aspect-Based Sentiment Analysis ──────────────────────────────────────────


class ContradictionRecord(BaseModel):
    """Documents a sentiment reversal for the same entity across chunks."""

    earlier_chunk_id: str
    later_chunk_id: str
    earlier_sentiment_score: float = Field(ge=-1.0, le=1.0)
    later_sentiment_score: float = Field(ge=-1.0, le=1.0)
    delta: float
    resolution_strategy: str
    resolved_score: float = Field(ge=-1.0, le=1.0)


class AspectEntity(BaseModel):
    entity_text: str
    entity_type: str
    """PERSON, ORG, PRODUCT, CONCEPT, LOCATION, ..."""
    mention_count: int = Field(ge=1)
    first_chunk_sequence: int = Field(ge=0)
    last_chunk_sequence: int = Field(ge=0)
    aspect_sentiment_score: float = Field(ge=-1.0, le=1.0)
    aspect_sentiment_label: SentimentLabel
    sentiment_trajectory: List[float] = Field(
        description="Score per mention in document order"
    )
    contradictions: List[ContradictionRecord] = Field(default_factory=list)
    chain_of_evidence: ChainOfEvidence
    sub_aspects: List[str] = Field(
        default_factory=list,
        description="Fine-grained aspects (quality, price, service, ...)",
    )

    @model_validator(mode="after")
    def trajectory_matches_mentions(self) -> AspectEntity:
        """Auto-reconcile mention_count ↔ sentiment_trajectory length."""
        if len(self.sentiment_trajectory) != self.mention_count:
            if len(self.sentiment_trajectory) < self.mention_count:
                self.sentiment_trajectory += [self.aspect_sentiment_score] * (
                    self.mention_count - len(self.sentiment_trajectory)
                )
            else:
                self.sentiment_trajectory = self.sentiment_trajectory[
                    : self.mention_count
                ]
        return self


# ── Narrative Arc ─────────────────────────────────────────────────────────────


class NarrativeSegment(BaseModel):
    label: str
    """e.g. 'introduction', 'body_1', 'climax', 'conclusion'"""
    chunk_sequence_start: int = Field(ge=0)
    chunk_sequence_end: int = Field(ge=0)
    mean_sentiment: float = Field(ge=-1.0, le=1.0)
    trend: str
    """'rising', 'falling', 'stable', 'volatile'"""
    peak_score: float = Field(ge=-1.0, le=1.0)
    trough_score: float = Field(ge=-1.0, le=1.0)


class SentimentTrajectory(BaseModel):
    scores: List[float] = Field(description="Sentiment score per chunk, in order")
    rolling_mean: List[float] = Field(
        description="5-chunk rolling mean for smoothed arc"
    )
    segments: List[NarrativeSegment]
    overall_trend: str
    """'improving', 'deteriorating', 'stable', 'u_shaped', 'inverted_u'"""
    inflection_points: List[int] = Field(
        description="Chunk sequences where direction changed"
    )
    intro_sentiment: float = Field(ge=-1.0, le=1.0)
    conclusion_sentiment: float = Field(ge=-1.0, le=1.0)
    sentiment_delta: float = Field(
        description="conclusion - intro; positive = ends better than starts"
    )


# ── Emotion & Intent ──────────────────────────────────────────────────────────


class EmotionProfile(BaseModel):
    dominant_emotion: EmotionLabel
    emotion_distribution: Dict[str, float]
    plutchik_wheel_position: Optional[str] = None
    """e.g. 'optimism' (anticipation + joy blend)"""


class IntentClassification(BaseModel):
    primary_intent: IntentLabel
    secondary_intents: List[IntentLabel] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    implicit_intent_notes: Optional[str] = None
    """Decoded passive-aggressiveness, doublespeak, etc."""


# ── Confidence Metrics ────────────────────────────────────────────────────────


class ConfidenceMetrics(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    chunk_coverage: float = Field(ge=0.0, le=1.0)
    """Fraction of chunks that produced a valid result"""
    evidence_density: float = Field(ge=0.0, le=1.0)
    """Mean grounding_coverage across all chain-of-evidence bundles"""
    model_agreement_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="SLM vs LLM agreement on overlapping chunks",
    )
    sarcasm_detection_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_flags: List[str] = Field(default_factory=list)
    """Human-readable flags: 'high_sarcasm_density', 'low_evidence', ..."""


# ── TOP-LEVEL RESPONSE ────────────────────────────────────────────────────────


class DocumentAnalysisResult(BaseModel):
    """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Master Pydantic response schema.
    Every field is grounded; no score is asserted
    without a corresponding ChainOfEvidence entry.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    document_metadata: DocumentMetadata

    # ── Scores ──────────────────────────────────
    overall_sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Weighted aggregate of all chunk scores. "
            "Recency-weighted: later chunks carry 20% more weight."
        ),
    )
    overall_sentiment_label: SentimentLabel

    # ── Narrative arc ────────────────────────────
    sentiment_trajectory: SentimentTrajectory

    # ── Aspect analysis ──────────────────────────
    aspect_analysis: List[AspectEntity] = Field(
        description="Entities/aspects with per-mention scores and evidence"
    )

    # ── Emotion & intent ─────────────────────────
    emotion_profile: EmotionProfile
    intent_classification: IntentClassification

    # ── Cultural / linguistic ────────────────────
    detected_idioms: List[str] = Field(default_factory=list)
    detected_corporate_speak: List[str] = Field(default_factory=list)
    sarcasm_detected: bool = False

    # ── Explainability ───────────────────────────
    document_level_evidence: ChainOfEvidence
    """Top-level evidence bundle justifying the overall_sentiment_score."""

    # ── Chunk detail ─────────────────────────────
    chunk_results: List[ChunkSentimentResult]

    # ── Quality ──────────────────────────────────
    confidence_metrics: ConfidenceMetrics
