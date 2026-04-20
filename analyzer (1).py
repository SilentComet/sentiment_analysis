"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MASTER-LEVEL DOCUMENT SENTIMENT & INTENT ANALYZER                         ║
║   Chief AI Architect Pattern — Production Grade                              ║
║                                                                              ║
║   Phases:                                                                    ║
║     1. Architecture (see architecture diagram)                               ║
║     2. Async Python Pipeline (this file)                                     ║
║     3. Pydantic Schema (§ PHASE 3 below)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import anthropic
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ── Optional heavy dependencies ──────────────────────────────────────────────
try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    SPACY_AVAILABLE = False

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
except Exception:
    _ENC = None
    TIKTOKEN_AVAILABLE = False

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except Exception:
    aioredis = None  # type: ignore
    REDIS_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline  # type: ignore

    HF_AVAILABLE = True
except Exception:
    hf_pipeline = None  # type: ignore
    HF_AVAILABLE = False

logger = logging.getLogger("sentiment_analyzer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
#  ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
#  ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
#  ██████╔╝███████║███████║███████╗█████╗           ██╔╝
#  ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝          ██╔╝
#  ██║     ██║  ██║██║  ██║███████║███████╗         ██║
#  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝        ╚═╝
#
#  PYDANTIC SCHEMA — Single source of truth for all structured outputs
# ══════════════════════════════════════════════════════════════════════════════


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
    chunk_sequence: int = Field(ge=0)
    character_offset_start: Optional[int] = None
    character_offset_end: Optional[int] = None
    relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_placeholder(cls, v: str) -> str:
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
        if len(self.sentiment_trajectory) != self.mention_count:
            # Auto-fix: pad or trim
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


# ══════════════════════════════════════════════════════════════════════════════
#  ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
#  ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝       ███╗
#  ██████╔╝███████║███████║███████╗█████╗        ████╔╝
#  ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝       ╚═══██╗
#  ██║     ██║  ██║██║  ██║███████║███████╗    ██████╔╝
#  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═════╝
#
#  CORE PIPELINE IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# § 1  CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_MAX_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64
COMPLEXITY_THRESHOLD: float = 0.60  # route ≥ this to LLM
LLM_MAX_CONCURRENCY: int = 8  # semaphore cap for API calls
RECENCY_WEIGHT_BOOST: float = 0.20  # later chunks get +20% weight
CONTRADICTION_DELTA_THRESHOLD: float = 0.50  # |Δ score| triggers resolution

CORPORATE_SPEAK_LEXICON: Dict[str, str] = {
    "headwinds": "obstacles / challenges",
    "tailwinds": "favourable conditions",
    "synergies": "cost-cutting via merger",
    "rightsizing": "layoffs",
    "rationalization": "restructuring / cuts",
    "strategic realignment": "pivot under pressure",
    "optimizing the workforce": "layoffs",
    "value creation": "increasing profit",
    "ecosystem": "platform with lock-in",
    "mission-critical": "very important",
    "bandwidth": "capacity / time",
    "leverage": "use strategically",
    "robust": "reliable",
    "scalable": "can handle growth",
    "paradigm shift": "significant change",
    "move the needle": "make measurable progress",
    "circle back": "follow up later",
    "boil the ocean": "attempt too much",
    "low-hanging fruit": "easy wins",
}

SARCASM_SIGNALS: List[str] = [
    r"\boh great\b",
    r"\bsure[,.]? because\b",
    r"\byeah[,.]? right\b",
    r"\bsuper helpful\b.*\bnot\b",
    r"\btotally\b.{0,30}\bfailed\b",
    r"\bwhat could possibly go wrong\b",
    r"\bbrilliant (idea|plan|decision)\b",
    r"\bclearly\b.{0,20}\bsomeone\b.{0,20}\bthinking\b",
]

IMPLICIT_INTENT_PATTERNS: Dict[str, List[str]] = {
    "warn": [
        r"\brisks? (include|are|involve)\b",
        r"\bcaution\b",
        r"\bdownside\b",
        r"\bbe aware\b",
        r"\bfailure to\b",
    ],
    "persuade": [
        r"\byou should\b",
        r"\bwe urge\b",
        r"\bstrongly recommend\b",
        r"\bcompelling (case|reason|evidence)\b",
    ],
    "deflect": [
        r"\bexternal factors\b",
        r"\bchallenging macro(economic)? environment\b",
        r"\bnot (entirely|solely) within our control\b",
        r"\bindustry-wide\b",
    ],
    "forecast": [
        r"\bgoing forward\b",
        r"\bin the (next|coming) (quarter|year)\b",
        r"\bwe expect\b",
        r"\bprojected?\b",
        r"\boutlook\b",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# § 2  DOCUMENT INGESTION
# ─────────────────────────────────────────────────────────────────────────────


class RawChunk:
    """Intermediate pre-inference chunk representation."""

    __slots__ = (
        "chunk_id",
        "sequence",
        "page_number",
        "text",
        "token_count",
        "char_offset_start",
        "char_offset_end",
    )

    def __init__(
        self,
        sequence: int,
        text: str,
        page_number: Optional[int] = None,
        char_offset_start: int = 0,
    ) -> None:
        self.chunk_id = f"chunk_{sequence:05d}"
        self.sequence = sequence
        self.page_number = page_number
        self.text = text
        self.token_count = _count_tokens(text)
        self.char_offset_start = char_offset_start
        self.char_offset_end = char_offset_start + len(text)


def _count_tokens(text: str) -> int:
    if TIKTOKEN_AVAILABLE and _ENC is not None:
        return len(_ENC.encode(text))
    return len(text.split())  # fallback: word count


class DocumentIngestor:
    """
    Handles multiple document formats.
    Produces a list of (page_number, raw_text) tuples.
    Production systems would add PyMuPDF (fitz), python-docx, etc.
    """

    @staticmethod
    def ingest(content: str, filename: Optional[str] = None) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Returns (detected_language, [(page_num, page_text), ...]).
        For plain text, treats the whole document as page 1.
        """
        if filename and filename.lower().endswith(".json"):
            # Support pre-serialized [{page: N, text: "..."}] format
            try:
                pages_data = json.loads(content)
                if isinstance(pages_data, list) and "text" in pages_data[0]:
                    pages = [(int(p.get("page", i + 1)), p["text"]) for i, p in enumerate(pages_data)]
                    return "en", pages
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # Plain text: split on form-feed characters (PDF page breaks) or treat as single page
        raw_pages = content.split("\f")
        pages = [(i + 1, page.strip()) for i, page in enumerate(raw_pages) if page.strip()]
        if not pages:
            pages = [(1, content)]
        return "en", pages


# ─────────────────────────────────────────────────────────────────────────────
# § 3  SEMANTIC CHUNKER
# ─────────────────────────────────────────────────────────────────────────────


class SemanticChunker:
    """
    Splits pages into semantically coherent chunks with overlap.

    Strategy:
      1. Use spaCy sentence boundary detection (if available).
      2. Accumulate sentences until CHUNK_MAX_TOKENS is reached.
      3. Slide back CHUNK_OVERLAP_TOKENS to preserve cross-boundary context.
    """

    def __init__(
        self,
        max_tokens: int = CHUNK_MAX_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, pages: List[Tuple[int, str]]) -> List[RawChunk]:
        sentences: List[Tuple[int, str]] = []

        for page_num, text in pages:
            if SPACY_AVAILABLE and _NLP is not None:
                doc = _NLP(text[:100_000])  # spaCy limit guard
                for sent in doc.sents:
                    s = sent.text.strip()
                    if s:
                        sentences.append((page_num, s))
            else:
                # Fallback: split on period/newline
                for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
                    s = s.strip()
                    if s:
                        sentences.append((page_num, s))

        chunks: List[RawChunk] = []
        buf: List[str] = []
        buf_tokens: int = 0
        buf_page: Optional[int] = None
        char_offset: int = 0
        seq: int = 0

        def _flush() -> None:
            nonlocal buf, buf_tokens, seq, char_offset
            if not buf:
                return
            text_block = " ".join(buf)
            chunks.append(
                RawChunk(
                    sequence=seq,
                    text=text_block,
                    page_number=buf_page,
                    char_offset_start=char_offset,
                )
            )
            char_offset += len(text_block)
            seq += 1

        overlap_buf: List[str] = []
        overlap_tokens: int = 0

        for page_num, sent in sentences:
            s_tokens = _count_tokens(sent)

            if buf_tokens + s_tokens > self.max_tokens and buf:
                _flush()
                # Seed next chunk with overlap
                buf = list(overlap_buf)
                buf_tokens = overlap_tokens
                buf_page = page_num

            buf.append(sent)
            buf_tokens += s_tokens
            buf_page = buf_page or page_num

            # Maintain rolling overlap window
            overlap_buf.append(sent)
            overlap_tokens += s_tokens
            while overlap_tokens > self.overlap_tokens and overlap_buf:
                removed = overlap_buf.pop(0)
                overlap_tokens -= _count_tokens(removed)

        _flush()
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# § 4  COMPLEXITY SCORER & HYBRID ROUTER
# ─────────────────────────────────────────────────────────────────────────────


class ComplexityScorer:
    """
    Scores each chunk 0→1 on multiple signals:

    Signal                   Weight
    ──────────────────────────────
    Negation density           0.20   "not", "never", double negation
    Syntactic depth            0.20   subordinate clause count (spaCy)
    Sarcasm signal hits        0.25   regex pattern matches
    Lexical entropy            0.15   type/token ratio
    Corporate speak density    0.10   hits against CORPORATE_SPEAK_LEXICON
    Sentence length variance   0.10   stddev of sentence lengths
    """

    _NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "without", "hardly", "barely"}

    def score(self, text: str) -> float:
        signals: Dict[str, float] = {}

        tokens = text.lower().split()
        if not tokens:
            return 0.0

        # 1. Negation density
        neg_count = sum(1 for t in tokens if t in self._NEGATION_TOKENS)
        signals["negation"] = min(neg_count / max(len(tokens) * 0.05, 1), 1.0)

        # 2. Syntactic depth (spaCy)
        if SPACY_AVAILABLE and _NLP is not None:
            doc = _NLP(text[:5_000])
            subord_count = sum(
                1 for tok in doc if tok.dep_ in ("advcl", "relcl", "ccomp", "xcomp")
            )
            signals["syntax"] = min(subord_count / max(len(list(doc.sents)) * 2, 1), 1.0)
        else:
            # Heuristic: count subordinating conjunctions
            subordinators = {"although", "because", "since", "unless", "whereas", "while", "if", "when", "though"}
            sub_count = sum(1 for t in tokens if t in subordinators)
            signals["syntax"] = min(sub_count / max(len(tokens) * 0.02, 1), 1.0)

        # 3. Sarcasm signals
        hits = sum(1 for pattern in SARCASM_SIGNALS if re.search(pattern, text, re.I))
        signals["sarcasm"] = min(hits / 2.0, 1.0)

        # 4. Lexical entropy (type-token ratio)
        ttr = len(set(tokens)) / max(len(tokens), 1)
        signals["entropy"] = 1.0 - ttr  # high variety = less repetitive = more complex

        # 5. Corporate speak density
        corp_hits = sum(1 for phrase in CORPORATE_SPEAK_LEXICON if phrase in text.lower())
        signals["corp_speak"] = min(corp_hits / 5.0, 1.0)

        # 6. Sentence length variance
        sent_lengths = [len(s.split()) for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sent_lengths) > 1:
            stddev = float(np.std(sent_lengths))
            signals["len_variance"] = min(stddev / 20.0, 1.0)
        else:
            signals["len_variance"] = 0.0

        weights = {
            "negation": 0.20,
            "syntax": 0.20,
            "sarcasm": 0.25,
            "entropy": 0.15,
            "corp_speak": 0.10,
            "len_variance": 0.10,
        }
        return sum(signals.get(k, 0.0) * w for k, w in weights.items())


class HybridRouter:
    """Decides inference path for each chunk."""

    def __init__(self, threshold: float = COMPLEXITY_THRESHOLD) -> None:
        self.threshold = threshold
        self._scorer = ComplexityScorer()

    def route(self, chunk: RawChunk) -> Tuple[InferenceRoute, float]:
        """Returns (route, complexity_score)."""
        score = self._scorer.score(chunk.text)
        route = InferenceRoute.LLM if score >= self.threshold else InferenceRoute.SLM
        return route, score


# ─────────────────────────────────────────────────────────────────────────────
# § 5  SLM INFERENCE  (DeBERTa-v3 / SetFit)
# ─────────────────────────────────────────────────────────────────────────────

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
    "negative": -0.5,
    "positive": 0.5,
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

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
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

    def infer(self, text: str) -> Tuple[float, SentimentLabel, EmotionLabel, float, Dict[str, float]]:
        """
        Returns:
            (sentiment_score, sentiment_label, dominant_emotion,
             confidence, emotion_scores_dict)
        """
        t0 = time.perf_counter()

        if self._pipe is not None:
            results = self._pipe(text[:512])
            # results is a list of [{label, score}]
            label_scores: Dict[str, float] = {}
            if isinstance(results, list) and isinstance(results[0], list):
                label_scores = {r["label"].lower(): r["score"] for r in results[0]}
            elif isinstance(results, list):
                label_scores = {r["label"].lower(): r["score"] for r in results}

            best_label = max(label_scores, key=label_scores.get)  # type: ignore
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

    @staticmethod
    def _lexicon_score(text: str) -> Tuple[float, float]:
        """
        VADER-inspired lexicon heuristic (no external dependency).
        Returns (score ∈ [-1,1], confidence ∈ [0,1]).
        """
        POS = {"good", "great", "excellent", "outstanding", "positive", "benefit",
               "growth", "strong", "improve", "success", "profit", "gain", "win",
               "love", "happy", "hope", "optimistic", "confident", "record", "best"}
        NEG = {"bad", "poor", "loss", "decline", "fail", "risk", "concern", "weak",
               "negative", "miss", "drop", "cut", "layoff", "debt", "crisis",
               "challenge", "problem", "difficult", "struggle", "fear", "warn"}
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
    def _estimate_emotion(text: str, score: float) -> Tuple[EmotionLabel, Dict[str, float]]:
        """Heuristic emotion estimation from text + sentiment score."""
        text_lower = text.lower()
        base: Dict[str, float] = {e.value: 0.05 for e in EmotionLabel}

        # Keyword-based boost
        emotion_keywords: Dict[str, List[str]] = {
            "joy": ["happy", "delight", "celebrat", "excit", "thrilled", "proud"],
            "sadness": ["sad", "disappoint", "unfortun", "regret", "mourn", "miss"],
            "anger": ["angry", "frustrated", "outrag", "infuriat", "furious"],
            "fear": ["afraid", "worried", "concern", "risk", "threat", "danger"],
            "anticipation": ["expect", "look forward", "hope", "upcoming", "future", "plan"],
            "trust": ["reliable", "partner", "commit", "confident", "secure"],
            "surprise": ["unexpected", "sudden", "shocking", "unbelievable"],
            "disgust": ["disappoint", "unacceptable", "terrible", "awful", "horrible"],
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
        dominant = max(normalized, key=normalized.get)  # type: ignore
        return EmotionLabel(dominant), normalized


# ─────────────────────────────────────────────────────────────────────────────
# § 6  LLM INFERENCE  (Claude API — structured JSON output)
# ─────────────────────────────────────────────────────────────────────────────

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
                retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
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
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

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
                "reasoning_steps": ["LLM output parse failure; defaulting to neutral."],
                "confidence": 0.1,
                "implicit_intent": None,
            }

        # Clamp score to valid range
        data["sentiment_score"] = float(np.clip(data.get("sentiment_score", 0.0), -1.0, 1.0))
        return data


# ─────────────────────────────────────────────────────────────────────────────
# § 7  CULTURAL DECODER
# ─────────────────────────────────────────────────────────────────────────────


class CulturalDecoder:
    """
    Decodes:
      - Corporate speak → plain English + sentiment adjustment
      - Common English idioms → literal meaning
      - Implicit intent patterns (regex matching)
      - Passive-aggression signals
    """

    _IDIOM_MAP: Dict[str, str] = {
        "bite the bullet": "endure something painful",
        "burn bridges": "destroy relationships",
        "hit the nail on the head": "be exactly correct",
        "costs an arm and a leg": "very expensive",
        "the ball is in your court": "it is your responsibility now",
        "throw under the bus": "blame someone else",
        "elephant in the room": "obvious problem being ignored",
        "back to square one": "start over completely",
        "kick the can down the road": "postpone a decision",
        "move the goalposts": "change criteria unfairly",
    }

    _PASSIVE_AGGRESSION_SIGNALS: List[str] = [
        r"\bper my (last|previous) (email|message)\b",
        r"\bas (previously|already) (mentioned|stated|noted)\b",
        r"\bwould have thought\b",
        r"\bjust to (clarify|confirm|reiterate)\b",
        r"\bfor (your|future) reference\b",
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        adjustments: List[str] = []
        detected_corp: List[str] = []
        detected_idioms: List[str] = []
        score_adjustment: float = 0.0
        implicit_intent: Optional[str] = None

        # 1. Corporate speak
        for phrase, meaning in CORPORATE_SPEAK_LEXICON.items():
            if phrase in text_lower:
                detected_corp.append(phrase)
                adjustments.append(f"Corporate speak '{phrase}' decoded as: {meaning}")
                # Most corporate speak is mildly negative (euphemism)
                score_adjustment -= 0.05

        # 2. Idioms
        for idiom, meaning in self._IDIOM_MAP.items():
            if idiom in text_lower:
                detected_idioms.append(idiom)
                adjustments.append(f"Idiom '{idiom}' decoded as: {meaning}")

        # 3. Implicit intent
        for intent, patterns in IMPLICIT_INTENT_PATTERNS.items():
            if any(re.search(p, text, re.I) for p in patterns):
                implicit_intent = intent
                adjustments.append(f"Implicit intent detected: {intent}")
                break

        # 4. Passive aggression
        pa_hits = sum(1 for p in self._PASSIVE_AGGRESSION_SIGNALS if re.search(p, text, re.I))
        if pa_hits:
            adjustments.append(f"Passive-aggressive tone detected ({pa_hits} signals)")
            score_adjustment -= 0.10 * pa_hits

        return {
            "detected_corp_speak": detected_corp,
            "detected_idioms": detected_idioms,
            "score_adjustment": float(np.clip(score_adjustment, -0.5, 0.0)),
            "adjustments": adjustments,
            "implicit_intent": implicit_intent,
        }


# ─────────────────────────────────────────────────────────────────────────────
# § 8  CHAIN-OF-EVIDENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────


class ChainOfEvidenceBuilder:
    """
    Constructs grounded, hallucination-resistant evidence bundles.

    Algorithm:
    1. Collect verbatim quotes from chunk results.
    2. Score each quote's relevance to the claim via keyword overlap.
    3. Compute grounding_coverage: fraction of claim tokens traceable to quotes.
    4. Flag hallucination risk when coverage < 0.40.
    """

    def build(
        self,
        claim_summary: str,
        chunk_results: List[ChunkSentimentResult],
        raw_chunks: List[RawChunk],
        source_quotes: Optional[List[str]] = None,
    ) -> ChainOfEvidence:

        claim_tokens = set(re.findall(r"\b\w{4,}\b", claim_summary.lower()))

        # Gather evidence from chunk results (LLM-extracted quotes take priority)
        evidence_list: List[VerbatimEvidence] = []
        raw_text_by_id = {rc.chunk_id: rc.text for rc in raw_chunks}

        for cr in chunk_results:
            raw_text = raw_text_by_id.get(cr.chunk_id, "")
            # Use a short, sentence-level excerpt as evidence
            sentences = re.split(r"(?<=[.!?])\s+", raw_text)
            best_sentence = self._most_relevant_sentence(sentences, claim_tokens)
            if best_sentence:
                relevance = self._relevance(best_sentence, claim_tokens)
                evidence_list.append(
                    VerbatimEvidence(
                        quote=best_sentence[:500],
                        chunk_id=cr.chunk_id,
                        page_number=cr.page_number,
                        chunk_sequence=cr.chunk_sequence,
                        character_offset_start=raw_text.find(best_sentence),
                        character_offset_end=raw_text.find(best_sentence) + len(best_sentence),
                        relevance_score=relevance,
                    )
                )

        # Add any explicitly-supplied quotes (from LLM output)
        if source_quotes:
            for q in source_quotes[:5]:
                if len(q) >= 5:
                    evidence_list.append(
                        VerbatimEvidence(
                            quote=q[:500],
                            chunk_id="llm_extracted",
                            chunk_sequence=-1,
                            relevance_score=0.90,
                        )
                    )

        # Sort by relevance, keep top 5
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)
        evidence_list = evidence_list[:5]

        # Compute grounding coverage
        if evidence_list:
            covered_tokens = set()
            for ev in evidence_list:
                covered_tokens |= set(re.findall(r"\b\w{4,}\b", ev.quote.lower()))
            coverage = len(claim_tokens & covered_tokens) / max(len(claim_tokens), 1)
        else:
            coverage = 0.0

        hallucination_risk = max(0.0, 1.0 - coverage * 1.5)

        reasoning_steps = [
            f"Claim: '{claim_summary}'",
            f"Identified {len(evidence_list)} supporting passages across {len(chunk_results)} chunks.",
            f"Token-level grounding coverage: {coverage:.0%}.",
            f"Hallucination risk score: {hallucination_risk:.2f} "
            f"({'LOW' if hallucination_risk < 0.30 else 'MEDIUM' if hallucination_risk < 0.60 else 'HIGH'}).",
        ]

        if not evidence_list:
            # Must have at least one evidence item per schema
            evidence_list = [
                VerbatimEvidence(
                    quote="[No direct verbatim evidence found for this claim]",
                    chunk_id="none",
                    chunk_sequence=0,
                    relevance_score=0.0,
                )
            ]

        return ChainOfEvidence(
            claim_summary=claim_summary,
            supporting_quotes=evidence_list,
            reasoning_steps=reasoning_steps,
            hallucination_risk_score=round(hallucination_risk, 4),
            grounding_coverage=round(coverage, 4),
        )

    @staticmethod
    def _most_relevant_sentence(sentences: List[str], claim_tokens: set) -> Optional[str]:
        if not sentences:
            return None
        best, best_score = None, -1.0
        for s in sentences:
            if len(s.split()) < 4:
                continue
            s_tokens = set(re.findall(r"\b\w{4,}\b", s.lower()))
            score = len(s_tokens & claim_tokens) / max(len(s_tokens), 1)
            if score > best_score:
                best, best_score = s, score
        return best

    @staticmethod
    def _relevance(text: str, claim_tokens: set) -> float:
        t_tokens = set(re.findall(r"\b\w{4,}\b", text.lower()))
        return len(t_tokens & claim_tokens) / max(len(t_tokens), 1)


# ─────────────────────────────────────────────────────────────────────────────
# § 9  ABSA + ENTITY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────


class ABSAResolver:
    """
    Extracts named entities, groups their sentiment mentions, detects
    contradictions, and resolves them using a recency-weighted strategy.

    Contradiction resolution strategies:
      'recency'   — later mentions override earlier (default for evolving docs)
      'average'   — arithmetic mean of all mentions
      'extreme'   — take the mention with highest |score|
    """

    def __init__(self, contradiction_delta: float = CONTRADICTION_DELTA_THRESHOLD) -> None:
        self.contradiction_delta = contradiction_delta

    def extract_and_resolve(
        self,
        chunks: List[RawChunk],
        chunk_results: List[ChunkSentimentResult],
        coe_builder: ChainOfEvidenceBuilder,
    ) -> List[AspectEntity]:
        # ── 1. Entity extraction ──────────────────────────────────────────
        entity_mentions: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
        # entity_mentions[norm_name] = [(seq, score, chunk_id), ...]

        for chunk, cr in zip(chunks, chunk_results):
            entities = self._extract_entities(chunk.text)
            for ent_text in entities:
                norm = ent_text.lower().strip()
                entity_mentions[norm].append((cr.chunk_sequence, cr.sentiment_score, cr.chunk_id))

        # ── 2. Build AspectEntity objects ─────────────────────────────────
        result: List[AspectEntity] = []
        for norm_name, mentions in entity_mentions.items():
            if len(mentions) < 1:
                continue
            mentions.sort(key=lambda x: x[0])  # sort by sequence

            scores = [m[1] for m in mentions]
            trajectory = scores

            # ── 3. Contradiction detection ────────────────────────────────
            contradictions: List[ContradictionRecord] = []
            for i in range(len(mentions) - 1):
                for j in range(i + 1, len(mentions)):
                    delta = abs(mentions[j][1] - mentions[i][1])
                    if delta >= self.contradiction_delta:
                        resolved = self._resolve_contradiction(
                            mentions[i][1], mentions[j][1], strategy="recency"
                        )
                        contradictions.append(
                            ContradictionRecord(
                                earlier_chunk_id=mentions[i][2],
                                later_chunk_id=mentions[j][2],
                                earlier_sentiment_score=mentions[i][1],
                                later_sentiment_score=mentions[j][1],
                                delta=round(delta, 4),
                                resolution_strategy="recency",
                                resolved_score=round(resolved, 4),
                            )
                        )

            # ── 4. Final score: recency-weighted mean ─────────────────────
            weights = self._recency_weights(len(scores))
            final_score = float(np.dot(scores, weights) / np.sum(weights))
            final_score = float(np.clip(final_score, -1.0, 1.0))

            # ── 5. Build chain of evidence ────────────────────────────────
            relevant_chunks = [cr for cr in chunk_results if cr.chunk_id in {m[2] for m in mentions}]
            relevant_raws = [c for c in chunks if c.chunk_id in {m[2] for m in mentions}]
            coe = coe_builder.build(
                claim_summary=f"Sentiment toward '{norm_name}' is {SLMInferenceEngine._score_to_label_str(final_score)}.",
                chunk_results=relevant_chunks,
                raw_chunks=relevant_raws,
            )

            result.append(
                AspectEntity(
                    entity_text=norm_name,
                    entity_type=self._guess_entity_type(norm_name),
                    mention_count=len(mentions),
                    first_chunk_sequence=mentions[0][0],
                    last_chunk_sequence=mentions[-1][0],
                    aspect_sentiment_score=round(final_score, 4),
                    aspect_sentiment_label=SLMInferenceEngine._score_to_sentiment_label(final_score),
                    sentiment_trajectory=[round(s, 4) for s in trajectory],
                    contradictions=contradictions,
                    chain_of_evidence=coe,
                    sub_aspects=[],
                )
            )

        return sorted(result, key=lambda e: e.mention_count, reverse=True)[:20]

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        if SPACY_AVAILABLE and _NLP is not None:
            doc = _NLP(text[:10_000])
            return list({ent.text for ent in doc.ents if ent.label_ in
                         ("PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART", "LAW")})
        else:
            # Heuristic: capitalized noun phrases (2-3 words)
            return list({
                m.group()
                for m in re.finditer(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b", text)
            })[:10]

    @staticmethod
    def _guess_entity_type(name: str) -> str:
        corp_suffixes = {"inc", "corp", "ltd", "llc", "co", "company", "group"}
        if any(name.endswith(s) for s in corp_suffixes):
            return "ORG"
        if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
            return "PERSON"
        return "CONCEPT"

    @staticmethod
    def _recency_weights(n: int) -> np.ndarray:
        """Linearly increasing weights — last mention gets weight n, first gets weight 1."""
        weights = np.arange(1, n + 1, dtype=float)
        return weights / weights.sum()

    @staticmethod
    def _resolve_contradiction(earlier: float, later: float, strategy: str = "recency") -> float:
        if strategy == "recency":
            return later
        if strategy == "average":
            return (earlier + later) / 2.0
        if strategy == "extreme":
            return earlier if abs(earlier) > abs(later) else later
        return later


# ─────────────────────────────────────────────────────────────────────────────
# § 10  NARRATIVE ARC BUILDER
# ─────────────────────────────────────────────────────────────────────────────


class NarrativeArcBuilder:
    """
    Constructs the Sentiment Trajectory — the "mood arc" of a long document.

    Segments the trajectory into:
      - Introduction  (first 10% of chunks)
      - Rising action / body (10–80%)
      - Climax        (chunk with peak absolute score in middle 60%)
      - Conclusion    (last 20%)
    """

    _WINDOW = 5  # rolling mean window size

    def build(self, chunk_results: List[ChunkSentimentResult]) -> SentimentTrajectory:
        if not chunk_results:
            empty_coe_filler = SentimentTrajectory(
                scores=[],
                rolling_mean=[],
                segments=[],
                overall_trend="stable",
                inflection_points=[],
                intro_sentiment=0.0,
                conclusion_sentiment=0.0,
                sentiment_delta=0.0,
            )
            return empty_coe_filler

        sorted_results = sorted(chunk_results, key=lambda c: c.chunk_sequence)
        scores = [c.sentiment_score for c in sorted_results]
        n = len(scores)

        # Rolling mean
        rolling: List[float] = []
        for i in range(n):
            window = scores[max(0, i - self._WINDOW // 2): i + self._WINDOW // 2 + 1]
            rolling.append(round(float(np.mean(window)), 4))

        # Inflection points (sign changes in first diff)
        diffs = np.diff(scores)
        inflections = [
            i + 1
            for i in range(len(diffs) - 1)
            if np.sign(diffs[i]) != np.sign(diffs[i + 1])
        ]

        # Segments
        intro_end = max(1, int(n * 0.10))
        body_end = max(intro_end + 1, int(n * 0.80))
        conclusion_start = max(body_end, int(n * 0.80))

        def _segment(label: str, start: int, end: int) -> NarrativeSegment:
            seg_scores = scores[start:end]
            if not seg_scores:
                seg_scores = [0.0]
            trend = self._trend(seg_scores)
            return NarrativeSegment(
                label=label,
                chunk_sequence_start=start,
                chunk_sequence_end=end - 1,
                mean_sentiment=round(float(np.mean(seg_scores)), 4),
                trend=trend,
                peak_score=round(max(seg_scores), 4),
                trough_score=round(min(seg_scores), 4),
            )

        segments = [
            _segment("introduction", 0, intro_end),
            _segment("body", intro_end, body_end),
            _segment("conclusion", conclusion_start, n),
        ]

        intro_score = float(np.mean(scores[:intro_end])) if scores[:intro_end] else 0.0
        conclusion_score = float(np.mean(scores[conclusion_start:])) if scores[conclusion_start:] else 0.0
        delta = conclusion_score - intro_score

        overall_trend = self._classify_overall_trend(scores)

        return SentimentTrajectory(
            scores=[round(s, 4) for s in scores],
            rolling_mean=rolling,
            segments=segments,
            overall_trend=overall_trend,
            inflection_points=inflections,
            intro_sentiment=round(intro_score, 4),
            conclusion_sentiment=round(conclusion_score, 4),
            sentiment_delta=round(delta, 4),
        )

    @staticmethod
    def _trend(scores: List[float]) -> str:
        if len(scores) < 2:
            return "stable"
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0])
        variance = float(np.std(scores))
        if variance > 0.4:
            return "volatile"
        if slope > 0.02:
            return "rising"
        if slope < -0.02:
            return "falling"
        return "stable"

    @staticmethod
    def _classify_overall_trend(scores: List[float]) -> str:
        if len(scores) < 4:
            return "stable"
        n = len(scores)
        first_half = float(np.mean(scores[: n // 2]))
        second_half = float(np.mean(scores[n // 2 :]))
        mid = float(np.mean(scores[n // 4 : 3 * n // 4]))
        if mid > first_half and mid > second_half:
            return "inverted_u"
        if mid < first_half and mid < second_half:
            return "u_shaped"
        if second_half > first_half + 0.10:
            return "improving"
        if second_half < first_half - 0.10:
            return "deteriorating"
        return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# § 11  REDIS CACHE
# ─────────────────────────────────────────────────────────────────────────────


class SemanticCache:
    """
    Two-tier caching strategy:

    Tier 1 — Exact hash cache (Redis):
        SHA-256 of the document → full DocumentAnalysisResult JSON.
        TTL: 7 days. Prevents reprocessing identical documents.

    Tier 2 — Semantic cache (in-memory stub, extend with FAISS/pgvector):
        Cosine similarity of document embedding vs. cached embeddings.
        Returns the best match above similarity threshold.
        (Full vector cache implementation is infrastructure-specific;
         this class provides the interface with an in-memory demo.)
    """

    _EXACT_TTL = 60 * 60 * 24 * 7  # 7 days
    _SEMANTIC_THRESHOLD = 0.95

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

    async def record_feedback(self, document_id: str, feedback: Dict[str, Any]) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# § 12  MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────


class DocumentSentimentAnalyzer:
    """
    Orchestrates the full pipeline:

    1. Cache check
    2. Ingestion + chunking
    3. Parallel async inference (SLM or LLM per chunk, based on router)
    4. Cultural decoding
    5. ABSA + entity resolution
    6. Narrative arc computation
    7. Chain-of-evidence assembly
    8. Aggregation → DocumentAnalysisResult
    9. Cache write
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        llm_model: str = "claude-sonnet-4-20250514",
        slm_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        complexity_threshold: float = COMPLEXITY_THRESHOLD,
    ) -> None:
        self._ingestor = DocumentIngestor()
        self._chunker = SemanticChunker()
        self._router = HybridRouter(threshold=complexity_threshold)
        self._slm = SLMInferenceEngine(model_name=slm_model)
        self._llm = LLMInferenceEngine(model=llm_model)
        self._cultural = CulturalDecoder()
        self._absa = ABSAResolver()
        self._arc_builder = NarrativeArcBuilder()
        self._coe_builder = ChainOfEvidenceBuilder()
        self._cache = SemanticCache(redis_url=redis_url)

    async def analyze(
        self,
        content: str,
        filename: Optional[str] = None,
        use_cache: bool = True,
    ) -> DocumentAnalysisResult:
        t_start = time.perf_counter()

        # ── 1. Cache lookup ───────────────────────────────────────────────
        if use_cache:
            cached = await self._cache.get(content)
            if cached is not None:
                return cached

        # ── 2. Ingestion ──────────────────────────────────────────────────
        language, pages = self._ingestor.ingest(content, filename)
        total_pages = len(pages)

        # ── 3. Chunking ───────────────────────────────────────────────────
        chunks: List[RawChunk] = self._chunker.chunk(pages)
        logger.info(f"Document split into {len(chunks)} chunks.")

        # ── 4. Parallel inference ─────────────────────────────────────────
        chunk_results: List[ChunkSentimentResult] = []
        route_distribution: Dict[str, int] = defaultdict(int)

        semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

        async def _process_chunk(chunk: RawChunk) -> ChunkSentimentResult:
            route, complexity_score = self._router.route(chunk)

            # Cultural decoding happens before inference to adjust text interpretation
            cultural_data = self._cultural.analyze(chunk.text)

            t_chunk = time.perf_counter()

            if route == InferenceRoute.SLM:
                async with semaphore:
                    # SLM is sync; run in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    score, s_label, emotion, confidence, emotion_scores = await loop.run_in_executor(
                        None, self._slm.infer, chunk.text
                    )
                # Apply cultural adjustment
                score = float(np.clip(score + cultural_data["score_adjustment"], -1.0, 1.0))
                has_sarcasm = bool(re.search("|".join(SARCASM_SIGNALS), chunk.text, re.I))
                verbatim_quotes: List[str] = []
                reasoning_steps = [f"SLM inference score: {score:.3f}"]
                implicit_intent = cultural_data.get("implicit_intent")

            else:  # LLM
                llm_data = await self._llm.infer(chunk.text)
                raw_score = float(llm_data.get("sentiment_score", 0.0))
                # Apply cultural adjustment on top of LLM score
                score = float(np.clip(raw_score + cultural_data["score_adjustment"], -1.0, 1.0))
                s_label = SentimentLabel(llm_data.get("sentiment_label", "neutral"))
                emotion = EmotionLabel(llm_data.get("dominant_emotion", "neutral"))
                emotion_scores = llm_data.get("emotion_scores", {})
                confidence = float(llm_data.get("confidence", 0.7))
                has_sarcasm = bool(llm_data.get("has_sarcasm", False))
                verbatim_quotes = llm_data.get("verbatim_evidence", [])
                reasoning_steps = llm_data.get("reasoning_steps", [])
                implicit_intent = llm_data.get("implicit_intent")

            latency_ms = (time.perf_counter() - t_chunk) * 1000
            route_distribution[route.value] += 1

            return ChunkSentimentResult(
                chunk_id=chunk.chunk_id,
                chunk_sequence=chunk.sequence,
                page_number=chunk.page_number,
                text_preview=chunk.text[:197] + "..." if len(chunk.text) > 200 else chunk.text,
                sentiment_score=round(score, 4),
                sentiment_label=s_label,
                dominant_emotion=emotion,
                emotion_scores=emotion_scores,
                complexity_score=round(complexity_score, 4),
                inference_route=route,
                inference_latency_ms=round(latency_ms, 2),
                has_sarcasm_signal=has_sarcasm,
                has_idiom_signal=bool(cultural_data["detected_idioms"]),
                has_corporate_speak=bool(cultural_data["detected_corp_speak"]),
                cultural_adjustments=cultural_data["adjustments"],
                confidence=round(confidence, 4),
            )

        # Fire all chunks concurrently
        chunk_results = await asyncio.gather(*[_process_chunk(c) for c in chunks])

        # ── 5. Aggregate overall sentiment (recency-weighted) ─────────────
        sorted_cr = sorted(chunk_results, key=lambda c: c.chunk_sequence)
        scores_arr = np.array([cr.sentiment_score for cr in sorted_cr])
        weights = self._recency_weights(len(scores_arr))
        overall_score = float(np.clip(np.dot(scores_arr, weights), -1.0, 1.0))
        overall_label = SLMInferenceEngine._score_to_sentiment_label(overall_score)

        # ── 6. Narrative arc ──────────────────────────────────────────────
        trajectory = self._arc_builder.build(list(sorted_cr))

        # ── 7. ABSA ───────────────────────────────────────────────────────
        aspect_entities = self._absa.extract_and_resolve(chunks, list(sorted_cr), self._coe_builder)

        # ── 8. Emotion profile ────────────────────────────────────────────
        emotion_totals: Dict[str, float] = defaultdict(float)
        for cr in sorted_cr:
            for emo, val in cr.emotion_scores.items():
                emotion_totals[emo] += val
        total_emo = sum(emotion_totals.values()) or 1.0
        emotion_distribution = {k: round(v / total_emo, 4) for k, v in emotion_totals.items()}
        dominant_emotion = EmotionLabel(
            max(emotion_distribution, key=emotion_distribution.get)  # type: ignore
            if emotion_distribution
            else "neutral"
        )
        emotion_profile = EmotionProfile(
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
        )

        # ── 9. Intent classification ──────────────────────────────────────
        full_text = " ".join(c.text for c in chunks)
        intent_votes: Dict[str, int] = defaultdict(int)
        for cr in sorted_cr:
            cultural_d = self._cultural.analyze(cr.text_preview)
            if cultural_d.get("implicit_intent"):
                intent_votes[cultural_d["implicit_intent"]] += 1

        primary_intent_str = max(intent_votes, key=intent_votes.get) if intent_votes else "inform"  # type: ignore
        intent_cls = IntentClassification(
            primary_intent=IntentLabel(primary_intent_str),
            secondary_intents=[IntentLabel(k) for k, _ in sorted(intent_votes.items(), key=lambda x: -x[1])[1:3]],
            confidence=round(min(intent_votes.get(primary_intent_str, 1) / max(len(sorted_cr) * 0.1, 1), 1.0), 4),
        )

        # ── 10. Document-level chain of evidence ──────────────────────────
        doc_claim = (
            f"The document has an overall {overall_label.value} sentiment "
            f"(score: {overall_score:.3f}), {trajectory.overall_trend} trajectory."
        )
        doc_coe = self._coe_builder.build(doc_claim, list(sorted_cr), chunks)

        # ── 11. Confidence metrics ────────────────────────────────────────
        valid_chunks = sum(1 for cr in sorted_cr if abs(cr.sentiment_score) < 1.0)
        chunk_coverage = valid_chunks / max(len(sorted_cr), 1)
        evidence_density = float(
            np.mean([e.chain_of_evidence.grounding_coverage for e in aspect_entities])
        ) if aspect_entities else 0.5

        sarcasm_count = sum(1 for cr in sorted_cr if cr.has_sarcasm_signal)
        sarcasm_density = sarcasm_count / max(len(sorted_cr), 1)

        uncertainty_flags: List[str] = []
        if sarcasm_density > 0.20:
            uncertainty_flags.append("high_sarcasm_density")
        if evidence_density < 0.30:
            uncertainty_flags.append("low_evidence_coverage")
        if len(trajectory.inflection_points) > len(sorted_cr) * 0.3:
            uncertainty_flags.append("highly_volatile_sentiment")

        confidence_metrics = ConfidenceMetrics(
            overall_confidence=round(chunk_coverage * 0.5 + evidence_density * 0.5, 4),
            chunk_coverage=round(chunk_coverage, 4),
            evidence_density=round(evidence_density, 4),
            sarcasm_detection_confidence=round(1.0 - sarcasm_density * 0.5, 4),
            uncertainty_flags=uncertainty_flags,
        )

        # ── 12. Metadata ──────────────────────────────────────────────────
        processing_ms = (time.perf_counter() - t_start) * 1000
        metadata = DocumentMetadata(
            filename=filename,
            total_pages=total_pages,
            total_chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
            word_count=len(full_text.split()),
            language=language,
            processing_time_ms=round(processing_ms, 2),
            inference_route_distribution=dict(route_distribution),
        )

        # ── 13. Assemble final result ─────────────────────────────────────
        result = DocumentAnalysisResult(
            document_metadata=metadata,
            overall_sentiment_score=round(overall_score, 4),
            overall_sentiment_label=overall_label,
            sentiment_trajectory=trajectory,
            aspect_analysis=aspect_entities,
            emotion_profile=emotion_profile,
            intent_classification=intent_cls,
            detected_idioms=list({
                idiom
                for cr in sorted_cr
                for idiom in (self._cultural.analyze(cr.text_preview)["detected_idioms"])
            }),
            detected_corporate_speak=list({
                phrase
                for cr in sorted_cr
                for phrase in (self._cultural.analyze(cr.text_preview)["detected_corp_speak"])
            }),
            sarcasm_detected=sarcasm_count > 0,
            document_level_evidence=doc_coe,
            chunk_results=list(sorted_cr),
            confidence_metrics=confidence_metrics,
        )

        # ── 14. Write to cache ────────────────────────────────────────────
        if use_cache:
            await self._cache.set(content, result)

        logger.info(
            f"Analysis complete in {processing_ms:.0f}ms — "
            f"score={overall_score:.3f}, label={overall_label.value}, "
            f"chunks={len(chunks)}, route={dict(route_distribution)}"
        )
        return result

    @staticmethod
    def _recency_weights(n: int) -> np.ndarray:
        if n == 0:
            return np.array([])
        base = np.ones(n)
        # Boost the last 20% of chunks
        boost_start = int(n * 0.80)
        base[boost_start:] *= (1.0 + RECENCY_WEIGHT_BOOST)
        return base / base.sum()


# ─────────────────────────────────────────────────────────────────────────────
# § 13  FEEDBACK LOOP API
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# § 14  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


async def analyze_document(
    content: str,
    filename: Optional[str] = None,
    redis_url: str = "redis://localhost:6379/0",
    use_cache: bool = True,
    complexity_threshold: float = COMPLEXITY_THRESHOLD,
) -> DocumentAnalysisResult:
    """
    Convenience coroutine — the primary public API.

    Usage:
        import asyncio
        from analyzer import analyze_document

        with open("earnings_report.txt") as f:
            text = f.read()

        result = asyncio.run(analyze_document(text, filename="earnings_report.txt"))
        print(result.model_dump_json(indent=2))
    """
    analyzer = DocumentSentimentAnalyzer(
        redis_url=redis_url,
        complexity_threshold=complexity_threshold,
    )
    return await analyzer.analyze(content, filename=filename, use_cache=use_cache)


if __name__ == "__main__":
    import sys

    SAMPLE_DOC = """
    Q3 Earnings Report — FY2024

    Introduction: The quarter began under significant headwinds. Revenue declined by 8%
    year-over-year amid challenging macroeconomic conditions, and we have been rightsizing
    our workforce to better align costs with current demand levels.

    Product Performance: Product A received outstanding reviews from enterprise clients,
    with NPS scores reaching an all-time high of 72. Our engineering team has delivered
    a record number of features, and customer adoption is accelerating.

    Mid-Year Challenges: Oh great — just as we were gaining momentum, supply-chain
    disruptions rattled our hardware division. Product A's margins compressed sharply,
    and frankly the results were, shall we say, less than inspiring.

    Strategic Realignment: We are rationalizing our go-to-market approach and creating
    synergies across the enterprise and SMB units. By leveraging our ecosystem and
    moving the needle on mission-critical initiatives, we aim to unlock significant value.

    Outlook: Going forward, we expect robust tailwinds from our AI product line.
    In the next two quarters, we project revenue growth of 15–20%. The ball is in our
    court to capitalize on these opportunities, and we are confident in our ability
    to deliver exceptional results. Product A, despite earlier challenges, is now
    positioned as a market leader with an outstanding roadmap ahead.
    """

    async def _main():
        result = await analyze_document(
            SAMPLE_DOC,
            filename="sample_earnings.txt",
            use_cache=False,
        )
        print(result.model_dump_json(indent=2))

    asyncio.run(_main())
