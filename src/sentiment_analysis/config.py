"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CONFIG — Single source of truth for all tunables
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("sentiment_analyzer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64

# ── Routing ───────────────────────────────────────────────────────────────────
COMPLEXITY_THRESHOLD: float = 0.60  # route ≥ this to LLM

# ── Concurrency ──────────────────────────────────────────────────────────────
LLM_MAX_CONCURRENCY: int = 8  # semaphore cap for API calls

# ── Scoring ───────────────────────────────────────────────────────────────────
RECENCY_WEIGHT_BOOST: float = 0.20  # later chunks get +20% weight
CONTRADICTION_DELTA_THRESHOLD: float = 0.50  # |Δ score| triggers resolution

# ── Corporate-Speak Lexicon (20 entries) ──────────────────────────────────────
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
    "deep dive": "thorough analysis",
}

# ── Sarcasm Signal Patterns ───────────────────────────────────────────────────
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

# ── Implicit Intent Patterns ──────────────────────────────────────────────────
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

# ── Optional Heavy Dependencies ───────────────────────────────────────────────
# Centralised import guards so every module can check availability uniformly.

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
    aioredis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
    HF_AVAILABLE = True
except Exception:
    hf_pipeline = None  # type: ignore[assignment]
    HF_AVAILABLE = False


def count_tokens(text: str) -> int:
    """Token counter with tiktoken → word-count fallback."""
    if TIKTOKEN_AVAILABLE and _ENC is not None:
        return len(_ENC.encode(text))
    return len(text.split())
