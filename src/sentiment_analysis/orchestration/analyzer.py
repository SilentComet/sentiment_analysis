"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 12  DOCUMENT SENTIMENT ANALYZER
  Master orchestrator; all 14 pipeline steps via
  asyncio.gather on chunks.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from sentiment_analysis.analysis.cultural import CulturalDecoder
from sentiment_analysis.analysis.router import HybridRouter
from sentiment_analysis.config import (
    COMPLEXITY_THRESHOLD,
    LLM_MAX_CONCURRENCY,
    RECENCY_WEIGHT_BOOST,
    SARCASM_SIGNALS,
    logger,
)
from sentiment_analysis.ingestion.chunker import RawChunk, SemanticChunker
from sentiment_analysis.ingestion.ingestor import DocumentIngestor
from sentiment_analysis.inference.llm import LLMInferenceEngine
from sentiment_analysis.inference.slm import SLMInferenceEngine
from sentiment_analysis.orchestration.cache import SemanticCache
from sentiment_analysis.postprocessing.absa import ABSAResolver
from sentiment_analysis.postprocessing.evidence import ChainOfEvidenceBuilder
from sentiment_analysis.postprocessing.narrative import NarrativeArcBuilder
from sentiment_analysis.schemas import (
    ChunkSentimentResult,
    ConfidenceMetrics,
    DocumentAnalysisResult,
    DocumentMetadata,
    EmotionLabel,
    EmotionProfile,
    InferenceRoute,
    IntentClassification,
    IntentLabel,
    SentimentLabel,
)


class DocumentSentimentAnalyzer:
    """
    Orchestrates the full pipeline:

    1.  Cache check
    2.  Ingestion + chunking
    3.  Parallel async inference (SLM or LLM per chunk, based on router)
    4.  Cultural decoding
    5.  ABSA + entity resolution
    6.  Narrative arc computation
    7.  Chain-of-evidence assembly
    8.  Aggregation → DocumentAnalysisResult
    9.  Cache write
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
        route_distribution: Dict[str, int] = defaultdict(int)
        semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

        async def _process_chunk(chunk: RawChunk) -> ChunkSentimentResult:
            route, complexity_score = self._router.route(chunk)

            # Cultural decoding happens before inference
            cultural_data = self._cultural.analyze(chunk.text)

            t_chunk = time.perf_counter()

            if route == InferenceRoute.SLM:
                async with semaphore:
                    # SLM is sync; run in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    score, s_label, emotion, confidence, emotion_scores = (
                        await loop.run_in_executor(
                            None, self._slm.infer, chunk.text
                        )
                    )
                # Apply cultural adjustment
                score = float(
                    np.clip(score + cultural_data["score_adjustment"], -1.0, 1.0)
                )
                has_sarcasm = bool(
                    re.search("|".join(SARCASM_SIGNALS), chunk.text, re.I)
                )

            else:  # LLM
                llm_data = await self._llm.infer(chunk.text)
                raw_score = float(llm_data.get("sentiment_score", 0.0))
                score = float(
                    np.clip(
                        raw_score + cultural_data["score_adjustment"], -1.0, 1.0
                    )
                )
                s_label = SentimentLabel(
                    llm_data.get("sentiment_label", "neutral")
                )
                emotion = EmotionLabel(
                    llm_data.get("dominant_emotion", "neutral")
                )
                emotion_scores = llm_data.get("emotion_scores", {})
                confidence = float(llm_data.get("confidence", 0.7))
                has_sarcasm = bool(llm_data.get("has_sarcasm", False))

            latency_ms = (time.perf_counter() - t_chunk) * 1000
            route_distribution[route.value] += 1

            return ChunkSentimentResult(
                chunk_id=chunk.chunk_id,
                chunk_sequence=chunk.sequence,
                page_number=chunk.page_number,
                text_preview=(
                    chunk.text[:197] + "..."
                    if len(chunk.text) > 200
                    else chunk.text
                ),
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
        chunk_results = await asyncio.gather(
            *[_process_chunk(c) for c in chunks]
        )

        # ── 5. Aggregate overall sentiment (recency-weighted) ─────────────
        sorted_cr = sorted(chunk_results, key=lambda c: c.chunk_sequence)
        scores_arr = np.array([cr.sentiment_score for cr in sorted_cr])
        weights = self._recency_weights(len(scores_arr))
        overall_score = float(np.clip(np.dot(scores_arr, weights), -1.0, 1.0))
        overall_label = SLMInferenceEngine._score_to_sentiment_label(
            overall_score
        )

        # ── 6. Narrative arc ──────────────────────────────────────────────
        trajectory = self._arc_builder.build(list(sorted_cr))

        # ── 7. ABSA ───────────────────────────────────────────────────────
        aspect_entities = self._absa.extract_and_resolve(
            chunks, list(sorted_cr), self._coe_builder
        )

        # ── 8. Emotion profile ────────────────────────────────────────────
        emotion_totals: Dict[str, float] = defaultdict(float)
        for cr in sorted_cr:
            for emo, val in cr.emotion_scores.items():
                emotion_totals[emo] += val
        total_emo = sum(emotion_totals.values()) or 1.0
        emotion_distribution = {
            k: round(v / total_emo, 4) for k, v in emotion_totals.items()
        }
        dominant_emotion = EmotionLabel(
            max(emotion_distribution, key=emotion_distribution.get)  # type: ignore[arg-type]
            if emotion_distribution
            else "neutral"
        )
        emotion_profile = EmotionProfile(
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
        )

        # ── 9. Intent classification ──────────────────────────────────────
        intent_votes: Dict[str, int] = defaultdict(int)
        for cr in sorted_cr:
            cultural_d = self._cultural.analyze(cr.text_preview)
            if cultural_d.get("implicit_intent"):
                intent_votes[cultural_d["implicit_intent"]] += 1

        primary_intent_str = (
            max(intent_votes, key=intent_votes.get)  # type: ignore[arg-type]
            if intent_votes
            else "inform"
        )
        intent_cls = IntentClassification(
            primary_intent=IntentLabel(primary_intent_str),
            secondary_intents=[
                IntentLabel(k)
                for k, _ in sorted(
                    intent_votes.items(), key=lambda x: -x[1]
                )[1:3]
            ],
            confidence=round(
                min(
                    intent_votes.get(primary_intent_str, 1)
                    / max(len(sorted_cr) * 0.1, 1),
                    1.0,
                ),
                4,
            ),
        )

        # ── 10. Document-level chain of evidence ──────────────────────────
        doc_claim = (
            f"The document has an overall {overall_label.value} sentiment "
            f"(score: {overall_score:.3f}), {trajectory.overall_trend} trajectory."
        )
        doc_coe = self._coe_builder.build(doc_claim, list(sorted_cr), chunks)

        # ── 11. Confidence metrics ────────────────────────────────────────
        valid_chunks = sum(
            1 for cr in sorted_cr if abs(cr.sentiment_score) < 1.0
        )
        chunk_coverage = valid_chunks / max(len(sorted_cr), 1)
        evidence_density = (
            float(
                np.mean(
                    [
                        e.chain_of_evidence.grounding_coverage
                        for e in aspect_entities
                    ]
                )
            )
            if aspect_entities
            else 0.5
        )

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
            overall_confidence=round(
                chunk_coverage * 0.5 + evidence_density * 0.5, 4
            ),
            chunk_coverage=round(chunk_coverage, 4),
            evidence_density=round(evidence_density, 4),
            sarcasm_detection_confidence=round(
                1.0 - sarcasm_density * 0.5, 4
            ),
            uncertainty_flags=uncertainty_flags,
        )

        # ── 12. Metadata ──────────────────────────────────────────────────
        full_text = " ".join(c.text for c in chunks)
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
            detected_idioms=list(
                {
                    idiom
                    for cr in sorted_cr
                    for idiom in self._cultural.analyze(cr.text_preview)[
                        "detected_idioms"
                    ]
                }
            ),
            detected_corporate_speak=list(
                {
                    phrase
                    for cr in sorted_cr
                    for phrase in self._cultural.analyze(cr.text_preview)[
                        "detected_corp_speak"
                    ]
                }
            ),
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
        base[boost_start:] *= 1.0 + RECENCY_WEIGHT_BOOST
        return base / base.sum()


# ── Convenience coroutine — the primary public API ────────────────────────────


async def analyze_document(
    content: str,
    filename: Optional[str] = None,
    redis_url: str = "redis://localhost:6379/0",
    use_cache: bool = True,
    complexity_threshold: float = COMPLEXITY_THRESHOLD,
) -> DocumentAnalysisResult:
    """
    Convenience coroutine — the primary public API.

    Usage::

        import asyncio
        from sentiment_analysis import analyze_document

        with open("earnings_report.txt") as f:
            text = f.read()

        result = asyncio.run(analyze_document(text, filename="earnings_report.txt"))
        print(result.model_dump_json(indent=2))
    """
    analyzer = DocumentSentimentAnalyzer(
        redis_url=redis_url,
        complexity_threshold=complexity_threshold,
    )
    return await analyzer.analyze(
        content, filename=filename, use_cache=use_cache
    )
