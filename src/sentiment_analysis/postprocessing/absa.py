"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 9  ABSA RESOLVER
  spaCy NER (or capitalization heuristic), per-entity
  sentiment trajectory, ContradictionRecord with
  configurable resolution strategy.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from sentiment_analysis.config import (
    CONTRADICTION_DELTA_THRESHOLD,
    SPACY_AVAILABLE,
    _NLP,
)
from sentiment_analysis.ingestion.chunker import RawChunk
from sentiment_analysis.inference.slm import SLMInferenceEngine
from sentiment_analysis.postprocessing.evidence import ChainOfEvidenceBuilder
from sentiment_analysis.schemas import (
    AspectEntity,
    ChunkSentimentResult,
    ContradictionRecord,
    SentimentLabel,
)


class ABSAResolver:
    """
    Extracts named entities, groups their sentiment mentions, detects
    contradictions, and resolves them using a configurable strategy.

    Contradiction resolution strategies:
      'recency'   — later mentions override earlier (default for evolving docs)
      'average'   — arithmetic mean of all mentions
      'extreme'   — take the mention with highest |score|
    """

    def __init__(
        self, contradiction_delta: float = CONTRADICTION_DELTA_THRESHOLD
    ) -> None:
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
                entity_mentions[norm].append(
                    (cr.chunk_sequence, cr.sentiment_score, cr.chunk_id)
                )

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
            mention_chunk_ids = {m[2] for m in mentions}
            relevant_chunks = [
                cr for cr in chunk_results if cr.chunk_id in mention_chunk_ids
            ]
            relevant_raws = [
                c for c in chunks if c.chunk_id in mention_chunk_ids
            ]
            coe = coe_builder.build(
                claim_summary=(
                    f"Sentiment toward '{norm_name}' is "
                    f"{SLMInferenceEngine._score_to_label_str(final_score)}."
                ),
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
                    aspect_sentiment_label=SLMInferenceEngine._score_to_sentiment_label(
                        final_score
                    ),
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
            return list(
                {
                    ent.text
                    for ent in doc.ents
                    if ent.label_
                    in (
                        "PERSON", "ORG", "PRODUCT", "GPE",
                        "EVENT", "WORK_OF_ART", "LAW",
                    )
                }
            )
        else:
            # Heuristic: capitalized noun phrases (2-3 words)
            return list(
                {
                    m.group()
                    for m in re.finditer(
                        r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b", text
                    )
                }
            )[:10]

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
        """Linearly increasing weights — last mention gets weight n, first 1."""
        weights = np.arange(1, n + 1, dtype=float)
        return weights / weights.sum()

    @staticmethod
    def _resolve_contradiction(
        earlier: float, later: float, strategy: str = "recency"
    ) -> float:
        if strategy == "recency":
            return later
        if strategy == "average":
            return (earlier + later) / 2.0
        if strategy == "extreme":
            return earlier if abs(earlier) > abs(later) else later
        return later
