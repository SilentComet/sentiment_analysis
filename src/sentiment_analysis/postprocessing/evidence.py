"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 8  CHAIN-OF-EVIDENCE BUILDER
  Token-level grounding coverage, verbatim quote
  extraction, hallucination risk score
  (0 = fully grounded).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import re
from typing import List, Optional

from sentiment_analysis.ingestion.chunker import RawChunk
from sentiment_analysis.schemas import (
    ChainOfEvidence,
    ChunkSentimentResult,
    VerbatimEvidence,
)


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
                        character_offset_end=(
                            raw_text.find(best_sentence) + len(best_sentence)
                        ),
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
            covered_tokens: set[str] = set()
            for ev in evidence_list:
                covered_tokens |= set(
                    re.findall(r"\b\w{4,}\b", ev.quote.lower())
                )
            coverage = len(claim_tokens & covered_tokens) / max(
                len(claim_tokens), 1
            )
        else:
            coverage = 0.0

        hallucination_risk = max(0.0, 1.0 - coverage * 1.5)

        reasoning_steps = [
            f"Claim: '{claim_summary}'",
            f"Identified {len(evidence_list)} supporting passages across "
            f"{len(chunk_results)} chunks.",
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
    def _most_relevant_sentence(
        sentences: List[str], claim_tokens: set
    ) -> Optional[str]:
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
