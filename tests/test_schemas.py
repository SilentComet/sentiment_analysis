"""
Tests for Phase 3 — Pydantic Schema validation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pydantic import ValidationError

from sentiment_analysis.schemas import (
    AspectEntity,
    ChainOfEvidence,
    ChunkSentimentResult,
    DocumentAnalysisResult,
    EmotionLabel,
    InferenceRoute,
    SentimentLabel,
    VerbatimEvidence,
)


class TestVerbatimEvidence:
    def test_valid_quote(self):
        ev = VerbatimEvidence(
            quote="Revenue increased by 15%",
            chunk_id="chunk_00001",
            chunk_sequence=0,
            relevance_score=0.85,
        )
        assert ev.quote == "Revenue increased by 15%"

    def test_placeholder_rejected(self):
        with pytest.raises(ValidationError):
            VerbatimEvidence(
                quote="n/a",
                chunk_id="chunk_00001",
                chunk_sequence=0,
                relevance_score=0.5,
            )

    def test_null_placeholder_rejected(self):
        with pytest.raises(ValidationError):
            VerbatimEvidence(
                quote="none",
                chunk_id="chunk_00001",
                chunk_sequence=0,
                relevance_score=0.5,
            )

    def test_ellipsis_placeholder_rejected(self):
        with pytest.raises(ValidationError):
            VerbatimEvidence(
                quote="...",
                chunk_id="chunk_00001",
                chunk_sequence=0,
                relevance_score=0.5,
            )

    def test_short_quote_rejected(self):
        with pytest.raises(ValidationError):
            VerbatimEvidence(
                quote="hi",
                chunk_id="chunk_00001",
                chunk_sequence=0,
                relevance_score=0.5,
            )


class TestAspectEntity:
    def test_trajectory_auto_padding(self):
        """mention_count=3, trajectory=[0.5] → should pad to length 3."""
        entity = AspectEntity(
            entity_text="Product A",
            entity_type="PRODUCT",
            mention_count=3,
            first_chunk_sequence=0,
            last_chunk_sequence=5,
            aspect_sentiment_score=0.5,
            aspect_sentiment_label=SentimentLabel.POSITIVE,
            sentiment_trajectory=[0.5],
            chain_of_evidence=ChainOfEvidence(
                claim_summary="Test claim",
                supporting_quotes=[
                    VerbatimEvidence(
                        quote="Product A is excellent",
                        chunk_id="chunk_00001",
                        chunk_sequence=0,
                        relevance_score=0.9,
                    )
                ],
                reasoning_steps=["Step 1"],
                hallucination_risk_score=0.1,
                grounding_coverage=0.8,
            ),
        )
        assert len(entity.sentiment_trajectory) == 3

    def test_trajectory_auto_trimming(self):
        """mention_count=1, trajectory=[0.5, 0.6, 0.7] → should trim to length 1."""
        entity = AspectEntity(
            entity_text="Product B",
            entity_type="PRODUCT",
            mention_count=1,
            first_chunk_sequence=0,
            last_chunk_sequence=0,
            aspect_sentiment_score=0.5,
            aspect_sentiment_label=SentimentLabel.POSITIVE,
            sentiment_trajectory=[0.5, 0.6, 0.7],
            chain_of_evidence=ChainOfEvidence(
                claim_summary="Test claim",
                supporting_quotes=[
                    VerbatimEvidence(
                        quote="Product B performed well",
                        chunk_id="chunk_00001",
                        chunk_sequence=0,
                        relevance_score=0.9,
                    )
                ],
                reasoning_steps=["Step 1"],
                hallucination_risk_score=0.1,
                grounding_coverage=0.8,
            ),
        )
        assert len(entity.sentiment_trajectory) == 1


class TestChainOfEvidence:
    def test_hallucination_risk_bounded(self):
        coe = ChainOfEvidence(
            claim_summary="Test claim",
            supporting_quotes=[
                VerbatimEvidence(
                    quote="Supporting evidence text",
                    chunk_id="chunk_00001",
                    chunk_sequence=0,
                    relevance_score=0.9,
                )
            ],
            reasoning_steps=["Step 1"],
            hallucination_risk_score=0.0,
            grounding_coverage=1.0,
        )
        assert coe.hallucination_risk_score == 0.0
        assert coe.grounding_coverage == 1.0

    def test_fully_hallucinated_risk(self):
        coe = ChainOfEvidence(
            claim_summary="Ungrounded claim",
            supporting_quotes=[
                VerbatimEvidence(
                    quote="Irrelevant quote text here",
                    chunk_id="chunk_00001",
                    chunk_sequence=0,
                    relevance_score=0.1,
                )
            ],
            reasoning_steps=["Step 1"],
            hallucination_risk_score=1.0,
            grounding_coverage=0.0,
        )
        assert coe.hallucination_risk_score == 1.0


class TestSentimentLabel:
    def test_all_labels_exist(self):
        labels = [sl.value for sl in SentimentLabel]
        assert "very_negative" in labels
        assert "neutral" in labels
        assert "very_positive" in labels
        assert "mixed" in labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
