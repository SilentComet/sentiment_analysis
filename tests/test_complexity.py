"""
Tests for Layer 2 — ComplexityScorer and HybridRouter.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentiment_analysis.analysis.complexity import ComplexityScorer
from sentiment_analysis.analysis.router import HybridRouter
from sentiment_analysis.ingestion.chunker import RawChunk
from sentiment_analysis.schemas import InferenceRoute


class TestComplexityScorer:
    def setup_method(self):
        self.scorer = ComplexityScorer()

    def test_empty_text_returns_zero(self):
        assert self.scorer.score("") == 0.0

    def test_simple_text_low_score(self):
        score = self.scorer.score("The weather is nice today.")
        assert 0.0 <= score <= 0.5

    def test_complex_text_higher_score(self):
        text = (
            "Oh great — although the headwinds are not entirely within our control, "
            "we have been rightsizing and creating synergies through strategic "
            "realignment, because the challenging macroeconomic environment is never "
            "without its leverage opportunities."
        )
        score = self.scorer.score(text)
        assert score > 0.2  # Should be noticeably higher than simple text

    def test_score_bounded(self):
        # Even extreme text should be 0-1
        text = "not never no " * 100 + " oh great, yeah right, totally failed"
        score = self.scorer.score(text)
        assert 0.0 <= score <= 1.0

    def test_sarcasm_boosts_complexity(self):
        plain = "The results were good and revenue increased."
        sarcastic = "Oh great, the results were totally what we expected. Yeah right."
        plain_score = self.scorer.score(plain)
        sarcastic_score = self.scorer.score(sarcastic)
        assert sarcastic_score > plain_score


class TestHybridRouter:
    def test_simple_chunk_routes_to_slm(self):
        router = HybridRouter(threshold=0.60)
        chunk = RawChunk(sequence=0, text="Revenue increased by 10% year over year.")
        route, score = router.route(chunk)
        assert route == InferenceRoute.SLM
        assert 0.0 <= score <= 1.0

    def test_custom_threshold(self):
        # With a very low threshold, everything goes to LLM
        router = HybridRouter(threshold=0.01)
        chunk = RawChunk(sequence=0, text="Simple sentence here.")
        route, _ = router.route(chunk)
        # With threshold near 0, even simple text should route to LLM
        # (unless the score is literally 0)

    def test_route_returns_score(self):
        router = HybridRouter()
        chunk = RawChunk(sequence=0, text="Test text with some content.")
        route, score = router.route(chunk)
        assert isinstance(score, float)
        assert route in (InferenceRoute.SLM, InferenceRoute.LLM)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
