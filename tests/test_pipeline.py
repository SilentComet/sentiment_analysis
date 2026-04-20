"""
Integration test — end-to-end pipeline with sample document (SLM-only, no API keys).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import pytest

from sentiment_analysis.orchestration.analyzer import (
    DocumentSentimentAnalyzer,
    analyze_document,
)
from sentiment_analysis.schemas import DocumentAnalysisResult


SAMPLE_DOC = """
Q3 Earnings Report — FY2024

Introduction: The quarter began under significant headwinds. Revenue declined by 8%
year-over-year amid challenging macroeconomic conditions, and we have been rightsizing
our workforce to better align costs with current demand levels.

Product Performance: Product A received outstanding reviews from enterprise clients,
with NPS scores reaching an all-time high of 72. Our engineering team has delivered
a record number of features, and customer adoption is accelerating.

Outlook: Going forward, we expect robust tailwinds from our AI product line.
We are confident in our ability to deliver exceptional results.
"""


class TestPipelineIntegration:
    def test_analyzer_returns_valid_result(self):
        async def _run():
            result = await analyze_document(
                SAMPLE_DOC,
                filename="test_earnings.txt",
                use_cache=False,
            )
            return result

        result = asyncio.run(_run())
        assert isinstance(result, DocumentAnalysisResult)

    def test_result_has_chunks(self):
        async def _run():
            return await analyze_document(
                SAMPLE_DOC,
                filename="test.txt",
                use_cache=False,
            )

        result = asyncio.run(_run())
        assert len(result.chunk_results) > 0

    def test_overall_score_bounded(self):
        async def _run():
            return await analyze_document(
                SAMPLE_DOC,
                filename="test.txt",
                use_cache=False,
            )

        result = asyncio.run(_run())
        assert -1.0 <= result.overall_sentiment_score <= 1.0

    def test_metadata_populated(self):
        async def _run():
            return await analyze_document(
                SAMPLE_DOC,
                filename="test.txt",
                use_cache=False,
            )

        result = asyncio.run(_run())
        assert result.document_metadata.total_chunks > 0
        assert result.document_metadata.total_tokens > 0
        assert result.document_metadata.processing_time_ms > 0

    def test_trajectory_has_scores(self):
        async def _run():
            return await analyze_document(
                SAMPLE_DOC,
                filename="test.txt",
                use_cache=False,
            )

        result = asyncio.run(_run())
        assert len(result.sentiment_trajectory.scores) > 0

    def test_json_serialization(self):
        async def _run():
            return await analyze_document(
                SAMPLE_DOC,
                filename="test.txt",
                use_cache=False,
            )

        result = asyncio.run(_run())
        json_str = result.model_dump_json(indent=2)
        assert len(json_str) > 100  # Non-trivial output
        # Roundtrip
        parsed = DocumentAnalysisResult.model_validate_json(json_str)
        assert parsed.overall_sentiment_score == result.overall_sentiment_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
