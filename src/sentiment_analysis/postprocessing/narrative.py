"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 10  NARRATIVE ARC BUILDER
  5-chunk rolling mean, auto-segmentation
  (intro 10% / body 70% / conclusion 20%),
  inflection point detection, arc classification.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from typing import List

import numpy as np

from sentiment_analysis.schemas import (
    ChunkSentimentResult,
    NarrativeSegment,
    SentimentTrajectory,
)


class NarrativeArcBuilder:
    """
    Constructs the Sentiment Trajectory — the "mood arc" of a long document.

    Segments the trajectory into:
      - Introduction  (first 10% of chunks)
      - Body          (10–80%)
      - Conclusion    (last 20%)

    Arc classifications:
      u_shaped / inverted_u / improving / deteriorating / stable
    """

    _WINDOW = 5  # rolling mean window size

    def build(
        self, chunk_results: List[ChunkSentimentResult]
    ) -> SentimentTrajectory:
        if not chunk_results:
            return SentimentTrajectory(
                scores=[],
                rolling_mean=[],
                segments=[],
                overall_trend="stable",
                inflection_points=[],
                intro_sentiment=0.0,
                conclusion_sentiment=0.0,
                sentiment_delta=0.0,
            )

        sorted_results = sorted(chunk_results, key=lambda c: c.chunk_sequence)
        scores = [c.sentiment_score for c in sorted_results]
        n = len(scores)

        # Rolling mean
        rolling: List[float] = []
        for i in range(n):
            window = scores[
                max(0, i - self._WINDOW // 2) : i + self._WINDOW // 2 + 1
            ]
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

        intro_score = (
            float(np.mean(scores[:intro_end])) if scores[:intro_end] else 0.0
        )
        conclusion_score = (
            float(np.mean(scores[conclusion_start:]))
            if scores[conclusion_start:]
            else 0.0
        )
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
