"""Layer 3 — Inference subpackage (bifurcated SLM / LLM)."""

from sentiment_analysis.inference.slm import SLMInferenceEngine
from sentiment_analysis.inference.llm import LLMInferenceEngine

__all__ = ["SLMInferenceEngine", "LLMInferenceEngine"]
