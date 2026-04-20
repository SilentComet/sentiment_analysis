"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MASTER-LEVEL DOCUMENT SENTIMENT & INTENT ANALYZER                         ║
║   Chief AI Architect Pattern — Production Grade                              ║
║                                                                              ║
║   Six-Layer Architecture:                                                    ║
║     L1  Ingestion        — DocumentIngestor, SemanticChunker                 ║
║     L2  Analysis         — ComplexityScorer, HybridRouter, CulturalDecoder   ║
║     L3  Inference        — SLMInferenceEngine, LLMInferenceEngine            ║
║     L4  Post-Processing  — ChainOfEvidenceBuilder, ABSAResolver, NarrativeArc║
║     L5  Orchestration    — DocumentSentimentAnalyzer, SemanticCache           ║
║     L6  Feedback & MLOps — FeedbackManager                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from sentiment_analysis.schemas import (
    DocumentAnalysisResult,
    SentimentLabel,
    IntentLabel,
    EmotionLabel,
    InferenceRoute,
)
from sentiment_analysis.orchestration.analyzer import (
    DocumentSentimentAnalyzer,
    analyze_document,
)

__all__ = [
    "DocumentAnalysisResult",
    "DocumentSentimentAnalyzer",
    "analyze_document",
    "SentimentLabel",
    "IntentLabel",
    "EmotionLabel",
    "InferenceRoute",
]

__version__ = "1.0.0"
