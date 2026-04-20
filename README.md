# Sentiment Analyzer — 6-Layer NLP Architecture

A production-grade, asynchronous sentiment analysis pipeline designed for highly nuanced, context-aware document processing. It features a bifurcated inference engine (SLM/LLM) and comprehensive post-processing, including aspect-based sentiment analysis, narrative arc extraction, token-level grounding, and cultural signal detection (corporate speak, idioms, sarcasm).

## Architecture

The system decomposes the analytical workload across 6 orchestrated layers:

1. **Ingestion**: Multi-format support (text, JSON, paginated PDFs) and semantic chunking with overlapping windows (spaCy/tiktoken).
2. **Analysis**: Complexity scoring and bifurcated routing based on linguistic density and sarcasm signals. Cultural decoding for idioms and corporate lexicon.
3. **Inference**: Parallel async inference paths:
   - **SLM Path**: Local HuggingFace pipeline (DeBERTa/RoBERTa) for standard-complexity chunks. High-throughput, low latency.
   - **LLM Path**: Anthropic Claude API for high-complexity, heavily nuanced, or highly ambiguous chunks.
4. **Post-Processing**: Entity extraction, aspect-based sentiment tracking with contradiction resolution, narrative arc mapping (rolling means, segment detection), and token-level chain-of-evidence construction.
5. **Orchestration**: Asynchronous execution via `asyncio.gather`, recency weighting, and a 2-tier Semantic Cache (Redis exact-hash with in-memory fallback).
6. **Feedback**: MLOps endpoints for human correction integration and continuous learning loops.

## Premium Data Dashboard

Included is a bespoke, zero-dependency, vanilla HTML/CSS/JS dashboard (`dashboard/index.html`) demonstrating the `DocumentAnalysisResult` contract. 

**Features:**
- Editorial Obsidian-themed design system (no glassmorphism, no gradient "slop")
- Custom spring physics gauge animation with overshoot
- Catmull-Rom spline interpolation for the sentiment trajectory chart
- Emotion radar visualization
- Staggered entry animations

## Installation

This project requires Python 3.9+.

```bash
# Core dependencies
pip install -e .

# NLP + AI dependencies
pip install transformers spacy tiktoken anthropic redis

# Install English model for spaCy
python -m spacy download en_core_web_sm
```

*Note: The LLM path requires an Anthropc API key set as `ANTHROPIC_API_KEY`.*

## Usage

### Public Interface

```python
import asyncio
from sentiment_analysis import analyze_document

async def run():
    with open("earnings_report.txt") as f:
        text = f.read()
    
    # 14-step async pipeline executes here
    result = await analyze_document(text, filename="earnings_report.txt")
    print(result.overall_sentiment_label)

    # Export to JSON
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(run())
```

### Module Entry Point

You can quickly run the sample logic using the module entry point:

```bash
python -m sentiment_analysis
```

## Testing

The package includes 32 tests covering everything from basic ingestion to full Pydantic schema validation and integration.

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```
