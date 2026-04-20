"""
Tests for Layer 1 — Ingestion (DocumentIngestor + SemanticChunker).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentiment_analysis.ingestion.ingestor import DocumentIngestor
from sentiment_analysis.ingestion.chunker import SemanticChunker, RawChunk


class TestDocumentIngestor:
    def test_plain_text_single_page(self):
        text = "This is a simple document. It has two sentences."
        lang, pages = DocumentIngestor.ingest(text)
        assert lang == "en"
        assert len(pages) == 1
        assert pages[0][0] == 1
        assert "simple document" in pages[0][1]

    def test_form_feed_splits_pages(self):
        text = "Page one content.\fPage two content.\fPage three content."
        lang, pages = DocumentIngestor.ingest(text)
        assert len(pages) == 3
        assert pages[0][0] == 1
        assert pages[1][0] == 2
        assert pages[2][0] == 3

    def test_json_format(self):
        import json
        data = [
            {"page": 1, "text": "First page text."},
            {"page": 2, "text": "Second page text."},
        ]
        content = json.dumps(data)
        lang, pages = DocumentIngestor.ingest(content, filename="doc.json")
        assert len(pages) == 2
        assert pages[0][1] == "First page text."

    def test_empty_pages_filtered(self):
        text = "Content.\f\f\fMore content."
        lang, pages = DocumentIngestor.ingest(text)
        assert len(pages) == 2  # empty pages removed


class TestSemanticChunker:
    def test_single_sentence_chunk(self):
        chunker = SemanticChunker(max_tokens=1000, overlap_tokens=0)
        pages = [(1, "This is a short sentence.")]
        chunks = chunker.chunk(pages)
        assert len(chunks) >= 1
        assert isinstance(chunks[0], RawChunk)
        assert chunks[0].sequence == 0

    def test_chunk_ids_sequential(self):
        chunker = SemanticChunker(max_tokens=10, overlap_tokens=2)
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        pages = [(1, text)]
        chunks = chunker.chunk(pages)
        for i, chunk in enumerate(chunks):
            assert chunk.sequence == i
            assert chunk.chunk_id == f"chunk_{i:05d}"

    def test_chunk_has_text(self):
        chunker = SemanticChunker()
        pages = [(1, "The quick brown fox jumps over the lazy dog.")]
        chunks = chunker.chunk(pages)
        assert all(len(c.text) > 0 for c in chunks)

    def test_token_count_positive(self):
        chunker = SemanticChunker()
        pages = [(1, "A document with multiple words and content.")]
        chunks = chunker.chunk(pages)
        assert all(c.token_count > 0 for c in chunks)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
