"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 2  SEMANTIC CHUNKER
  spaCy sentence-boundary + tiktoken counting;
  512-tok window, 64-tok overlap.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from sentiment_analysis.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    SPACY_AVAILABLE,
    _NLP,
    count_tokens,
)


class RawChunk:
    """Intermediate pre-inference chunk representation."""

    __slots__ = (
        "chunk_id",
        "sequence",
        "page_number",
        "text",
        "token_count",
        "char_offset_start",
        "char_offset_end",
    )

    def __init__(
        self,
        sequence: int,
        text: str,
        page_number: Optional[int] = None,
        char_offset_start: int = 0,
    ) -> None:
        self.chunk_id = f"chunk_{sequence:05d}"
        self.sequence = sequence
        self.page_number = page_number
        self.text = text
        self.token_count = count_tokens(text)
        self.char_offset_start = char_offset_start
        self.char_offset_end = char_offset_start + len(text)


class SemanticChunker:
    """
    Splits pages into semantically coherent chunks with overlap.

    Strategy:
      1. Use spaCy sentence boundary detection (if available).
      2. Accumulate sentences until CHUNK_MAX_TOKENS is reached.
      3. Slide back CHUNK_OVERLAP_TOKENS to preserve cross-boundary context.
    """

    def __init__(
        self,
        max_tokens: int = CHUNK_MAX_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, pages: List[Tuple[int, str]]) -> List[RawChunk]:
        sentences: List[Tuple[int, str]] = []

        for page_num, text in pages:
            if SPACY_AVAILABLE and _NLP is not None:
                doc = _NLP(text[:100_000])  # spaCy limit guard
                for sent in doc.sents:
                    s = sent.text.strip()
                    if s:
                        sentences.append((page_num, s))
            else:
                # Fallback: split on period/newline
                for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
                    s = s.strip()
                    if s:
                        sentences.append((page_num, s))

        chunks: List[RawChunk] = []
        buf: List[str] = []
        buf_tokens: int = 0
        buf_page: Optional[int] = None
        char_offset: int = 0
        seq: int = 0

        def _flush() -> None:
            nonlocal buf, buf_tokens, seq, char_offset
            if not buf:
                return
            text_block = " ".join(buf)
            chunks.append(
                RawChunk(
                    sequence=seq,
                    text=text_block,
                    page_number=buf_page,
                    char_offset_start=char_offset,
                )
            )
            char_offset += len(text_block)
            seq += 1

        overlap_buf: List[str] = []
        overlap_tok_count: int = 0

        for page_num, sent in sentences:
            s_tokens = count_tokens(sent)

            if buf_tokens + s_tokens > self.max_tokens and buf:
                _flush()
                # Seed next chunk with overlap
                buf = list(overlap_buf)
                buf_tokens = overlap_tok_count
                buf_page = page_num

            buf.append(sent)
            buf_tokens += s_tokens
            buf_page = buf_page or page_num

            # Maintain rolling overlap window
            overlap_buf.append(sent)
            overlap_tok_count += s_tokens
            while overlap_tok_count > self.overlap_tokens and overlap_buf:
                removed = overlap_buf.pop(0)
                overlap_tok_count -= count_tokens(removed)

        _flush()
        return chunks
