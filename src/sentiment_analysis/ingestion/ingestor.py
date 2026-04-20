"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 1  DOCUMENT INGESTOR
  Multi-format ingestion; splits on \\f (PDF page
  breaks) or JSON page arrays.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple


class DocumentIngestor:
    """
    Handles multiple document formats.
    Produces a list of (page_number, raw_text) tuples.

    Supported inputs:
      • Plain text  — whole document treated as page 1 (or split on \\f)
      • JSON array  — [{page: N, text: "..."}, ...] pre-serialized format
      • PDF text    — form-feed (\\f) separated pages

    Production extension points: PyMuPDF (fitz), python-docx, etc.
    """

    @staticmethod
    def ingest(
        content: str, filename: Optional[str] = None
    ) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Returns (detected_language, [(page_num, page_text), ...]).
        For plain text, treats the whole document as page 1.
        """
        # ── JSON format ───────────────────────────────────────────────────
        if filename and filename.lower().endswith(".json"):
            try:
                pages_data = json.loads(content)
                if isinstance(pages_data, list) and pages_data and "text" in pages_data[0]:
                    pages = [
                        (int(p.get("page", i + 1)), p["text"])
                        for i, p in enumerate(pages_data)
                    ]
                    return "en", pages
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # ── Plain text / PDF page breaks ──────────────────────────────────
        raw_pages = content.split("\f")
        pages = [
            (i + 1, page.strip())
            for i, page in enumerate(raw_pages)
            if page.strip()
        ]
        if not pages:
            pages = [(1, content)]

        return "en", pages
