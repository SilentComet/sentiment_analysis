"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 3  COMPLEXITY SCORER
  6-signal weighted scorer — negation, syntax depth,
  sarcasm regex, TTR entropy, corporate-speak density,
  sentence-length variance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from sentiment_analysis.config import (
    CORPORATE_SPEAK_LEXICON,
    SARCASM_SIGNALS,
    SPACY_AVAILABLE,
    _NLP,
)


class ComplexityScorer:
    """
    Scores each chunk 0→1 on multiple signals:

    Signal                   Weight
    ──────────────────────────────
    Negation density           0.20   "not", "never", double negation
    Syntactic depth            0.20   subordinate clause count (spaCy)
    Sarcasm signal hits        0.25   regex pattern matches
    Lexical entropy            0.15   type/token ratio
    Corporate speak density    0.10   hits against CORPORATE_SPEAK_LEXICON
    Sentence length variance   0.10   stddev of sentence lengths
    """

    _NEGATION_TOKENS = {
        "not", "never", "no", "neither", "nor",
        "without", "hardly", "barely",
    }

    _WEIGHTS: Dict[str, float] = {
        "negation": 0.20,
        "syntax": 0.20,
        "sarcasm": 0.25,
        "entropy": 0.15,
        "corp_speak": 0.10,
        "len_variance": 0.10,
    }

    def score(self, text: str) -> float:
        signals: Dict[str, float] = {}

        tokens = text.lower().split()
        if not tokens:
            return 0.0

        # 1. Negation density
        neg_count = sum(1 for t in tokens if t in self._NEGATION_TOKENS)
        signals["negation"] = min(neg_count / max(len(tokens) * 0.05, 1), 1.0)

        # 2. Syntactic depth (spaCy)
        if SPACY_AVAILABLE and _NLP is not None:
            doc = _NLP(text[:5_000])
            subord_count = sum(
                1 for tok in doc if tok.dep_ in ("advcl", "relcl", "ccomp", "xcomp")
            )
            signals["syntax"] = min(
                subord_count / max(len(list(doc.sents)) * 2, 1), 1.0
            )
        else:
            # Heuristic: count subordinating conjunctions
            subordinators = {
                "although", "because", "since", "unless",
                "whereas", "while", "if", "when", "though",
            }
            sub_count = sum(1 for t in tokens if t in subordinators)
            signals["syntax"] = min(sub_count / max(len(tokens) * 0.02, 1), 1.0)

        # 3. Sarcasm signals
        hits = sum(
            1 for pattern in SARCASM_SIGNALS if re.search(pattern, text, re.I)
        )
        signals["sarcasm"] = min(hits / 2.0, 1.0)

        # 4. Lexical entropy (type-token ratio)
        ttr = len(set(tokens)) / max(len(tokens), 1)
        signals["entropy"] = 1.0 - ttr  # high variety = less repetitive = more complex

        # 5. Corporate speak density
        corp_hits = sum(
            1 for phrase in CORPORATE_SPEAK_LEXICON if phrase in text.lower()
        )
        signals["corp_speak"] = min(corp_hits / 5.0, 1.0)

        # 6. Sentence length variance
        sent_lengths = [
            len(s.split()) for s in re.split(r"[.!?]+", text) if s.strip()
        ]
        if len(sent_lengths) > 1:
            stddev = float(np.std(sent_lengths))
            signals["len_variance"] = min(stddev / 20.0, 1.0)
        else:
            signals["len_variance"] = 0.0

        return sum(signals.get(k, 0.0) * w for k, w in self._WEIGHTS.items())
