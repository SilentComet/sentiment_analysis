"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  § 5  CULTURAL DECODER
  20-entry corporate-speak lexicon, 10-entry idiom
  map, passive-aggression regex, implicit intent
  patterns.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from sentiment_analysis.config import (
    CORPORATE_SPEAK_LEXICON,
    IMPLICIT_INTENT_PATTERNS,
)


class CulturalDecoder:
    """
    Decodes:
      - Corporate speak → plain English + sentiment adjustment
      - Common English idioms → literal meaning
      - Implicit intent patterns (regex matching)
      - Passive-aggression signals
    """

    _IDIOM_MAP: Dict[str, str] = {
        "bite the bullet": "endure something painful",
        "burn bridges": "destroy relationships",
        "hit the nail on the head": "be exactly correct",
        "costs an arm and a leg": "very expensive",
        "the ball is in your court": "it is your responsibility now",
        "throw under the bus": "blame someone else",
        "elephant in the room": "obvious problem being ignored",
        "back to square one": "start over completely",
        "kick the can down the road": "postpone a decision",
        "move the goalposts": "change criteria unfairly",
    }

    _PASSIVE_AGGRESSION_SIGNALS: List[str] = [
        r"\bper my (last|previous) (email|message)\b",
        r"\bas (previously|already) (mentioned|stated|noted)\b",
        r"\bwould have thought\b",
        r"\bjust to (clarify|confirm|reiterate)\b",
        r"\bfor (your|future) reference\b",
    ]

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        adjustments: List[str] = []
        detected_corp: List[str] = []
        detected_idioms: List[str] = []
        score_adjustment: float = 0.0
        implicit_intent: Optional[str] = None

        # 1. Corporate speak
        for phrase, meaning in CORPORATE_SPEAK_LEXICON.items():
            if phrase in text_lower:
                detected_corp.append(phrase)
                adjustments.append(
                    f"Corporate speak '{phrase}' decoded as: {meaning}"
                )
                # Most corporate speak is mildly negative (euphemism)
                score_adjustment -= 0.05

        # 2. Idioms
        for idiom, meaning in self._IDIOM_MAP.items():
            if idiom in text_lower:
                detected_idioms.append(idiom)
                adjustments.append(f"Idiom '{idiom}' decoded as: {meaning}")

        # 3. Implicit intent
        for intent, patterns in IMPLICIT_INTENT_PATTERNS.items():
            if any(re.search(p, text, re.I) for p in patterns):
                implicit_intent = intent
                adjustments.append(f"Implicit intent detected: {intent}")
                break

        # 4. Passive aggression
        pa_hits = sum(
            1 for p in self._PASSIVE_AGGRESSION_SIGNALS if re.search(p, text, re.I)
        )
        if pa_hits:
            adjustments.append(
                f"Passive-aggressive tone detected ({pa_hits} signals)"
            )
            score_adjustment -= 0.10 * pa_hits

        return {
            "detected_corp_speak": detected_corp,
            "detected_idioms": detected_idioms,
            "score_adjustment": float(np.clip(score_adjustment, -0.5, 0.0)),
            "adjustments": adjustments,
            "implicit_intent": implicit_intent,
        }
