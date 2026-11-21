"""Script & Language Agent for PJ-09 IndicRAG-QA.

Identifies script and language:
- Devanagari Hindi
- Roman-script Hinglish
- Latin-script English
Preserves question text, provenance, and script markers.
"""

from __future__ import annotations

import re

from indicrag.agents.state import IndicRAGState, LanguageType, ScriptType

# Distinctive phonetic Hindi function words in Roman script (exclude ambiguous English words like 'the', 'me', 'to')
HINGLISH_LEXICAL_MARKERS = {
    "kya", "kyu", "kyun", "kaun", "kahan", "kab", "kaise", "kitna", "kitne", "kitni",
    "hai", "hain", "tha", "thi", "hoga", "hogi", "hoge", "raha", "rahi", "rahe",
    "mein", "ko", "ka", "ki", "ke", "ne", "par", "pe", "aur", "bhi",
    "nahi", "nahin", "mat", "kuch", "apna", "apne", "apni", "yeh", "woh", "unka",
    "inka", "batao", "bataiye", "kijiye", "karo", "diya", "liya", "gaya", "gayi", "gaye",
}

COMMON_ENGLISH_WORDS = {
    "what", "is", "the", "capital", "of", "in", "to", "for", "with", "on", "at",
    "from", "by", "about", "as", "into", "like", "through", "after", "over", "between",
    "out", "against", "during", "without", "before", "under", "around", "among", "who", "which",
    "where", "when", "why", "how", "are", "was", "were", "been", "being", "have", "has", "had",
}


class ScriptLanguageAgent:
    """Agent determining script and linguistic characteristics of incoming queries."""

    def detect_script_and_language(self, text: str) -> tuple[ScriptType, LanguageType]:
        """Detect script and language using character ranges and lexical heuristics."""
        if not text or not text.strip():
            return ScriptType.UNKNOWN, LanguageType.UNKNOWN

        # Count character frequencies
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        total_letters = devanagari_chars + latin_chars

        if total_letters == 0:
            return ScriptType.UNKNOWN, LanguageType.UNKNOWN

        devanagari_ratio = devanagari_chars / total_letters

        # Dominant Devanagari
        if devanagari_ratio >= 0.40:
            return ScriptType.DEVANAGARI, LanguageType.HINDI

        # Dominant Latin
        tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]+\b", text)]
        hinglish_hits = sum(1 for t in tokens if t in HINGLISH_LEXICAL_MARKERS)
        english_hits = sum(1 for t in tokens if t in COMMON_ENGLISH_WORDS)

        if hinglish_hits > english_hits or (hinglish_hits > 0 and english_hits == 0) or devanagari_ratio > 0.05:
            # Roman Hinglish with Hindi markers or code-mixing
            return ScriptType.ROMAN_HINGLISH, LanguageType.HINGLISH

        # Pure English or unclassified Latin
        return ScriptType.LATIN_ENGLISH, LanguageType.ENGLISH

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Process state and set script, language, and initial active query."""
        script, lang = self.detect_script_and_language(state.question)
        state.script = script
        state.language = lang
        state.active_query = state.question
        state.agent_history.append("script_detector")
        return state


def script_language_node(state: IndicRAGState) -> IndicRAGState:
    """LangGraph node wrapper for ScriptLanguageAgent."""
    agent = ScriptLanguageAgent()
    return agent.run(state)
