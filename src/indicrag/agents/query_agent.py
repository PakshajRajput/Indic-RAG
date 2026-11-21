"""Query Agent for PJ-09 IndicRAG-QA.

Handles:
- Transliteration / normalization of Roman Hinglish queries into Devanagari
- Query reformulation upon verification failure or retries
- Tracking of candidate search queries
"""

from __future__ import annotations

import re
from typing import Any

from indicrag.agents.state import IndicRAGState, ScriptType

# Basic rule-based reverse phonetic mapping for common Hinglish tokens
# Enables deterministic transliteration without requiring heavy external models
COMMON_XLIT_DICT: dict[str, str] = {
    "kya": "क्या",
    "kyu": "क्यों",
    "kyun": "क्यों",
    "kaun": "कौन",
    "kahan": "कहाँ",
    "kab": "कब",
    "kaise": "कैसे",
    "kitna": "कितना",
    "kitne": "कितने",
    "kitni": "कितनी",
    "hai": "है",
    "hain": "हैं",
    "tha": "था",
    "thi": "थी",
    "the": "थे",
    "bharat": "भारत",
    "delhi": "दिल्ली",
    "rajdhani": "राजधानी",
    "naam": "नाम",
    "pratham": "प्रथम",
    "rashtrapati": "राष्ट्रपति",
    "pradhanmantri": "प्रधानमंत्री",
    "sanvidhan": "संविधान",
    "mein": "में",
    "me": "में",
    "se": "से",
    "ko": "को",
    "ka": "का",
    "ki": "की",
    "ke": "के",
    "par": "पर",
    "aur": "और",
}


def rule_based_hinglish_to_hindi(text: str) -> str:
    """Deterministic rule-based transliteration of common Hinglish query terms into Devanagari."""
    words = re.findall(r"\b\w+\b|[^\w\s]", text)
    converted: list[str] = []
    for w in words:
        low = w.lower()
        if low in COMMON_XLIT_DICT:
            converted.append(COMMON_XLIT_DICT[low])
        else:
            converted.append(w)
    return " ".join(converted)


class QueryAgent:
    """Agent for query reformulation, transliteration, and search adaptation."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    def transliterate_or_normalize(self, query: str, script: ScriptType) -> str:
        """Transliterate Roman Hinglish queries to Devanagari if applicable."""
        if script == ScriptType.ROMAN_HINGLISH:
            # Check if live hosted normalization via Groq is available
            import os
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    import json
                    import urllib.request
                    payload = json.dumps({
                        "model": "qwen/qwen3.8-27b",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a Hindi transliteration tool. Convert Roman Hinglish queries into pure Devanagari Hindi text for search indexing. Output ONLY the Devanagari text, with no explanations, notes, or punctuation changes."
                            },
                            {"role": "user", "content": query}
                        ],
                        "temperature": 0.0,
                        "max_tokens": 128
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        "https://api.groq.com/openai/v1/chat/completions",
                        data=payload,
                        headers={"Authorization": f"Bearer {groq_key}", "User-Agent": "IndicRAG/1.0", "Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        txt = data["choices"][0]["message"].get("content", "").strip()
                        if txt:
                            return txt
                except Exception:
                    pass

            xlit = rule_based_hinglish_to_hindi(query)
            if xlit != query:
                return xlit
        return query

    def reformulate_for_retry(self, original_query: str, active_query: str, failure_reason: str) -> str:
        """Generate an alternative search formulation for a retry round."""
        # Simple deterministic noun-phrase / entity-oriented extraction
        # Strip interrogative terms to improve lexical retrieval
        stopwords = {"क्या", "कौन", "कब", "कहाँ", "कैसे", "कितना", "है", "था", "थी", "थे", "हैं", "बताइए", "kya", "kaun", "kab"}
        tokens = [t for t in active_query.split() if t not in stopwords]
        if tokens and len(tokens) >= 2:
            return " ".join(tokens)
        return original_query

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Process state and set active search query."""
        # Check if this is a retry round
        is_retry = getattr(state, "needs_retry", False) or getattr(state, "_needs_retry", False)
        if is_retry:
            new_query = self.reformulate_for_retry(
                state.question,
                state.active_query,
                str(state.reason_code or "UNSUPPORTED_ANSWER"),
            )
            state.active_query = new_query
            state.reformulated_queries.append(new_query)
            # Reset retry trigger flags
            state.needs_retry = False
            state._needs_retry = False
            state.agent_history.append("query_agent:retry_reformulation")
            return state

        # Initial formulation
        transformed_query = self.transliterate_or_normalize(state.question, state.script)
        state.active_query = transformed_query
        if transformed_query != state.question:
            state.reformulated_queries.append(transformed_query)

        state.agent_history.append("query_agent")
        return state


def query_agent_node(state: IndicRAGState) -> IndicRAGState:
    """LangGraph node wrapper for QueryAgent."""
    agent = QueryAgent()
    return agent.run(state)
