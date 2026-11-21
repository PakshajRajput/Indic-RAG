"""Evidence Grader Agent for PJ-09 IndicRAG-QA (Systems S3, S4, S5).

Evaluates retrieved passages for:
- Topical relevance to the question
- Filtering of counterfactual distractors and noise passages
- Verification that sufficient answerable evidence exists
"""

from __future__ import annotations

from typing import Any

from indicrag.agents.state import IndicRAGState, ReasonCode
from indicrag.eval.metrics import tokenize_indic


def compute_lexical_overlap(text: str, query: str) -> float:
    """Compute character and token overlap between passage and query."""
    text_words = set(tokenize_indic(text.lower()))
    query_words = set(tokenize_indic(query.lower()))

    # Ignore common interrogatives and stop words
    stopwords = {"क्या", "कौन", "कब", "कहाँ", "कैसे", "कितना", "है", "था", "थी", "थे", "हैं", "का", "की", "के", "में", "से", "ko"}
    content_query_words = query_words - stopwords

    if not content_query_words:
        content_query_words = query_words

    if not content_query_words:
        return 0.0

    overlap = len(content_query_words & text_words) / len(content_query_words)
    return overlap


class EvidenceGraderAgent:
    """Agent grading retrieved evidence passages prior to synthesis."""

    def __init__(self, relevance_threshold: float = 0.20, llm_client: Any | None = None) -> None:
        self.relevance_threshold = relevance_threshold
        self.llm_client = llm_client

    def grade_passages(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Grade passages into relevant and filtered subsets."""
        relevant: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []

        for p in passages:
            text = p.get("text", "")
            title = p.get("title", "")
            combined = f"{title} {text}".strip()

            overlap = compute_lexical_overlap(combined, query)
            p_copy = dict(p)
            p_copy["overlap_score"] = overlap

            # If passage contains key content words or high retrieval score
            if overlap >= self.relevance_threshold or p.get("score", 0.0) >= 0.015:
                p_copy["is_relevant"] = True
                p_copy["grading_reason"] = f"Relevant (overlap={overlap:.2f}, score={p.get('score', 0.0):.4f})"
                relevant.append(p_copy)
            else:
                p_copy["is_relevant"] = False
                p_copy["grading_reason"] = f"Noise/Distractor (overlap={overlap:.2f} < {self.relevance_threshold})"
                filtered.append(p_copy)

        return relevant, filtered

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Execute evidence grading."""
        query = state.active_query or state.question
        relevant, filtered = self.grade_passages(query, state.retrieved_passages)

        state.graded_passages = relevant

        # If zero passages survive grading, abstain with NO_EVIDENCE
        if not relevant:
            state.is_abstention = True
            state.reason_code = ReasonCode.NO_EVIDENCE
            state.final_answer = "पुनर्प्राप्त किए गए अनुच्छेदों में प्रश्न से संबंधित कोई प्रासंगिक साक्ष्य नहीं मिला।"
            state.agent_history.append("evidence_grader:zero_relevant_passages")
            return state

        state.agent_history.append(f"evidence_grader:passed_{len(relevant)}_passages")
        return state


def evidence_grader_node(state: IndicRAGState, relevance_threshold: float = 0.20) -> IndicRAGState:
    """LangGraph node wrapper for EvidenceGraderAgent."""
    grader = EvidenceGraderAgent(relevance_threshold=relevance_threshold)
    return grader.run(state)
