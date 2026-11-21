"""Non-agentic Score Gate for PJ-09 IndicRAG-QA (System S1).

Inspects retrieval scores without an LLM call:
- If max retrieval score is below a strict threshold (or passages empty), immediately abstains with NO_EVIDENCE.
- Otherwise allows execution to proceed to the Answerer.
"""

from __future__ import annotations

from indicrag.agents.state import IndicRAGState, ReasonCode


class ScoreGate:
    """Non-agentic score-based gating mechanism."""

    def __init__(self, score_threshold: float = 0.005) -> None:
        self.score_threshold = score_threshold

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Evaluate retrieval scores against the gating threshold."""
        passages = state.retrieved_passages
        scores = state.retrieval_scores

        if not passages or not scores:
            state.is_abstention = True
            state.reason_code = ReasonCode.NO_EVIDENCE
            state.final_answer = "पुनर्प्राप्ति में कोई साक्ष्य प्राप्त नहीं हुआ।"
            state.agent_history.append("score_gate:rejected_empty")
            return state

        max_score = max(scores.values()) if scores else 0.0

        if max_score < self.score_threshold:
            state.is_abstention = True
            state.reason_code = ReasonCode.NO_EVIDENCE
            state.final_answer = f"पुनर्प्राप्ति प्रासंगिकता स्कोर ({max_score:.4f}) आवश्यक सीमा ({self.score_threshold}) से कम है।"
            state.agent_history.append(f"score_gate:rejected_low_score_{max_score:.4f}")
            return state

        state.agent_history.append(f"score_gate:passed_score_{max_score:.4f}")
        return state


def score_gate_node(state: IndicRAGState, score_threshold: float = 0.005) -> IndicRAGState:
    """LangGraph node wrapper for ScoreGate."""
    gate = ScoreGate(score_threshold=score_threshold)
    return gate.run(state)
