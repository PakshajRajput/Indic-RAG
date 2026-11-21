"""Abstention Arbiter Agent for PJ-09 IndicRAG-QA (System S5).

Handles:
- Arbitration of verification outcomes
- Bounded retry loops (max 1 retry loop, strictly budget-capped)
- Finalization of answer or assignment of typed abstention reason codes
"""

from __future__ import annotations

from indicrag.agents.state import (
    IndicRAGState,
    ReasonCode,
    VerificationVerdict,
)


class AbstentionArbiter:
    """Agent arbitrating final output, bounded retries, and explicit abstentions."""

    def __init__(self, max_retries: int = 1) -> None:
        self.max_retries = max_retries

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Arbitrate state outcome."""
        verdict = state.verification_verdict or VerificationVerdict.NO_EVIDENCE

        # 1. Supported answer: Accept and finalize
        if verdict == VerificationVerdict.SUPPORTED:
            state.final_answer = state.draft_answer
            state.is_abstention = False
            state.confidence_score = 0.95
            state.agent_history.append("abstention_arbiter:accepted")
            return state

        # 2. Answerer explicitly refused: Accept refusal
        if verdict == VerificationVerdict.ANSWERER_REFUSED:
            state.final_answer = state.draft_answer
            state.is_abstention = True
            state.reason_code = ReasonCode.ANSWERER_REFUSED
            state.confidence_score = 0.90
            state.agent_history.append("abstention_arbiter:accepted_refusal")
            return state

        # 3. No evidence available
        if verdict == VerificationVerdict.NO_EVIDENCE:
            state.is_abstention = True
            state.reason_code = ReasonCode.NO_EVIDENCE
            state.final_answer = "दिए गए संदर्भ में प्रश्न का उत्तर देने हेतु कोई साक्ष्य उपलब्ध नहीं है।"
            state.confidence_score = 0.90
            state.agent_history.append("abstention_arbiter:abstained_no_evidence")
            return state

        # 4. Unsupported answer or contradiction: evaluate bounded retry
        can_retry = (
            state.retry_count < self.max_retries
            and state.retrieval_rounds < state.max_retrieval_rounds
            and state.llm_calls <= state.max_llm_calls - 2
        )

        if can_retry:
            state.retry_count += 1
            state.needs_retry = True
            state._needs_retry = True  # Backward compatibility
            state.agent_history.append(f"abstention_arbiter:trigger_retry_{state.retry_count}")
            # Clear previous draft for fresh generation
            state.draft_answer = None
            state.cited_passage_ids = []
            state.verification_verdict = None
            return state

        # Bounded retry not possible or budget exhausted: finalize abstention
        state.is_abstention = True
        state.needs_retry = False
        state._needs_retry = False

        if state.llm_calls >= state.max_llm_calls or state.retrieval_rounds > state.max_retrieval_rounds:
            state.reason_code = ReasonCode.BUDGET_EXHAUSTED
            state.final_answer = "निष्पादन बजट सीमा समाप्त होने के कारण उत्तर सत्यापित नहीं किया जा सका।"
        elif state.retry_count < self.max_retries and (
            state.retrieval_rounds >= state.max_retrieval_rounds or state.llm_calls > state.max_llm_calls - 2
        ):
            # Wanted to retry but was blocked by round or LLM call budget limits
            state.reason_code = ReasonCode.BUDGET_EXHAUSTED
            state.final_answer = "निष्पादन बजट सीमा समाप्त होने के कारण पुनः प्रयास नहीं किया जा सका।"
        elif verdict == VerificationVerdict.CONTRADICTORY:
            state.reason_code = ReasonCode.CONTRADICTORY_EVIDENCE
            state.final_answer = "संदर्भ में परस्पर विरोधी साक्ष्य पाए गए, इसलिए उत्तर अस्वीकार किया जाता है।"
        else:
            state.reason_code = ReasonCode.UNSUPPORTED_ANSWER
            state.final_answer = "सत्यापन में दावे असमर्थित पाए गए, इसलिए उत्तर अस्वीकार किया जाता है।"

        state.confidence_score = 0.85
        state.agent_history.append(f"abstention_arbiter:final_abstention_{state.reason_code.value}")
        return state


def abstention_arbiter_node(state: IndicRAGState, max_retries: int = 1) -> IndicRAGState:
    """LangGraph node wrapper for AbstentionArbiter."""
    arbiter = AbstentionArbiter(max_retries=max_retries)
    return arbiter.run(state)
