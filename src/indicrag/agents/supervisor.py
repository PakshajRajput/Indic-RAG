"""Supervisor Agent for PJ-09 IndicRAG-QA.

Enforces:
- Hard per-question budget (max 8 LLM calls, max 2 retrieval rounds)
- System ladder routing (S0-S5)
- Bounded retries (max 1 retry loop in S5)
- Deterministic budget-exhaustion termination with typed reason codes
"""

from __future__ import annotations

from indicrag.agents.state import IndicRAGState, ReasonCode, SystemLadderConfig


class SupervisorAgent:
    """Supervisor enforcing execution budgets, loop limits, and routing."""

    def __init__(self, max_llm_calls: int = 8, max_retrieval_rounds: int = 2) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_retrieval_rounds = max_retrieval_rounds

    def check_budget(self, state: IndicRAGState) -> tuple[bool, str]:
        """Check whether execution is within hard limits.

        Returns:
            (is_exhausted, reason_message)
        """
        if state.llm_calls >= self.max_llm_calls:
            return True, f"LLM call budget exhausted ({state.llm_calls}/{self.max_llm_calls})"
        if state.retrieval_rounds > self.max_retrieval_rounds:
            return True, f"Retrieval round budget exhausted ({state.retrieval_rounds}/{self.max_retrieval_rounds})"
        return False, "Budget OK"

    def handle_budget_exhaustion(self, state: IndicRAGState, msg: str) -> IndicRAGState:
        """Deterministic budget exhaustion handler."""
        state.is_abstention = True
        state.reason_code = ReasonCode.BUDGET_EXHAUSTED
        state.final_answer = "निष्पादन बजट सीमा समाप्त होने के कारण उत्तर नहीं दिया जा सका।"
        state.errors.append(msg)
        state.agent_history.append("supervisor:budget_exhausted")
        return state

    def route_after_script_detection(self, state: IndicRAGState) -> str:
        """Route after script detection node."""
        is_exhausted, msg = self.check_budget(state)
        if is_exhausted:
            return "budget_exhaustion"

        # S0-S3 jump directly to retriever (no QueryAgent)
        if state.system_ladder in (SystemLadderConfig.S0, SystemLadderConfig.S1, SystemLadderConfig.S2, SystemLadderConfig.S3):
            return "retriever"

        # S4 and S5 use QueryAgent
        return "query_agent"

    def route_after_retriever(self, state: IndicRAGState) -> str:
        """Route after retrieval node."""
        is_exhausted, msg = self.check_budget(state)
        if is_exhausted:
            return "budget_exhaustion"

        ladder = state.system_ladder
        if ladder == SystemLadderConfig.S0:
            return "answerer"
        if ladder == SystemLadderConfig.S1:
            return "score_gate"
        if ladder == SystemLadderConfig.S2:
            return "answerer"
        # S3, S4, S5 use EvidenceGrader
        return "evidence_grader"

    def route_after_score_gate(self, state: IndicRAGState) -> str:
        """Route after non-agentic score gate (S1)."""
        if state.is_abstention:
            return "end"
        return "answerer"

    def route_after_evidence_grader(self, state: IndicRAGState) -> str:
        """Route after evidence grader (S3-S5)."""
        is_exhausted, msg = self.check_budget(state)
        if is_exhausted:
            return "budget_exhaustion"

        if state.is_abstention:
            return "end"
        return "answerer"

    def route_after_answerer(self, state: IndicRAGState) -> str:
        """Route after answerer."""
        is_exhausted, msg = self.check_budget(state)
        if is_exhausted:
            return "budget_exhaustion"

        # S0 and S1 terminate after Answerer
        if state.system_ladder in (SystemLadderConfig.S0, SystemLadderConfig.S1):
            return "end"

        # S2-S5 route to Verifier
        return "verifier"

    def route_after_verifier(self, state: IndicRAGState) -> str:
        """Route after verifier."""
        is_exhausted, msg = self.check_budget(state)
        if is_exhausted:
            return "budget_exhaustion"

        # S2, S3, S4 terminate or finalize after Verifier
        if state.system_ladder in (SystemLadderConfig.S2, SystemLadderConfig.S3, SystemLadderConfig.S4):
            return "end"

        # S5 routes to Abstention Arbiter for bounded retry analysis
        return "abstention_arbiter"

    def route_after_arbiter(self, state: IndicRAGState) -> str:
        """Route after abstention arbiter (S5)."""
        # If arbiter flagged a retry and budget permits
        if getattr(state, "needs_retry", False) or getattr(state, "_needs_retry", False):
            if state.retry_count <= state.max_retries and state.retrieval_rounds < state.max_retrieval_rounds:
                return "query_agent"

        return "end"


def supervisor_node(state: IndicRAGState) -> IndicRAGState:
    """LangGraph node wrapper for supervisor budget validation."""
    supervisor = SupervisorAgent(max_llm_calls=state.max_llm_calls, max_retrieval_rounds=state.max_retrieval_rounds)
    is_exhausted, msg = supervisor.check_budget(state)
    if is_exhausted:
        return supervisor.handle_budget_exhaustion(state, msg)
    return state
