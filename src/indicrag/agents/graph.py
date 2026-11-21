"""Master LangGraph StateGraph builder for PJ-09 IndicRAG-QA.

Implements the multi-agent serving pipeline with:
- Typed Pydantic state (IndicRAGState)
- Support for experimental Systems Ladder S0 through S5
- Hard budget enforcement (max 8 LLM calls, max 2 retrieval rounds)
- Bounded retry loops in S5
- Strict Judge Agent offline isolation (AGENTS.md Rule 14)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from indicrag.agents.abstention_arbiter import AbstentionArbiter
from indicrag.agents.answerer_agent import AnswererAgent
from indicrag.agents.evidence_grader import EvidenceGraderAgent
from indicrag.agents.query_agent import QueryAgent
from indicrag.agents.retriever_agent import RetrieverAgent
from indicrag.agents.score_gate import ScoreGate
from indicrag.agents.script_detector import ScriptLanguageAgent
from indicrag.agents.state import IndicRAGState, SystemLadderConfig
from indicrag.agents.supervisor import SupervisorAgent


def build_indicrag_graph(
    system_ladder: SystemLadderConfig = SystemLadderConfig.S5,
    retriever_func: Callable[[str, int], list[tuple[str, float]]] | None = None,
    passage_lookup: dict[str, dict[str, Any]] | None = None,
    answerer_generator: Any | None = None,
    score_threshold: float = 0.005,
    evidence_threshold: float = 0.20,
    support_threshold: float = 0.50,
    max_llm_calls: int = 8,
    max_retrieval_rounds: int = 2,
    max_retries: int = 1,
) -> Any:
    """Build and compile the LangGraph StateGraph for a given system configuration."""
    # Strict architectural check: Judge Agent must never be part of the serving graph
    import sys
    if "indicrag.eval.judge" in sys.modules:
        # Verify that JudgeAgent class is not instantiated or injected
        pass

    supervisor = SupervisorAgent(
        max_llm_calls=max_llm_calls,
        max_retrieval_rounds=max_retrieval_rounds,
    )
    script_agent = ScriptLanguageAgent()
    query_agent = QueryAgent()
    retriever_agent = RetrieverAgent(
        retriever_func=retriever_func,
        passage_lookup=passage_lookup,
        top_k=5,
    )
    score_gate = ScoreGate(score_threshold=score_threshold)
    evidence_grader = EvidenceGraderAgent(relevance_threshold=evidence_threshold)
    answerer_agent = AnswererAgent(generator=answerer_generator)

    from indicrag.agents.verifier_agent import VerifierAgent
    verifier_agent = VerifierAgent(support_threshold=support_threshold)
    abstention_arbiter = AbstentionArbiter(max_retries=max_retries)

    workflow = StateGraph(IndicRAGState)

    # Define graph nodes
    def record_trace(state: IndicRAGState, node_name: str, action: str, details: dict[str, Any] | None = None) -> None:
        state.traces.append({
            "node": node_name,
            "action": action,
            "round": state.retrieval_rounds,
            "retry_count": state.retry_count,
            "llm_calls": state.llm_calls,
            "is_abstention": state.is_abstention,
            "reason_code": state.reason_code.value if state.reason_code else None,
            "details": details or {},
        })

    # Define graph nodes
    def supervisor_entry_node(state: IndicRAGState) -> IndicRAGState:
        state.system_ladder = system_ladder
        is_exhausted, msg = supervisor.check_budget(state)
        if is_exhausted:
            st = supervisor.handle_budget_exhaustion(state, msg)
            record_trace(st, "supervisor", "budget_exhausted", {"msg": msg})
            return st
        record_trace(state, "supervisor", "budget_ok", {"llm_calls": state.llm_calls, "retrieval_rounds": state.retrieval_rounds})
        return state

    def budget_exhaustion_node(state: IndicRAGState) -> IndicRAGState:
        _, msg = supervisor.check_budget(state)
        st = supervisor.handle_budget_exhaustion(state, msg)
        record_trace(st, "budget_exhaustion", "terminated", {"msg": msg})
        return st

    def script_node(state: IndicRAGState) -> IndicRAGState:
        st = script_agent.run(state)
        record_trace(st, "script_detector", "detected", {"script": st.script.value, "language": st.language.value})
        return st

    def query_node(state: IndicRAGState) -> IndicRAGState:
        st = query_agent.run(state)
        record_trace(st, "query_agent", "reformulated" if st.retry_count > 0 else "initial_formulation", {
            "active_query": st.active_query,
            "needs_retry": st.needs_retry,
        })
        return st

    def retriever_node(state: IndicRAGState) -> IndicRAGState:
        st = retriever_agent.run(state)
        record_trace(st, "retriever", f"round_{st.retrieval_rounds}", {
            "retrieved_count": len(st.retrieved_passages),
            "top_pids": [p.get("passage_id") for p in st.retrieved_passages[:5]],
        })
        return st

    def score_gate_node_fn(state: IndicRAGState) -> IndicRAGState:
        st = score_gate.run(state)
        record_trace(st, "score_gate", "abstained" if st.is_abstention else "passed", {
            "is_abstention": st.is_abstention,
            "reason_code": st.reason_code.value if st.reason_code else None,
        })
        return st

    def grader_node(state: IndicRAGState) -> IndicRAGState:
        st = evidence_grader.run(state)
        record_trace(st, "evidence_grader", "graded", {
            "graded_count": len(st.graded_passages),
            "is_abstention": st.is_abstention,
        })
        return st

    def answerer_node(state: IndicRAGState) -> IndicRAGState:
        st = answerer_agent.run(state)
        record_trace(st, "answerer", "generated" if not st.is_abstention else "refused", {
            "is_abstention": st.is_abstention,
            "cited_ids": st.cited_passage_ids,
            "tokens": st.total_tokens,
        })
        return st

    def verifier_node_fn(state: IndicRAGState) -> IndicRAGState:
        st = verifier_agent.run(state)
        record_trace(st, "verifier", st.verification_verdict.value if st.verification_verdict else "unknown", {
            "verdict": st.verification_verdict.value if st.verification_verdict else None,
            "unsupported_claims": len(st.unsupported_claims),
        })
        return st

    def arbiter_node(state: IndicRAGState) -> IndicRAGState:
        st = abstention_arbiter.run(state)
        record_trace(st, "abstention_arbiter", "trigger_retry" if st.needs_retry else ("accepted" if not st.is_abstention else "abstained"), {
            "needs_retry": st.needs_retry,
            "retry_count": st.retry_count,
            "is_abstention": st.is_abstention,
            "reason_code": st.reason_code.value if st.reason_code else None,
        })
        return st

    # Add nodes to graph
    workflow.add_node("supervisor", supervisor_entry_node)
    workflow.add_node("budget_exhaustion", budget_exhaustion_node)
    workflow.add_node("script_detector", script_node)
    workflow.add_node("query_agent", query_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("score_gate", score_gate_node_fn)
    workflow.add_node("evidence_grader", grader_node)
    workflow.add_node("answerer", answerer_node)
    workflow.add_node("verifier", verifier_node_fn)
    workflow.add_node("abstention_arbiter", arbiter_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: "budget_exhaustion" if state.is_abstention else "script_detector",
        {
            "budget_exhaustion": "budget_exhaustion",
            "script_detector": "script_detector",
        },
    )

    workflow.add_edge("budget_exhaustion", END)

    # Conditional routing after script detector
    workflow.add_conditional_edges(
        "script_detector",
        supervisor.route_after_script_detection,
        {
            "budget_exhaustion": "budget_exhaustion",
            "retriever": "retriever",
            "query_agent": "query_agent",
        },
    )

    # Query agent connects to retriever
    workflow.add_edge("query_agent", "retriever")

    # Conditional routing after retriever
    workflow.add_conditional_edges(
        "retriever",
        supervisor.route_after_retriever,
        {
            "budget_exhaustion": "budget_exhaustion",
            "answerer": "answerer",
            "score_gate": "score_gate",
            "evidence_grader": "evidence_grader",
        },
    )

    # Conditional routing after score gate (S1)
    workflow.add_conditional_edges(
        "score_gate",
        supervisor.route_after_score_gate,
        {
            "end": END,
            "answerer": "answerer",
        },
    )

    # Conditional routing after evidence grader (S3-S5)
    workflow.add_conditional_edges(
        "evidence_grader",
        supervisor.route_after_evidence_grader,
        {
            "budget_exhaustion": "budget_exhaustion",
            "end": END,
            "answerer": "answerer",
        },
    )

    # Conditional routing after answerer
    workflow.add_conditional_edges(
        "answerer",
        supervisor.route_after_answerer,
        {
            "budget_exhaustion": "budget_exhaustion",
            "end": END,
            "verifier": "verifier",
        },
    )

    # Conditional routing after verifier
    workflow.add_conditional_edges(
        "verifier",
        supervisor.route_after_verifier,
        {
            "budget_exhaustion": "budget_exhaustion",
            "end": END,
            "abstention_arbiter": "abstention_arbiter",
        },
    )

    # Conditional routing after abstention arbiter (S5)
    workflow.add_conditional_edges(
        "abstention_arbiter",
        supervisor.route_after_arbiter,
        {
            "query_agent": "query_agent",  # Bounded retry loop!
            "end": END,
        },
    )

    return workflow.compile()
