"""Retriever Agent for PJ-09 IndicRAG-QA.

Handles:
- Orchestration of hybrid/dense/BM25 retrieval over the search corpus
- Enforcement of retrieval round budget (max 2 rounds)
- Passage caching and ranking telemetry
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from indicrag.agents.state import IndicRAGState, ReasonCode, SystemLadderConfig


class RetrieverAgent:
    """Agent executing retrieval over the indexed passage collection."""

    def __init__(
        self,
        retriever_func: Callable[[str, int], list[tuple[str, float]]] | None = None,
        passage_lookup: dict[str, dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever_func = retriever_func
        self.passage_lookup = passage_lookup or {}
        self.top_k = top_k

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Execute retrieval round."""
        # Enforce retrieval round budget
        if state.retrieval_rounds >= state.max_retrieval_rounds:
            state.is_abstention = True
            state.reason_code = ReasonCode.BUDGET_EXHAUSTED
            state.final_answer = "अधिकतम पुनर्प्राप्ति राउंड सीमा (2) समाप्त हो गई।"
            state.agent_history.append("retriever_agent:budget_exhausted")
            return state

        query_to_run = state.active_query or state.question
        state.retrieval_rounds += 1

        passages: list[dict[str, Any]] = []
        scores: dict[str, float] = {}

        if self.retriever_func:
            results = self.retriever_func(query_to_run, self.top_k)
            for pid, score in results:
                scores[pid] = float(score)
                if pid in self.passage_lookup:
                    psg_data = dict(self.passage_lookup[pid])
                    psg_data["score"] = float(score)
                    passages.append(psg_data)
                else:
                    passages.append({"passage_id": pid, "title": "", "text": "", "score": float(score)})
        else:
            # Fallback or stub mode: keep existing passages if already provided
            passages = state.retrieved_passages
            scores = state.retrieval_scores

        state.retrieved_passages = passages
        state.retrieval_scores = scores

        # For S0, S1, S2 (which skip EvidenceGrader), forward retrieved directly to graded
        if state.system_ladder in (SystemLadderConfig.S0, SystemLadderConfig.S1, SystemLadderConfig.S2):
            state.graded_passages = passages

        state.agent_history.append(f"retriever_agent:round_{state.retrieval_rounds}")
        return state


def create_retriever_node(
    retriever_func: Callable[[str, int], list[tuple[str, float]]] | None = None,
    passage_lookup: dict[str, dict[str, Any]] | None = None,
    top_k: int = 5,
) -> Callable[[IndicRAGState], IndicRAGState]:
    """Factory creating a configured LangGraph retriever node."""
    agent = RetrieverAgent(retriever_func=retriever_func, passage_lookup=passage_lookup, top_k=top_k)

    def node(state: IndicRAGState) -> IndicRAGState:
        return agent.run(state)

    return node
