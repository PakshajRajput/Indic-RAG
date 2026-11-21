"""Answerer Agent for PJ-09 IndicRAG-QA.

Handles:
- Grounded answer synthesis with inline passage citations [psg_xxx]
- Enforcing refusal when evidence is absent from context
- Disk-cached model calls and budget tracking
"""

from __future__ import annotations

import re
from typing import Any

from indicrag.agents.state import IndicRAGState, ReasonCode, SystemLadderConfig
from indicrag.generation.generator import AnswererGenerator


class AnswererAgent:
    """Agent generating candidate answers grounded exclusively in provided evidence."""

    def __init__(self, generator: AnswererGenerator | None = None) -> None:
        self.generator = generator or AnswererGenerator()

    def extract_citations(self, text: str) -> list[str]:
        """Extract cited passage IDs from the answer text (e.g. [psg_1234])."""
        return re.findall(r"\[([a-zA-Z0-9_\-\.]+)\]", text)

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Synthesize answer from available evidence."""
        # Evidence source: graded_passages if available, else retrieved_passages
        passages = state.graded_passages if state.graded_passages else state.retrieved_passages

        # If no passages available, refuse immediately
        if not passages:
            state.draft_answer = "संदर्भ में जानकारी उपलब्ध नहीं है।"
            state.is_abstention = True
            state.reason_code = ReasonCode.NO_EVIDENCE
            state.agent_history.append("answerer_agent:no_passages_refused")
            if state.system_ladder in (SystemLadderConfig.S0, SystemLadderConfig.S1):
                state.final_answer = state.draft_answer
            return state

        # Call generator (disk-cached)
        script_str = "hinglish" if state.script.value == "roman_hinglish" else "hindi"
        try:
            gen_res = self.generator.generate(
                query_id=state.query_id,
                question=state.question,
                retrieved_passages=passages,
                script=script_str,
                system_id=state.system_ladder.value,
            )
        except TypeError:
            gen_res = self.generator.generate(
                query_id=state.query_id,
                question=state.question,
                retrieved_passages=passages,
                script=script_str,
            )

        state.llm_calls += 1
        state.total_tokens += gen_res.prompt_tokens + gen_res.completion_tokens
        state.total_latency_ms += gen_res.latency_ms

        state.draft_answer = gen_res.response_text
        state.cited_passage_ids = self.extract_citations(gen_res.response_text)
        state.is_abstention = gen_res.is_abstention

        if gen_res.is_abstention:
            state.reason_code = ReasonCode.ANSWERER_REFUSED

        # For S0 and S1, Answerer is the final terminal node
        if state.system_ladder in (SystemLadderConfig.S0, SystemLadderConfig.S1):
            state.final_answer = state.draft_answer

        state.agent_history.append("answerer_agent:generated")
        return state


def create_answerer_node(generator: AnswererGenerator | None = None) -> Any:
    """Factory creating an AnswererAgent node."""
    agent = AnswererAgent(generator=generator)

    def node(state: IndicRAGState) -> IndicRAGState:
        return agent.run(state)

    return node
