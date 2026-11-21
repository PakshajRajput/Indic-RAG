"""Deterministic stub implementations for testing IndicRAG agents and graphs.

Enables full offline regression testing of:
- State transitions and typed hand-offs
- Hard budget enforcement (max 8 LLM calls, max 2 retrieval rounds)
- Bounded retries and termination conditions
- Grounding verification and citation checking
- Abstention reason codes
"""

from __future__ import annotations

from typing import Any

from indicrag.generation.generator import AnswererGenerator, GenerationResult


class StubAnswererGenerator(AnswererGenerator):
    """Deterministic stub generator for testing AnswererAgent."""

    def __init__(
        self,
        canned_answers: dict[str, str] | None = None,
        default_answer: str = "[psg_001] नई दिल्ली भारत की राजधानी है।",
        always_refuse: bool = False,
    ) -> None:
        super().__init__(model_name="stub/deterministic-answerer", cache_dir="scratch/stub_cache")
        self.canned_answers = canned_answers or {}
        self.default_answer = default_answer
        self.always_refuse = always_refuse
        self.call_count = 0

    def generate(
        self,
        query_id: str,
        question: str,
        retrieved_passages: list[dict[str, Any]],
        script: str = "hindi",
        **kwargs: Any,
    ) -> GenerationResult:
        """Return deterministic test response."""
        self.call_count += 1

        if self.always_refuse:
            return GenerationResult(
                query_id=query_id,
                model_name=self.model_name,
                prompt="stub_prompt",
                response_text="दिए गए संदर्भ में जानकारी उपलब्ध नहीं है।",
                is_abstention=True,
                prompt_tokens=50,
                completion_tokens=15,
                latency_ms=1.5,
                is_cache_hit=False,
            )

        if query_id in self.canned_answers:
            resp = self.canned_answers[query_id]
        else:
            # If retrieved passages exist, cite the first passage
            if retrieved_passages:
                first_pid = retrieved_passages[0].get("passage_id", "psg_001")
                resp = f"[{first_pid}] {self.default_answer}"
            else:
                resp = "दिए गए संदर्भ में जानकारी उपलब्ध नहीं है।"

        is_abs = "उपलब्ध नहीं है" in resp or "नहीं पता" in resp

        return GenerationResult(
            query_id=query_id,
            model_name=self.model_name,
            prompt="stub_prompt",
            response_text=resp,
            is_abstention=is_abs,
            prompt_tokens=60,
            completion_tokens=20,
            latency_ms=2.0,
            is_cache_hit=False,
        )


def create_stub_retriever(
    canned_results: dict[str, list[tuple[str, float]]] | None = None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Create a deterministic stub retriever and passage lookup for tests."""
    canned = canned_results or {
        "default": [("psg_001", 0.08), ("psg_002", 0.05), ("psg_003", 0.02)],
    }

    passage_lookup: dict[str, dict[str, Any]] = {
        "psg_001": {
            "passage_id": "psg_001",
            "title": "भारत की राजधानी",
            "text": "नई दिल्ली भारत की आधिकारिक राजधानी है। यह उत्तरी भारत में स्थित है।",
        },
        "psg_002": {
            "passage_id": "psg_002",
            "title": "दिल्ली का इतिहास",
            "text": "दिल्ली एक ऐतिहासिक नगर है जो यमुना नदी के तट पर बसा है।",
        },
        "psg_003": {
            "passage_id": "psg_003",
            "title": "अप्रासंगिक लेख",
            "text": "यह एक अन्य विषय पर लिखा गया असंबंधित अनुच्छेद है।",
        },
    }

    def retriever_func(query: str, top_k: int) -> list[tuple[str, float]]:
        if query in canned:
            return canned[query][:top_k]
        return canned["default"][:top_k]

    return retriever_func, passage_lookup
