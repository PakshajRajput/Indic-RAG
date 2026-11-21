"""Typed Pydantic state and enum specifications for PJ-09 IndicRAG-QA multi-agent architecture.

Enforces:
- Typed hand-offs between agents
- Script and provenance tracking
- Budget tracking (calls, rounds, tokens, latency)
- Standardized reason codes for abstention
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScriptType(str, Enum):
    """Detected script type."""
    DEVANAGARI = "devanagari"
    ROMAN_HINGLISH = "roman_hinglish"
    LATIN_ENGLISH = "latin_english"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LanguageType(str, Enum):
    """Detected language."""
    HINDI = "hi"
    HINGLISH = "hi-Latn"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class VerificationVerdict(str, Enum):
    """Outcome from the Verifier Agent."""
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTORY = "CONTRADICTORY"
    NO_EVIDENCE = "NO_EVIDENCE"
    ANSWERER_REFUSED = "ANSWERER_REFUSED"


class ReasonCode(str, Enum):
    """Typed reason codes for abstention (AGENTS.md Rule 12)."""
    NO_EVIDENCE = "NO_EVIDENCE"
    UNSUPPORTED_ANSWER = "UNSUPPORTED_ANSWER"
    ANSWERER_REFUSED = "ANSWERER_REFUSED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"


class SystemLadderConfig(str, Enum):
    """Experimental systems ladder configurations (S0-S5)."""
    S0 = "S0"  # Single-pass RAG (Retriever -> Answerer)
    S1 = "S1"  # Score Gate (Retriever -> Score Check -> Answerer / Abstain)
    S2 = "S2"  # Verifier (Retriever -> Answerer -> Verifier)
    S3 = "S3"  # Evidence Grader (Retriever -> Grader -> Answerer -> Verifier)
    S4 = "S4"  # Script/Lang + Query Agent + Grader + Answerer + Verifier
    S5 = "S5"  # Full Multi-Agent with Abstention Arbiter and Bounded Retry


class PassageRecord(BaseModel):
    """Typed representation of a passage during agent processing."""
    passage_id: str
    title: str = ""
    text: str
    score: float = 0.0
    is_relevant: bool = True
    grading_reason: str = ""


class IndicRAGState(BaseModel):
    """Unified typed state for the IndicRAG LangGraph serving pipeline.

    Pydantic schema validated at every node transition.
    """
    # Query Identifiers & Provenance
    query_id: str
    question: str
    script: ScriptType = ScriptType.UNKNOWN
    language: LanguageType = LanguageType.UNKNOWN
    synthetic_hinglish_link: str | None = None  # Original Hindi ID if synthetic Hinglish

    # Active System Configuration
    system_ladder: SystemLadderConfig = SystemLadderConfig.S5

    # Query Reformulation / Transliteration
    active_query: str = ""
    reformulated_queries: list[str] = Field(default_factory=list)

    # Retrieval State
    retrieval_rounds: int = 0
    retrieved_passages: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    graded_passages: list[dict[str, Any]] = Field(default_factory=list)

    # Answerer State
    draft_answer: str | None = None
    cited_passage_ids: list[str] = Field(default_factory=list)

    # Verifier State
    verification_verdict: VerificationVerdict | None = None
    verification_details: dict[str, Any] = Field(default_factory=dict)
    unsupported_claims: list[str] = Field(default_factory=list)

    # Final Decision & Abstention
    final_answer: str | None = None
    is_abstention: bool = False
    reason_code: ReasonCode | None = None
    confidence_score: float = 0.0

    # Hard Execution Budgets
    llm_calls: int = 0
    max_llm_calls: int = 8
    max_retrieval_rounds: int = 2
    retry_count: int = 0
    max_retries: int = 1
    needs_retry: bool = False  # Schema-backed flag for LangGraph state propagation across nodes

    # Resource & Telemetry Tracking
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    agent_history: list[str] = Field(default_factory=list)
    traces: list[dict[str, Any]] = Field(default_factory=list)  # Granular structured node traces
    errors: list[str] = Field(default_factory=list)

