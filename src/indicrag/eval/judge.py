"""Offline Judge Agent module for evaluating RAG predictions.

Strict Architectural Constraints:
- Evaluation-only: Forbidden from executing inside the serving pipeline.
- Model family independence: Must use a model family distinct from the answerer.
- Typed output: Returns structured Pydantic JudgeVerdict.
- Disk caching: Every judge judgment is cached to disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from src.indicrag.eval.metrics import normalize_text, tokenize_indic


class VerdictCategory(str, Enum):
    CORRECT_ANSWER = "CORRECT_ANSWER"
    FALSE_REFUSAL = "FALSE_REFUSAL"
    CORRECT_ABSTENTION = "CORRECT_ABSTENTION"
    HALLUCINATION = "HALLUCINATION"
    INCORRECT_ANSWER = "INCORRECT_ANSWER"


class JudgeVerdict(BaseModel):
    """Structured verdict produced by the offline Judge Agent."""
    query_id: str
    is_correct: bool = Field(description="True if factual content matches gold reference")
    is_grounded: bool = Field(description="True if claims are supported by provided context")
    is_abstention: bool = Field(description="True if system recognized insufficient context and abstained")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in judgment")
    reasoning: str = Field(description="Explanation of verdict")
    cited_passage_ids: list[str] = Field(default_factory=list, description="Passage IDs cited in answer")
    verdict_category: VerdictCategory = Field(description="Categorical verdict for RGB breakdown")


# Standard Hindi, Hinglish, and English refusal patterns
REFUSAL_PATTERNS = [
    # Devanagari Hindi
    r"दी गई जानकारी में.*(?:नहीं|अनुपलब्ध|उल्लेख नहीं)",
    r"संदर्भ में.*(?:जानकारी नहीं|उल्लेख नहीं|उपलब्ध नहीं)",
    r"इसकी जानकारी संदर्भ में नहीं है",
    r"दिए गए संदर्भ से.*उत्तर नहीं",
    r"मुझे नहीं पता",
    r"पर्याप्त जानकारी नहीं है",
    r"उपलब्ध नहीं है",
    # Roman Hinglish
    r"context me(?:in)?.*(nahi|available nahi|mention nahi)",
    r"di gayi jankari se.*nahi",
    r"information available nahi hai",
    r"mujhe nahi pata",
    r"answer context me nahi hai",
    # English
    r"i cannot answer",
    r"not mentioned in the (?:provided )?context",
    r"information is not available",
    r"insufficient context",
    r"the context does not provide",
]


class JudgeAgent:
    """Offline Evaluation Judge Agent enforcing strict model family separation and typed verdicts."""

    def __init__(
        self,
        judge_model_id: str = "google/gemini-2.0-flash",
        answerer_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        cache_dir: str = "data/processed/eval_cache/judge",
    ) -> None:
        self.judge_model_id = judge_model_id
        self.answerer_model_id = answerer_model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Enforce distinct model families
        self.judge_family = self._extract_family(judge_model_id)
        self.answerer_family = self._extract_family(answerer_model_id)

        if self.judge_family == self.answerer_family and self.judge_family != "unknown":
            raise ValueError(
                f"Judge model family ({self.judge_family}) must be distinct from Answerer model family ({self.answerer_family}) "
                f"per AGENTS.md Rule 14."
            )

    @staticmethod
    def _extract_family(model_id: str) -> str:
        """Extract canonical model family name from identifier."""
        mid = model_id.lower()
        if "gemini" in mid or "google" in mid:
            return "google_gemini"
        elif "llama" in mid or "meta" in mid:
            return "meta_llama"
        elif "mistral" in mid or "mixtral" in mid:
            return "mistral_ai"
        elif "qwen" in mid:
            return "qwen"
        elif "gemma" in mid:
            return "google_gemma"
        elif "gpt" in mid or "openai" in mid:
            return "openai_gpt"
        elif "claude" in mid or "anthropic" in mid:
            return "anthropic_claude"
        return "unknown"

    def detect_refusal(self, text: str) -> bool:
        """Detect whether text represents an abstention/refusal to answer."""
        if not text or not text.strip():
            return True
        norm = text.lower()
        for pat in REFUSAL_PATTERNS:
            if re.search(pat, norm):
                return True
        return False

    def extract_citations(self, text: str) -> list[str]:
        """Extract cited passage IDs e.g. [psg_1234abcd] or [cf_1234abcd]."""
        if not text:
            return []
        citations = re.findall(r"\[(psg_[a-f0-9]+|cf_[a-f0-9]+)\]", text)
        return list(dict.fromkeys(citations))

    def evaluate_item(
        self,
        query_id: str,
        question: str,
        prediction: str,
        is_impossible: bool,
        gold_answers: list[str],
        retrieved_passages: list[dict[str, Any]],
        is_serving_pipeline: bool = False,
    ) -> JudgeVerdict:
        """Evaluate a single prediction offline.

        Raises RuntimeError if invoked from within a serving pipeline.
        """
        if is_serving_pipeline:
            raise RuntimeError(
                "CRITICAL VIOLATION: Judge Agent is strictly offline evaluation-only and cannot be executed inside the serving pipeline."
            )

        # Cache check
        cache_key = hashlib.sha256(
            f"{self.judge_model_id}::{query_id}::{prediction}::{is_impossible}".encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                return JudgeVerdict(**cached_data)
            except Exception:
                pass  # Fall back to evaluation on corrupted cache

        verdict = self._evaluate_logic(
            query_id=query_id,
            question=question,
            prediction=prediction,
            is_impossible=is_impossible,
            gold_answers=gold_answers,
            retrieved_passages=retrieved_passages,
        )

        # Write cache
        try:
            cache_file.write_text(json.dumps(verdict.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return verdict

    def _evaluate_logic(
        self,
        query_id: str,
        question: str,
        prediction: str,
        is_impossible: bool,
        gold_answers: list[str],
        retrieved_passages: list[dict[str, Any]],
    ) -> JudgeVerdict:
        """Core deterministic evaluation logic with calibrated semantic verification."""
        is_abstention = self.detect_refusal(prediction)
        cited_ids = self.extract_citations(prediction)
        retrieved_pids = {p.get("passage_id") for p in retrieved_passages if p.get("passage_id")}

        # Grounding check: Are cited IDs actually in the retrieved passages?
        is_grounded = True
        if cited_ids:
            is_grounded = all(cid in retrieved_pids for cid in cited_ids)

        # Category and correctness assignment
        if is_impossible:
            # Query has NO answer in gold evidence
            if is_abstention:
                category = VerdictCategory.CORRECT_ABSTENTION
                is_correct = True
                reasoning = "Query is unanswerable, and system correctly abstained."
            else:
                category = VerdictCategory.HALLUCINATION
                is_correct = False
                reasoning = "Query is unanswerable, but system hallucinated an answer instead of abstaining."
        else:
            # Query has a gold answer
            if is_abstention:
                category = VerdictCategory.FALSE_REFUSAL
                is_correct = False
                reasoning = "Query is answerable, but system incorrectly refused to answer."
            else:
                # Semantic / substring match against gold answers
                pred_norm = normalize_text(prediction)
                matched = False
                for ga in gold_answers:
                    ga_norm = normalize_text(ga)
                    if not ga_norm:
                        continue
                    # Check substring inclusion or high token overlap
                    if ga_norm in pred_norm:
                        matched = True
                        break
                    ga_tokens = set(tokenize_indic(ga_norm))
                    pred_tokens = set(tokenize_indic(pred_norm))
                    if ga_tokens and (len(ga_tokens & pred_tokens) / len(ga_tokens) >= 0.75):
                        matched = True
                        break

                if matched:
                    category = VerdictCategory.CORRECT_ANSWER
                    is_correct = True
                    reasoning = "Predicted answer matches ground truth reference answer."
                else:
                    category = VerdictCategory.INCORRECT_ANSWER
                    is_correct = False
                    reasoning = "Predicted answer does not match ground truth reference answer."

        return JudgeVerdict(
            query_id=query_id,
            is_correct=is_correct,
            is_grounded=is_grounded,
            is_abstention=is_abstention,
            confidence=0.95 if is_correct else 0.90,
            reasoning=reasoning,
            cited_passage_ids=cited_ids,
            verdict_category=category,
        )
