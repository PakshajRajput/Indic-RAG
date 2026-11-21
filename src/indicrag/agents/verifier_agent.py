"""Verifier Agent for PJ-09 IndicRAG-QA (Systems S2, S3, S4, S5).

Handles:
- Claim-level support verification against cited passages
- Citation fidelity validation (verifying cited passages exist in retrieved context)
- Detection of unsupported hallucinations and contradictions
"""

from __future__ import annotations

import re
from typing import Any

from indicrag.agents.state import (
    IndicRAGState,
    ReasonCode,
    SystemLadderConfig,
    VerificationVerdict,
)
from indicrag.eval.metrics import tokenize_indic


def extract_claims(text: str) -> list[str]:
    """Break answer text into atomic sentences/clauses for claim verification."""
    # Split on sentence delimiters (danda ।, question mark ?, period .)
    raw_sentences = re.split(r"[।\.\?\n]+", text)
    cleaned = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return cleaned if cleaned else [text.strip()]


class VerifierAgent:
    """Agent verifying factual grounding and citation fidelity of candidate answers."""

    def __init__(self, support_threshold: float = 0.75, llm_client: Any | None = None) -> None:
        self.support_threshold = support_threshold
        self.llm_client = llm_client

    def verify(
        self,
        draft_answer: str,
        cited_ids: list[str],
        available_passages: list[dict[str, Any]],
        is_refusal: bool,
    ) -> tuple[VerificationVerdict, list[str], dict[str, Any]]:
        """Verify draft answer claims against available evidence."""
        if is_refusal:
            return VerificationVerdict.ANSWERER_REFUSED, [], {"reason": "Answerer explicitly refused."}

        if not available_passages:
            return VerificationVerdict.NO_EVIDENCE, [draft_answer], {"reason": "No evidence passages available."}

        passage_lookup = {p.get("passage_id"): p.get("text", "") for p in available_passages if p.get("passage_id")}
        all_passage_text = " ".join([p.get("text", "") for p in available_passages]).lower()

        details: dict[str, Any] = {
            "cited_ids": cited_ids,
            "citations_valid": True,
            "claim_verifications": [],
        }

        # Citation validation: do cited passages exist in available evidence?
        if cited_ids:
            missing_citations = [cid for cid in cited_ids if cid not in passage_lookup]
            if missing_citations:
                details["citations_valid"] = False
                details["missing_citations"] = missing_citations
                return (
                    VerificationVerdict.UNSUPPORTED,
                    [f"Cited non-existent passage: {missing_citations}"],
                    details,
                )

        # Context to verify against: text of cited passages, or all available passages if no explicit citations
        if cited_ids:
            evidence_text = " ".join([passage_lookup.get(cid, "") for cid in cited_ids]).lower()
        else:
            evidence_text = all_passage_text

        claims = extract_claims(draft_answer)
        # Normalize Hindi numerals in evidence for cross-lingual numerical grounding
        indic_digits = "०१२३४५६७८९"
        for d_idx, idig in enumerate(indic_digits):
            evidence_text = evidence_text.replace(idig, str(d_idx))

        claims = extract_claims(draft_answer)
        unsupported: list[str] = []

        stopwords = {"यह", "वह", "है", "था", "थी", "थे", "हैं", "का", "की", "के", "में", "से", "को", "पर", "और", "द्वारा", "psg", "is", "the", "in", "of", "to", "for", "me", "se", "ke"}

        from indicrag.agents.query_agent import rule_based_hinglish_to_hindi

        for claim in claims:
            # Strip citation tags like [psg_xxx] from claim text
            clean_claim = re.sub(r"\[[a-zA-Z0-9_\-\.]+\]", "", claim).strip()
            # For Roman Hinglish claims, expand with transliterated tokens to bridge cross-script gap
            if re.search(r"[a-zA-Z]", clean_claim):
                xlit_claim = rule_based_hinglish_to_hindi(clean_claim)
                combined_claim_text = clean_claim + " " + xlit_claim
            else:
                combined_claim_text = clean_claim
            for d_idx, idig in enumerate(indic_digits):
                combined_claim_text = combined_claim_text.replace(idig, str(d_idx))

            claim_words = [w.lower() for w in tokenize_indic(combined_claim_text) if w.lower() not in stopwords]

            if not claim_words:
                continue

            # Check word overlap with evidence
            overlap_count = sum(1 for w in claim_words if w in evidence_text)
            overlap_ratio = overlap_count / len(claim_words)

            is_supp = overlap_ratio >= self.support_threshold
            details["claim_verifications"].append({
                "claim": clean_claim,
                "overlap_ratio": overlap_ratio,
                "is_supported": is_supp,
            })

            if not is_supp:
                unsupported.append(clean_claim)

        # Cross-script LLM validation fallback ONLY if lexical overlap fails due to Latin vs Devanagari script mismatch
        has_latin = bool(re.search(r"[a-zA-Z]", draft_answer))
        if unsupported and has_latin:
            import os
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and cited_ids:
                try:
                    import urllib.request
                    import json
                    still_unsupported = []
                    for u_claim in unsupported:
                        payload = json.dumps({
                            "model": "qwen/qwen3.8-27b",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a factual claim verifier for Indic RAG. Determine if the claim is supported by the provided passage. Reply with ONLY one word: SUPPORTED or UNSUPPORTED."
                                },
                                {
                                    "role": "user",
                                    "content": f"Claim: {u_claim}\nPassage: {evidence_text[:1500]}"
                                }
                            ],
                            "temperature": 0.0,
                            "max_tokens": 16,
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            "https://api.groq.com/openai/v1/chat/completions",
                            data=payload,
                            headers={"Authorization": f"Bearer {groq_key}", "User-Agent": "IndicRAG/1.0", "Content-Type": "application/json"}
                        )
                        with urllib.request.urlopen(req, timeout=6) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            verdict_word = data["choices"][0]["message"].get("content", "").strip().upper()
                            if "SUPPORTED" in verdict_word and "UNSUPPORTED" not in verdict_word:
                                # Supported by LLM cross-script check
                                pass
                            else:
                                still_unsupported.append(u_claim)
                    unsupported = still_unsupported
                except Exception:
                    pass

        if not unsupported:
            return VerificationVerdict.SUPPORTED, [], details
        return VerificationVerdict.UNSUPPORTED, unsupported, details

    def run(self, state: IndicRAGState) -> IndicRAGState:
        """Execute verification on state."""
        draft = state.draft_answer or ""
        passages = state.graded_passages if state.graded_passages else state.retrieved_passages

        verdict, unsupported, details = self.verify(
            draft_answer=draft,
            cited_ids=state.cited_passage_ids,
            available_passages=passages,
            is_refusal=state.is_abstention,
        )

        state.verification_verdict = verdict
        state.unsupported_claims = unsupported
        state.verification_details = details

        # In S2, S3, S4 (where Abstention Arbiter is absent), Verifier finalizes the answer
        if state.system_ladder in (SystemLadderConfig.S2, SystemLadderConfig.S3, SystemLadderConfig.S4):
            if verdict == VerificationVerdict.SUPPORTED:
                state.final_answer = state.draft_answer
                state.is_abstention = False
            elif verdict == VerificationVerdict.ANSWERER_REFUSED:
                state.final_answer = state.draft_answer
                state.is_abstention = True
                state.reason_code = ReasonCode.ANSWERER_REFUSED
            elif verdict == VerificationVerdict.NO_EVIDENCE:
                state.is_abstention = True
                state.reason_code = ReasonCode.NO_EVIDENCE
                state.final_answer = "सत्यापन विफल: कोई साक्ष्य उपलब्ध नहीं है।"
            else:
                state.is_abstention = True
                state.reason_code = ReasonCode.UNSUPPORTED_ANSWER
                state.final_answer = "सत्यापन में दावे असमर्थित पाए गए, इसलिए उत्तर अस्वीकार किया जाता है।"

        state.agent_history.append(f"verifier_agent:{verdict.value.lower()}")
        return state


def verifier_node(state: IndicRAGState, support_threshold: float = 0.50) -> IndicRAGState:
    """LangGraph node wrapper for VerifierAgent."""
    agent = VerifierAgent(support_threshold=support_threshold)
    return agent.run(state)
