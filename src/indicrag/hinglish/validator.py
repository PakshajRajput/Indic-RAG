"""Hinglish Validator module for PJ-09 IndicRAG-QA.

Enforces script compliance, length bounds, entity preservation, duplicate detection,
and error-detection heuristics. Flags uncertain records for human review.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of validating a candidate Hinglish generated question."""

    is_valid: bool = Field(description="True if strict criteria are met")
    flagged_for_review: bool = Field(default=False, description="True if manual inspection is needed")
    review_reasons: list[str] = Field(default_factory=list, description="List of reasons for flagging or invalidation")
    devanagari_count: int = Field(default=0, description="Count of leaked Devanagari characters")
    length_ratio: float = Field(default=1.0, description="Ratio of Hinglish length to Hindi length")
    preserved_digits: bool = Field(default=True, description="Whether numerical digits were preserved")


class HinglishValidator:
    """Validates synthetic Roman-script Hinglish questions generated from Hindi sources."""

    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
    DIGIT_REGEX = re.compile(r"\b\d+\b")

    def __init__(
        self,
        min_length_ratio: float = 0.4,
        max_length_ratio: float = 2.5,
        seen_cache: set[str] | None = None,
    ) -> None:
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.seen_cache: set[str] = seen_cache if seen_cache is not None else set()

    def validate(self, hindi_source: str, hinglish_candidate: str) -> ValidationResult:
        """Run all validation rules on the generated Hinglish question."""
        reasons: list[str] = []
        flagged = False
        is_valid = True

        clean_candidate = hinglish_candidate.strip()
        clean_source = hindi_source.strip()

        # 1. Non-empty check
        if not clean_candidate:
            return ValidationResult(
                is_valid=False,
                flagged_for_review=True,
                review_reasons=["EMPTY_OUTPUT"],
                devanagari_count=0,
                length_ratio=0.0,
                preserved_digits=False,
            )

        # 2. Script compliance: ZERO Devanagari characters allowed in Hinglish field
        devanagari_chars = self.DEVANAGARI_REGEX.findall(clean_candidate)
        devanagari_count = len(devanagari_chars)
        if devanagari_count > 0:
            is_valid = False
            flagged = True
            reasons.append(f"DEVANAGARI_LEAKAGE: {devanagari_count} characters found ({''.join(devanagari_chars[:5])})")

        # 3. Length ratio validation
        len_source = max(1, len(clean_source))
        len_cand = len(clean_candidate)
        ratio = round(len_cand / len_source, 3)

        if ratio < self.min_length_ratio:
            flagged = True
            reasons.append(f"LENGTH_TOO_SHORT: ratio {ratio} < {self.min_length_ratio}")
        elif ratio > self.max_length_ratio:
            flagged = True
            reasons.append(f"LENGTH_TOO_LONG: ratio {ratio} > {self.max_length_ratio}")

        # 4. Numerical entity preservation (e.g. years, measurements)
        source_digits = set(self.DIGIT_REGEX.findall(clean_source))
        cand_digits = set(self.DIGIT_REGEX.findall(clean_candidate))
        digits_preserved = source_digits.issubset(cand_digits)
        if not digits_preserved:
            missing = source_digits - cand_digits
            flagged = True
            reasons.append(f"MISSING_DIGITS: Source had {source_digits}, candidate missing {missing}")

        # 5. Duplicate detection
        cand_lower = clean_candidate.lower()
        if cand_lower in self.seen_cache:
            flagged = True
            reasons.append("DUPLICATE_OUTPUT: Identical string previously generated")
        else:
            self.seen_cache.add(cand_lower)

        # 6. Question mark consistency
        if clean_source.endswith("?") and not clean_candidate.endswith("?"):
            flagged = True
            reasons.append("MISSING_QUESTION_MARK")

        # Final validity determination
        if devanagari_count > 0:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            flagged_for_review=flagged,
            review_reasons=reasons,
            devanagari_count=devanagari_count,
            length_ratio=ratio,
            preserved_digits=digits_preserved,
        )
