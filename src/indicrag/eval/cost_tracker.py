"""Resource, latency, and API cost tracking module.

Provides accounting for:
- Prompt and completion token counts
- Measured invocation latency (mean, median, p95)
- Cache hit vs. live API call distinction
- Cost estimation across model families (Gemini, Llama, Mistral)
"""

from __future__ import annotations

import time

import numpy as np
from pydantic import BaseModel, Field

# Pricing per 1M tokens (USD) as of standard hosted tiers
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Google AI Studio
    "gemini-1.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
    "gemini-2.0-flash": {"input_per_m": 0.100, "output_per_m": 0.40},
    "gemini-1.5-pro": {"input_per_m": 1.250, "output_per_m": 5.00},
    # Groq / Together Llama models
    "llama-3.1-8b-instruct": {"input_per_m": 0.05, "output_per_m": 0.08},
    "llama-3.3-70b-versatile": {"input_per_m": 0.59, "output_per_m": 0.79},
    "mistral-7b-instruct": {"input_per_m": 0.06, "output_per_m": 0.06},
    # Default fallback
    "default": {"input_per_m": 0.10, "output_per_m": 0.20},
}


class CallRecord(BaseModel):
    """Metadata recorded for an individual LLM invocation."""
    call_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    is_cache_hit: bool
    estimated_cost_usd: float
    timestamp: float = Field(default_factory=time.time)


class CostTracker:
    """Tracks token usage, latencies, and computes dollar cost estimates."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def calculate_call_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int, is_cache_hit: bool) -> float:
        """Calculate dollar cost of an invocation. Cached calls incur $0 cost."""
        if is_cache_hit:
            return 0.0

        rates = MODEL_PRICING.get(model_name.lower())
        if not rates:
            for k in MODEL_PRICING:
                if k in model_name.lower():
                    rates = MODEL_PRICING[k]
                    break
        if not rates:
            rates = MODEL_PRICING["default"]

        cost_in = (prompt_tokens / 1_000_000.0) * rates["input_per_m"]
        cost_out = (completion_tokens / 1_000_000.0) * rates["output_per_m"]
        return float(cost_in + cost_out)

    def record_call(
        self,
        call_id: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        is_cache_hit: bool = False,
    ) -> CallRecord:
        """Record an LLM call and calculate cost."""
        cost = self.calculate_call_cost(model_name, prompt_tokens, completion_tokens, is_cache_hit)
        rec = CallRecord(
            call_id=call_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            is_cache_hit=is_cache_hit,
            estimated_cost_usd=cost,
        )
        self.records.append(rec)
        return rec

    def summary(self) -> dict[str, float]:
        """Aggregate summary statistics over all recorded calls."""
        if not self.records:
            return {
                "total_calls": 0,
                "cache_hits": 0,
                "live_calls": 0,
                "cache_hit_rate": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
                "median_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            }

        total_calls = len(self.records)
        cache_hits = sum(1 for r in self.records if r.is_cache_hit)
        live_calls = total_calls - cache_hits

        p_toks = sum(r.prompt_tokens for r in self.records)
        c_toks = sum(r.completion_tokens for r in self.records)
        total_cost = sum(r.estimated_cost_usd for r in self.records)

        # Compute live latency vs cached latency
        live_lats = [r.latency_ms for r in self.records if not r.is_cache_hit]
        all_lats = [r.latency_ms for r in self.records]

        target_lats = live_lats if live_lats else all_lats

        return {
            "total_calls": float(total_calls),
            "cache_hits": float(cache_hits),
            "live_calls": float(live_calls),
            "cache_hit_rate": float(cache_hits / total_calls),
            "total_prompt_tokens": float(p_toks),
            "total_completion_tokens": float(c_toks),
            "total_tokens": float(p_toks + c_toks),
            "total_cost_usd": float(total_cost),
            "mean_latency_ms": float(np.mean(target_lats)),
            "median_latency_ms": float(np.median(target_lats)),
            "p95_latency_ms": float(np.percentile(target_lats, 95)),
        }

    get_summary = summary
