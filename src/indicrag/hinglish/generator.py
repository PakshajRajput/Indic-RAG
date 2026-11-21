"""Synthetic Hinglish Generator with persistent disk caching and quality auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .validator import HinglishValidator, ValidationResult


class GeneratedHinglishItem(BaseModel):
    """Complete record of a synthetic Hinglish conversion, preserving provenance."""

    id: str
    original_hindi: str
    synthetic_hinglish: str
    model_id: str
    prompt_version: str
    generation_params: dict[str, Any]
    validation: ValidationResult
    is_cached: bool = False
    flagged_for_human_review: bool = False


class DiskCache:
    """Persistent on-disk cache keyed by hash of model, prompt, parameters, and input text."""

    def __init__(self, cache_dir: str = ".cache/hinglish_gen") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self,
        model_id: str,
        prompt_version: str,
        params: dict[str, Any],
        hindi_text: str,
    ) -> str:
        param_str = json.dumps(params, sort_keys=True)
        raw_key = f"{model_id}::{prompt_version}::{param_str}::{hindi_text.strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        model_id: str,
        prompt_version: str,
        params: dict[str, Any],
        hindi_text: str,
    ) -> dict[str, Any] | None:
        key = self._compute_key(model_id, prompt_version, params, hindi_text)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception:
                return None
        return None

    def set(
        self,
        model_id: str,
        prompt_version: str,
        params: dict[str, Any],
        hindi_text: str,
        result_data: dict[str, Any],
    ) -> None:
        key = self._compute_key(model_id, prompt_version, params, hindi_text)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)


class HinglishGenerator:
    """Manages few-shot API generation of synthetic Hinglish questions with disk caching."""

    def __init__(
        self,
        prompt_config_path: str = "configs/prompts/hinglish_generation_v1.yaml",
        cache_dir: str = ".cache/hinglish_gen",
        validator: HinglishValidator | None = None,
    ) -> None:
        self.prompt_config_path = prompt_config_path
        self.cache = DiskCache(cache_dir=cache_dir)
        self.validator = validator or HinglishValidator()

        # Load version-controlled prompt
        with open(prompt_config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.prompt_version = self.config.get("version", "v1.0")
        self.model_id = self.config.get("default_model", "gemini-2.0-flash")
        self.params = self.config.get("parameters", {})

    def generate(
        self,
        item_id: str,
        hindi_question: str,
        api_client_override: Any | None = None,
    ) -> GeneratedHinglishItem:
        """Convert a Hindi question to Roman-script Hinglish using disk cache or API."""
        # 1. Check disk cache first to enforce Rule 10 (cost control)
        cached = self.cache.get(
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            params=self.params,
            hindi_text=hindi_question,
        )

        if cached is not None:
            val_res = self.validator.validate(hindi_question, cached["hinglish"])
            return GeneratedHinglishItem(
                id=item_id,
                original_hindi=hindi_question,
                synthetic_hinglish=cached["hinglish"],
                model_id=self.model_id,
                prompt_version=self.prompt_version,
                generation_params=self.params,
                validation=val_res,
                is_cached=True,
                flagged_for_human_review=val_res.flagged_for_review,
            )

        # 2. If not cached, call client or stub
        if api_client_override is not None:
            raw_hinglish = api_client_override(hindi_question)
        else:
            raise RuntimeError(
                "API client is not configured. Real API calls require explicit credentials in .env "
                "and verification of available quota."
            )

        # 3. Validate output
        val_res = self.validator.validate(hindi_question, raw_hinglish)

        # 4. Save to disk cache
        self.cache.set(
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            params=self.params,
            hindi_text=hindi_question,
            result_data={
                "hinglish": raw_hinglish,
                "validation": val_res.model_dump(),
            },
        )

        return GeneratedHinglishItem(
            id=item_id,
            original_hindi=hindi_question,
            synthetic_hinglish=raw_hinglish,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            generation_params=self.params,
            validation=val_res,
            is_cached=False,
            flagged_for_human_review=val_res.flagged_for_review,
        )
