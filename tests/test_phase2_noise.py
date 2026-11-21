"""Unit tests for Phase 2: Controlled Noise Injection, RGB Protocol, and Counterfactual Distractors."""

import hashlib
import json
from pathlib import Path

import pytest
from src.indicrag.data.noise import (
    CounterfactualGenerator,
    build_phase2_artifacts,
)


@pytest.fixture(scope="module")
def phase2_data():
    """Ensure Phase 2 artifacts exist."""
    data_dir = Path("data/processed")
    noise_log_path = data_dir / "noise_eval" / "provenance_log.json"
    if not noise_log_path.exists():
        build_phase2_artifacts(data_dir=str(data_dir), seed=42)
    return data_dir


def test_baseline_manifest_preservation(phase2_data):
    """Verify that Phase 1 clean baseline files remain 100% cryptographically unchanged."""
    data_dir = phase2_data
    manifest_path = data_dir / "baseline_manifest.json"
    assert manifest_path.exists(), "baseline_manifest.json missing"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, info in manifest["files"].items():
        fp = Path(rel_path)
        assert fp.exists(), f"Baseline file missing: {rel_path}"

        hasher = hashlib.sha256()
        with open(fp, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_hash = hasher.hexdigest()
        assert (
            current_hash == info["sha256"]
        ), f"Baseline artifact {rel_path} was modified! Expected {info['sha256']}, got {current_hash}"


def test_deterministic_reproduction(tmp_path):
    """Verify that same seed generates byte-identical RGB sets and counterfactuals."""
    gen1 = CounterfactualGenerator(seed=42)
    gen2 = CounterfactualGenerator(seed=42)

    text = "पानीपत का प्रथम युद्ध 1526 में लड़ा गया था, जिसमें 50000 सैनिक थे।"
    out1, mods1 = gen1.perturb(text, "पानीपत का युद्ध")
    out2, mods2 = gen2.perturb(text, "पानीपत का युद्ध")

    assert out1 == out2, "Counterfactual generation is non-deterministic!"
    assert mods1 == mods2


def test_rgb_noise_bounds_and_structure(phase2_data):
    """Verify RGB candidate sets obey k=5, noise rates, and gold evidence retention."""
    data_dir = phase2_data
    with open(data_dir / "noise_eval" / "rgb_dev_eval_sets.json", encoding="utf-8") as f:
        dev_sets = json.load(f)

    rates_seen = set()
    for rate_group in dev_sets:
        r = rate_group["noise_rate"]
        rates_seen.add(r)
        items = rate_group["items"]
        assert len(items) == 150

        for item in items:
            assert item["target_k"] == 5
            assert len(item["passages"]) == 5

            if item["is_impossible"]:
                # Unanswerable query must NEVER have gold evidence in candidates
                assert item["has_gold_in_candidates"] is False, "Unanswerable query unexpectedly has gold passage!"
            else:
                # Answerable query
                if r < 1.0:
                    assert item["has_gold_in_candidates"] is True, f"Answerable query lost gold passage under noise {r}!"
                else:
                    # Noise rate 1.0 (pure distractor test for negative rejection)
                    assert item["has_gold_in_candidates"] is False, "Noise rate 1.0 unexpectedly retains gold passage!"

    assert rates_seen == {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def test_zero_annotation_leakage_in_perturbed_corpora(phase2_data):
    """Verify perturbed searchable corpora contain zero QA labels or questions."""
    data_dir = phase2_data
    forbidden_keys = {"question", "answers", "is_impossible", "synthetic_hinglish_id"}

    for cfile in ["corpus_dev_perturbed.json", "corpus_test_perturbed.json"]:
        path = data_dir / "counterfactual" / cfile
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            corpus = json.load(f)

        for p in corpus:
            keys = set(p.keys())
            leak = keys & forbidden_keys
            assert len(leak) == 0, f"Annotation leakage in {cfile} passage {p.get('passage_id')}: {leak}"


def test_counterfactual_distractor_provenance(phase2_data):
    """Verify counterfactual distractor passages store complete transformation provenance."""
    data_dir = phase2_data
    path = data_dir / "counterfactual" / "corpus_dev_perturbed.json"
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)

    cf_passages = [p for p in corpus if p.get("is_counterfactual")]
    assert len(cf_passages) > 0, "No counterfactual passages found in corpus_dev_perturbed.json"

    for p in cf_passages:
        assert p["passage_id"].startswith("cf_")
        assert "source_passage_id" in p
        assert p["source_passage_id"].startswith("psg_")
        assert "provenance" in p
        assert p["provenance"]["seed"] == 42
        assert len(p["provenance"]["modifications"]) > 0


def test_distractor_does_not_inject_gold_answer():
    """Verify that CounterfactualGenerator respects forbidden answers list."""
    gen = CounterfactualGenerator(seed=42)
    text = "दिल्ली की स्थापना 1911 में हुई थी।"
    # Forbid year 1923
    out_text, mods = gen.perturb(text, "दिल्ली", forbidden_answers=["1923"])
    assert "1923" not in out_text, "Forbidden answer was introduced by perturber!"
