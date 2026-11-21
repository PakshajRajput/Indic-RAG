"""Unit tests for Hinglish generator, validator, and disk caching."""


from src.indicrag.hinglish.generator import DiskCache, HinglishGenerator
from src.indicrag.hinglish.validator import HinglishValidator


def test_validator_clean_roman():
    validator = HinglishValidator()
    res = validator.validate(
        hindi_source="भारत की राजधानी क्या है?",
        hinglish_candidate="Bharat ki capital kya hai?",
    )
    assert res.is_valid is True
    assert res.flagged_for_review is False
    assert res.devanagari_count == 0
    assert len(res.review_reasons) == 0


def test_validator_devanagari_leakage():
    validator = HinglishValidator()
    res = validator.validate(
        hindi_source="राइन नदी कहाँ बहती है?",
        hinglish_candidate="Rhine नदी kahan behti hai?",
    )
    assert res.is_valid is False
    assert res.flagged_for_review is True
    assert res.devanagari_count > 0
    assert any("DEVANAGARI_LEAKAGE" in r for r in res.review_reasons)


def test_validator_missing_digits():
    validator = HinglishValidator()
    # Source has 1947, candidate omitted it
    res = validator.validate(
        hindi_source="भारत 1947 में आज़ाद हुआ था?",
        hinglish_candidate="Bharat aazad hua tha?",
    )
    assert res.flagged_for_review is True
    assert res.preserved_digits is False
    assert any("MISSING_DIGITS" in r for r in res.review_reasons)


def test_validator_duplicate_detection():
    validator = HinglishValidator()
    res1 = validator.validate("सवाल एक?", "Yeh ek question hai?")
    assert res1.flagged_for_review is False

    res2 = validator.validate("सवाल दो?", "Yeh ek question hai?")
    assert res2.flagged_for_review is True
    assert any("DUPLICATE_OUTPUT" in r for r in res2.review_reasons)


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path / "cache"))
    params = {"temperature": 0.1}

    # Miss
    hit = cache.get("model-x", "v1.0", params, "सवाल?")
    assert hit is None

    # Set
    cache.set("model-x", "v1.0", params, "सवाल?", {"hinglish": "Sawaal?"})

    # Hit
    hit2 = cache.get("model-x", "v1.0", params, "सवाल?")
    assert hit2 is not None
    assert hit2["hinglish"] == "Sawaal?"


def test_generator_with_stub_client(tmp_path):
    cache_dir = tmp_path / "test_cache"
    generator = HinglishGenerator(
        prompt_config_path="configs/prompts/hinglish_generation_v1.yaml",
        cache_dir=str(cache_dir),
    )

    call_count = 0

    def stub_client(hindi_text: str) -> str:
        nonlocal call_count
        call_count += 1
        return "Bingen aur Bonn ke beech kya behti hai?"

    # First call -> calls client and caches
    item1 = generator.generate(
        item_id="item-001",
        hindi_question="बिंगन और बॉन के बीच क्या बहती है?",
        api_client_override=stub_client,
    )
    assert call_count == 1
    assert item1.is_cached is False
    assert item1.original_hindi == "बिंगन और बॉन के बीच क्या बहती है?"
    assert item1.synthetic_hinglish == "Bingen aur Bonn ke beech kya behti hai?"
    assert item1.validation.is_valid is True

    # Second call -> serves from cache (0 client calls)
    item2 = generator.generate(
        item_id="item-001",
        hindi_question="बिंगन और बॉन के बीच क्या बहती है?",
        api_client_override=stub_client,
    )
    assert call_count == 1  # No new call!
    assert item2.is_cached is True
    assert item2.synthetic_hinglish == "Bingen aur Bonn ke beech kya behti hai?"
