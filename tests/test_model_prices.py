"""Tests for llmprover.model_prices."""

from __future__ import annotations

from llmprover.model_prices import DEFAULT_PRICE, MODEL_PRICES, ModelPrice, price_for


def test_price_for_known_model() -> None:
    assert price_for("gpt-4o-mini") == MODEL_PRICES["gpt-4o-mini"]


def test_price_for_unknown_model_uses_default() -> None:
    assert price_for("not-a-real-model") == DEFAULT_PRICE


def test_default_price_is_at_least_as_expensive_as_all_known_models() -> None:
    """DEFAULT_PRICE must be >= every known model to stay conservative."""
    for name, price in MODEL_PRICES.items():
        assert DEFAULT_PRICE.input_usd_per_million >= price.input_usd_per_million, (
            f"{name} input ${price.input_usd_per_million} exceeds DEFAULT_PRICE"
        )
        assert DEFAULT_PRICE.output_usd_per_million >= price.output_usd_per_million, (
            f"{name} output ${price.output_usd_per_million} exceeds DEFAULT_PRICE"
        )


def test_gpt56_luna_has_explicit_write_and_read_prices() -> None:
    price = MODEL_PRICES["gpt-5.6-luna"]
    assert price == ModelPrice(
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        cache_write_usd_per_million=0.25,
        cache_read_usd_per_million=0.02,
    )


def test_older_openai_has_read_but_no_write_price() -> None:
    price = MODEL_PRICES["gpt-5.4-mini"]
    assert price.cache_read_usd_per_million == 0.075
    assert price.cache_write_usd_per_million is None


def test_mistral_has_cached_read_no_write() -> None:
    price = MODEL_PRICES["mistral-large-latest"]
    assert price.cache_read_usd_per_million == 0.05
    assert price.cache_write_usd_per_million is None
