"""USD list prices per 1 million tokens for models used by LLMprover.

Unknown model names fall back to ``DEFAULT_PRICE``, which is set to the highest
list price among known models so that cost estimates for unrecognised models are
conservative (over-estimate rather than under-estimate).

Prices are standard API rates (no batch or Flex discount applied).

Cache pricing columns (``None`` = category not used for that model):
- ``cache_write_usd_per_million``: tokens written into the prompt cache.
- ``cache_read_usd_per_million``: tokens read from the prompt cache.

Sources (verified August 2026):
  OpenAI   — https://developers.openai.com/api/docs/pricing
  Mistral  — https://docs.mistral.ai/studio-api/conversations/advanced/prompt-caching
  Anthropic — https://platform.claude.com/docs/en/about-claude/pricing
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD charged per 1 million tokens for each token category."""

    input_usd_per_million: float
    output_usd_per_million: float
    cache_write_usd_per_million: float | None = None
    cache_read_usd_per_million: float | None = None


# API model name → list price (USD / 1M tokens).
# Reasoning tokens are billed as output by providers that report them, and are
# already included in ``TokenUsage.output_tokens``.
MODEL_PRICES: dict[str, ModelPrice] = {
    # ── OpenAI (pre GPT-5.6: automatic caching; no separate write fee) ─────────
    # gpt-4o-mini: cache read = 50 % of input (legacy tier).
    "gpt-4o-mini": ModelPrice(
        input_usd_per_million=0.15,
        output_usd_per_million=0.60,
        cache_read_usd_per_million=0.075,
    ),
    # gpt-5-nano: cache read = 10 % of input.
    "gpt-5-nano": ModelPrice(
        input_usd_per_million=0.05,
        output_usd_per_million=0.40,
        cache_read_usd_per_million=0.005,
    ),
    # gpt-5.4-mini: automatic caching only (no explicit breakpoints / write fee).
    "gpt-5.4-mini": ModelPrice(
        input_usd_per_million=0.75,
        output_usd_per_million=4.50,
        cache_read_usd_per_million=0.075,
    ),
    # ── OpenAI GPT-5.6+: explicit caching with write + read rates ──────────────
    # Cache write = 1.25× input; cache read = 0.10× input (official list).
    "gpt-5.6-luna": ModelPrice(
        input_usd_per_million=0.20,
        output_usd_per_million=1.20,
        cache_write_usd_per_million=0.25,
        cache_read_usd_per_million=0.02,
    ),
    # ── Mistral (automatic prefix cache; cached input @ 10 %; no write fee) ───
    "mistral-small-latest": ModelPrice(
        input_usd_per_million=0.15,
        output_usd_per_million=0.60,
        cache_read_usd_per_million=0.015,
    ),
    "mistral-large-latest": ModelPrice(
        input_usd_per_million=0.50,
        output_usd_per_million=1.50,
        cache_read_usd_per_million=0.05,
    ),
    # ── Anthropic (5-minute ephemeral TTL: write 1.25×, read 0.10×) ───────────
    "claude-haiku-4-5": ModelPrice(
        input_usd_per_million=1.00,
        output_usd_per_million=5.00,
        cache_write_usd_per_million=1.25,
        cache_read_usd_per_million=0.10,
    ),
}

# Conservative fallback: use the highest input+output price among known models
# so that cost estimates for unrecognised models over-estimate rather than
# under-estimate.  Currently claude-haiku-4-5 ($1/$5) is the most expensive.
DEFAULT_PRICE = ModelPrice(
    input_usd_per_million=1.00,
    output_usd_per_million=5.00,
)


def price_for(model: str) -> ModelPrice:
    """Return the list price for *model*, or ``DEFAULT_PRICE`` if unknown."""
    return MODEL_PRICES.get(model, DEFAULT_PRICE)
