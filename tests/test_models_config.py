"""Tests for model defaults, provider inference, and sampling compatibility."""

import aikit

from vidflow.models_config import (
    DEFAULT_MODEL,
    LOCAL_QUICK,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    model_accepts_temperature,
)


def test_default_is_local_primary():
    # Local inference is the default lane; claude-* is the escape hatch.
    assert DEFAULT_MODEL == "primary"
    assert LOCAL_QUICK == "quick"


def test_provider_inference():
    assert aikit.provider_for(DEFAULT_MODEL) == "local"
    assert aikit.provider_for(MODEL_OPUS) == "anthropic"


def test_escape_hatch_models_are_current():
    assert MODEL_OPUS == "claude-opus-5"
    assert MODEL_SONNET == "claude-sonnet-5"
    assert MODEL_HAIKU == "claude-haiku-4-5"


def test_five_family_rejects_temperature():
    for model in (
        MODEL_OPUS,
        MODEL_SONNET,
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-opus-4-8",
        "claude-opus-4-7",
    ):
        assert not model_accepts_temperature(model), model


def test_pre_47_models_accept_temperature():
    for model in (
        MODEL_HAIKU,
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-latest",
    ):
        assert model_accepts_temperature(model), model


def test_local_slots_accept_temperature():
    assert model_accepts_temperature(DEFAULT_MODEL)
    assert model_accepts_temperature(LOCAL_QUICK)


def test_unknown_anthropic_model_omits_temperature():
    # Safe default: omitting sampling params is always valid; sending them to
    # a newer model is not. A claude-* id we cannot parse gets no temperature.
    assert not model_accepts_temperature("claude-some-future-model")
    assert not model_accepts_temperature("claude-opus-6")
