"""Tests for model defaults, provider inference, and sampling compatibility."""

import aikit

from vidflow.models_config import (
    DEFAULT_MODEL,
    FIXED_SAMPLING_MODELS,
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
    for model in (MODEL_OPUS, MODEL_SONNET, "claude-fable-5",
                  "claude-opus-4-8", "claude-opus-4-7"):
        assert model in FIXED_SAMPLING_MODELS
        assert not model_accepts_temperature(model)


def test_other_models_accept_temperature():
    assert model_accepts_temperature(MODEL_HAIKU)
    assert model_accepts_temperature("claude-opus-4-6")
    # Local slots always accept temperature
    assert model_accepts_temperature(DEFAULT_MODEL)
    assert model_accepts_temperature(LOCAL_QUICK)


def test_unknown_model_defaults_to_accepting_temperature():
    # Conservative default: unknown models assumed to accept temperature.
    # If a future fixed-sampling model ships, it must be added explicitly.
    assert model_accepts_temperature("claude-some-future-model")
