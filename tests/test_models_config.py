"""Tests for model ID registry and sampling-param compatibility."""

from vidflow.models_config import (
    DEFAULT_MODEL,
    FIXED_SAMPLING_MODELS,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_OPUS_47,
    MODEL_SONNET,
    TRANSCRIBE_MODELS,
    model_accepts_temperature,
)


def test_default_is_opus_4_6():
    # Intentionally pinned: Opus 4.7 rejects non-default temperature,
    # and transcription relies on temperature control.
    assert DEFAULT_MODEL == MODEL_OPUS
    assert MODEL_OPUS == "claude-opus-4-6"


def test_opus_4_7_available_as_choice():
    assert MODEL_OPUS_47 in TRANSCRIBE_MODELS
    assert MODEL_OPUS_47 == "claude-opus-4-7"


def test_all_current_models_present():
    assert set(TRANSCRIBE_MODELS) == {
        MODEL_OPUS, MODEL_OPUS_47, MODEL_SONNET, MODEL_HAIKU,
    }


def test_opus_4_7_rejects_temperature():
    assert MODEL_OPUS_47 in FIXED_SAMPLING_MODELS
    assert not model_accepts_temperature(MODEL_OPUS_47)


def test_other_models_accept_temperature():
    assert model_accepts_temperature(MODEL_OPUS)
    assert model_accepts_temperature(MODEL_SONNET)
    assert model_accepts_temperature(MODEL_HAIKU)


def test_unknown_model_defaults_to_accepting_temperature():
    # Conservative default: unknown models assumed to accept temperature.
    # If a future fixed-sampling model ships, it must be added explicitly.
    assert model_accepts_temperature("claude-some-future-model")
