"""Sampling parameters never reach the Anthropic SDK as keyword arguments.

anthropic>=1.0 removed ``temperature``/``top_p``/``top_k`` from the
``messages.create()`` / ``messages.stream()`` signatures (a ``TypeError``
if passed), and Opus 4.7+ / the 5-family reject them server-side (HTTP
400). Models that still honour temperature get it via ``extra_body``.
"""

from unittest.mock import MagicMock

import pytest

from vidflow.transcribe.processor import VidscribeProcessor


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


def _processor(model: str, temperature: float = 0.2) -> VidscribeProcessor:
    processor = VidscribeProcessor(
        api_key="fake-key",
        model=model,
        temperature=temperature,
        json_output=True,
        text_only=True,
    )
    processor.client = MagicMock()
    stream = processor.client.messages.stream.return_value.__enter__.return_value
    stream.text_stream = iter(["hello"])
    stream.get_final_message.return_value = MagicMock(stop_reason="end_turn")
    return processor


def _stream_kwargs(processor: VidscribeProcessor) -> dict:
    processor._make_streaming_api_request(
        messages=[{"role": "user", "content": "hi"}],
        progress_task=None,
        progress=MagicMock(),
    )
    return processor.client.messages.stream.call_args.kwargs


@pytest.mark.parametrize(
    "model", ["claude-fable-5-1", "claude-opus-5", "claude-opus-4-7", "claude-unknown-9"]
)
def test_fixed_sampling_models_omit_temperature(no_warm, model):
    kwargs = _stream_kwargs(_processor(model))
    assert "temperature" not in kwargs
    assert "extra_body" not in kwargs


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-6"])
def test_sampling_models_pass_temperature_via_extra_body(no_warm, model):
    kwargs = _stream_kwargs(_processor(model, temperature=0.4))
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 0.4}
