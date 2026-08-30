"""Tests for tiered frontmatter model selection in VidscribeProcessor."""

from unittest.mock import MagicMock

import pytest

from vidflow.transcribe.processor import VidscribeProcessor

VALID_YAML = (
    "title: Test Talk on Neural Data\n"
    "created: 2026-08-29\n"
    "tags:\n  - test\n"
    "description: A test transcript.\n"
)


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr(
        "vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None
    )


def _local_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    return response


def _anthropic_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content[0].text = text
    return response


class TestLocalLane:
    """Local lane: frontmatter runs on the resident session model."""

    def test_uses_session_model(self, no_warm):
        processor = VidscribeProcessor(
            api_key=None, model="primary", json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.local_client.chat.completions.create.return_value = (
            _local_response(VALID_YAML)
        )

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.local_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "primary"
        assert frontmatter["title"] == "Test Talk on Neural Data"

    def test_failure_falls_back_to_static(self, no_warm):
        processor = VidscribeProcessor(
            api_key=None, model="primary", json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )

        frontmatter = processor.generate_frontmatter("some transcript")

        assert frontmatter["title"] == "Workshop Transcript"


class TestAnthropicLane:
    """Anthropic lane: quick slot first, session model second, static last."""

    def _processor(self, no_warm_unused, model="claude-opus-5") -> VidscribeProcessor:
        processor = VidscribeProcessor(
            api_key="fake-key", model=model, json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.client = MagicMock()
        return processor

    def test_quick_slot_preferred(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.return_value = (
            _local_response(VALID_YAML)
        )

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.local_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "quick"
        assert frontmatter["title"] == "Test Talk on Neural Data"
        processor.client.messages.create.assert_not_called()

    def test_quick_failure_falls_back_to_session_model(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.return_value = _anthropic_response(
            VALID_YAML
        )

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-5"
        # Fixed-sampling model: temperature must be omitted
        assert "temperature" not in call_kwargs
        assert frontmatter["title"] == "Test Talk on Neural Data"

    def test_fallback_passes_temperature_when_supported(self, no_warm):
        processor = self._processor(no_warm, model="claude-haiku-4-5")
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.return_value = _anthropic_response(
            VALID_YAML
        )

        processor.generate_frontmatter("some transcript")

        call_kwargs = processor.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    def test_both_lanes_failing_uses_static_fallback(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.side_effect = RuntimeError("api down")

        frontmatter = processor.generate_frontmatter("some transcript")

        assert frontmatter["title"] == "Workshop Transcript"
